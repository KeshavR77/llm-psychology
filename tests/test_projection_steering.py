"""CPU checks for directional ablation and experiment scheduling."""
import contextlib
import importlib
import io
import json
from pathlib import Path
import sys
import tempfile
import types
import unittest
from unittest.mock import patch

import torch
from steering import make_resid_project_hook, steering_hook


class ProjectionTests(unittest.TestCase):
    def test_geometry_and_scaling(self):
        h = torch.randn(2, 5, 7)
        d = torch.randn(7)
        v = d / d.norm()
        for beta in (0, .25, .5, .75, 1):
            out = make_resid_project_hook(d, beta)(None, None, h)
            torch.testing.assert_close(out @ v, (1-beta)*(h @ v), atol=1e-6, rtol=1e-5)
            torch.testing.assert_close(out - (out @ v)[..., None]*v,
                                       h - (h @ v)[..., None]*v, atol=1e-6, rtol=1e-5)
            torch.testing.assert_close(out, make_resid_project_hook(-3*d, beta)(None, None, h))

    def test_tuple_and_low_precision(self):
        for dtype in (torch.float32, torch.float16, torch.bfloat16):
            h = torch.randn(2, 3, 8).to(dtype)
            marker = object()
            out = make_resid_project_hook(torch.randn(8))(None, None, (h, marker))
            self.assertEqual(out[0].dtype, dtype)
            self.assertIs(out[1], marker)
            self.assertTrue(torch.isfinite(out[0]).all())

    def test_invalid_direction_and_beta(self):
        for d in (torch.zeros(4), torch.tensor([float('nan')]), torch.tensor([float('inf')])):
            with self.assertRaises(ValueError):
                make_resid_project_hook(d)
        for beta in (-1, 2, float('nan')):
            with self.assertRaises(ValueError):
                make_resid_project_hook(torch.ones(4), beta)

    def test_hook_cleanup_on_error(self):
        layer = torch.nn.Identity()
        model = types.SimpleNamespace(model=types.SimpleNamespace(layers=[layer]))
        with self.assertRaises(RuntimeError):
            with steering_hook(model, 0, torch.tensor([1., 0.]), 0, intervention='projection'):
                torch.testing.assert_close(layer(torch.ones(1, 2)), torch.tensor([[0., 1.]]))
                raise RuntimeError('test')
        self.assertFalse(layer._forward_hooks)

    def test_runners_schedule_and_dispatch(self):
        fake = types.ModuleType('multi_actor')
        fake.MODEL_IDS = {'qwen': 'fake'}
        fake.CausalLMRunner = lambda _: types.SimpleNamespace(model=types.SimpleNamespace(
            model=types.SimpleNamespace(layers=[torch.nn.Identity(), torch.nn.Identity()])))
        calls = []
        def run(*args, **kwargs):
            calls.append(kwargs)
            Path(kwargs['output_file']).write_text(json.dumps({'results': []}))
        fake.run_experiment = run
        with patch.dict(sys.modules, {'multi_actor': fake}):
            for name in ('run_steering', 'run_ablation'):
                module = importlib.import_module(name)
                with tempfile.TemporaryDirectory() as directory:
                    vectors = {0: torch.ones(4096), 1: torch.ones(4096)}
                    argv = [name, '--output-dir', directory, '--alphas', '-1', '1']
                    argv += ['--layers', '0', '1', '--vector-path', 'unused'] if name == 'run_steering' else ['--layer', '0']
                    calls.clear()
                    with patch.object(module, 'load_diffmeans_dict', return_value=vectors), patch.object(sys, 'argv', argv), contextlib.redirect_stdout(io.StringIO()):
                        module.main()
                        count = len(calls)
                        module.main()  # Existing projection outputs must be skipped too.
                    self.assertEqual(len(calls), count)
                    self.assertEqual(count, 22 if name == 'run_steering' else 7)
                    projections = list(Path(directory).glob('*projection*.json'))
                    self.assertEqual(len(projections), 16 if name == 'run_steering' else 4)
                    if name == 'run_steering':
                        self.assertEqual(sum(c.get('steering_intervention') == 'projection' for c in calls), 16)
                    else:
                        self.assertTrue(all(json.loads(p.read_text())['steering']['intervention'] == 'projection' for p in projections))


if __name__ == '__main__':
    unittest.main()
