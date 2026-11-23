"""Import a reproducibly shuffled copy of the full, unscreened MMLU test split."""

import argparse
import hashlib
import json
import random
import tempfile
from pathlib import Path
from urllib.request import urlretrieve

import pyarrow.parquet as pq

REVISION = "c30699e8356da336a370243923dbaf21066bb9fe"
URL = f"https://huggingface.co/datasets/cais/mmlu/resolve/{REVISION}/all/test-00000-of-00001.parquet"
SHA256 = "74a41822ce7d3def56e1682f958469c04642a5336a5ce912fa375fdb90fb25d7"


def import_rows(parquet_path):
    if hashlib.sha256(parquet_path.read_bytes()).hexdigest() != SHA256:
        raise ValueError("MMLU source checksum does not match the pinned revision")
    rows = pq.read_table(parquet_path).to_pylist()
    random.Random(42).shuffle(rows)
    return rows


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, default=Path("data/mmlu_test.json"))
    parser.add_argument("--parquet", type=Path, help="Use a previously downloaded upstream parquet file")
    args = parser.parse_args()
    if args.parquet:
        rows = import_rows(args.parquet)
    else:
        with tempfile.TemporaryDirectory() as directory:
            source = Path(directory) / "test.parquet"
            urlretrieve(URL, source)
            rows = import_rows(source)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(rows, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    print(f"Imported {len(rows)} unscreened MMLU questions to {args.output}")


if __name__ == "__main__":
    main()
