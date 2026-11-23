# Normative conformity: literature and experimental implications

Research date: 2026-11-16. Scope: primary sources relevant to separating approval-related response shifts, informational updating, and repeated-answer effects in Qwen3-8B. Literature findings and proposed experiments are separated below. No model experiments were run for these notes.

## Main implication

Public/private conformity comparisons and normative/informational activation-vector comparisons already exist. The strongest proposed extension is **selective causal control of an approval-related response effect while preserving useful evidence updating**, with repetition, source dependence, and prompt-family generalization measured explicitly. Behavioral and activation evidence can support this operational mechanism; it cannot establish human-like social motivation.

## Primary-source findings

### Deutsch and Gerard: different causes can produce the same answer

The original article is Deutsch & Gerard (1955), *A study of normative and informational social influences upon individual judgment*, DOI [10.1037/h0046408](https://doi.org/10.1037/h0046408). The [APA-authored abstract reproduced by EBSCO](https://openurl.ebsco.com/contentitem/pdh%3A1957-02366-001?crl=c&id=ebsco%3Apdh%3A1957-02366-001&jrnl=0096851X&sid=ebsco%3Aplink%3Acrawler-pdh) describes face-to-face, anonymous, group, and commitment conditions. Importantly, anonymous participants still made more errors under exposure to others' judgments than participants judging alone. Therefore removing public exposure does not remove every influence of others.

In [Deutsch's own retrospective](https://garfield.library.upenn.edu/classics1980/A1980KF12300001.pdf), he explains the intended distinction: normative influence is associated with group expectations, whereas informational influence does not require a group. He also emphasizes that group norms can support independent judgment. This motivates a reversal control: vary whether the group approves agreement or approves accurate independent reporting, holding the factual claims fixed.

Access limitation: the original English full text was not retrievable through APA/DOI/archived-PDF links during this search. These historical claims are grounded in the article's APA abstract and an author's retrospective, not an independently inspected original PDF. Avoid attributing unverified numerical effects or precise procedural details to the original.

### Asch: unanimity is separable from the number agreeing

In his primary report [*Opinions and Social Pressure* (1955)](https://web.mit.edu/curhan/www/docs/Articles/15341_Readings/Influence_Compliance/Asch_1955_Opinions_and_social_pressure.pdf), Asch reports that a truthful partner reduced yielding substantially; a dissenter who also disagreed with the subject could reduce yielding too. A partner joining the majority differed from simply leaving. Thus unanimity, support for the correct answer, and group size should not be treated as interchangeable manipulations. The report also describes participants' uncertainty and heterogeneous reactions, which makes “the answer was initially correct” insufficient to identify one psychological cause.

### Bito et al. 2026: closest normative/informational mechanistic prior work

[*Large Language Models Exhibit Normative Conformity*, arXiv:2604.19301v1](https://arxiv.org/html/2604.19301v1), §§3–4, studies six models using binary majority voting, including fruit-display preferences without an objectively correct answer. Initial private preferences are supplied in the prompt. It varies named/public versus anonymous voting, subsequent evaluation, continuing relationships, and peers possessing additional room information. Options are swapped; generations are repeated 120 times per condition. Four models increase conformity under publicness; evaluation or relationship context affects additional models.

For Llama-3.1-70B-Instruct-AWQ, §3.4 computes per-layer residual differences between with/without publicness and between with/without informational influence, immediately before generating the first token. It compares cosine similarities; §4.3 reports different shallow-layer directions and greater similarity near output. The reported internal analysis is geometric; it does not provide selective steering or mediation tests.

Assessment: publicness effects plus direction comparisons are already prior art. Supplied preferences and aesthetics do not uniquely identify approval seeking: coordination, demand following, and new context are alternative explanations. Qwen3-8B causal selectivity across independently manipulated evidence and approval is a more defensible next question.

### Bellina et al. 2026: paired paraphrases already used for public/private effects

[*Conformity and Social Impact on AI Agents*, arXiv:2601.05384v1](https://arxiv.org/html/2601.05384v1), §4.1.5, compares private and public responses on visual tasks using initially correct items. It uses 100 paired paraphrased visibility statements and defines a normative score as the difference between public and private conformity-curve AUCs. The authors report a positive effect for most models. This is directly relevant to wording robustness and prevents claiming that public/private template variation is novel. A public-minus-private score remains an operational visibility effect; calling the private condition purely informational requires further assumptions.

### Hu and Qu 2026: removing speakers does not remove answer influence

[*Most LLM Conformity Needs No Speaker*, arXiv:2607.05545v1](https://arxiv.org/html/2607.05545v1), studies six open-weight models and seven datasets. It reports 66.5% harmful revision under source-free wrong assertions versus 10.3% under a plain re-ask, with expert framing adding an increment. It distinguishes repetition count from distinct-speaker count, and includes paraphrase, open-ended, source-noun, position, and invalid-label controls. The paper explicitly defines “speaker-free” as removing explicit attribution, while allowing the text to function as evidence. Its absolute rates should not be transferred to this repository's different setup.

### Earlier sources already cited in this repository

- [Zhu et al., ACL 2025](https://aclanthology.org/2025.acl-long.195/) finds conformity across tasks/models and increased susceptibility with uncertainty; majority tone also matters. This supports measuring uncertainty and wording dependence, without making them diagnostic of approval seeking.
- [Weng et al., ICLR 2025](https://openreview.net/pdf/cefcfec072e2db0f4b2196da7ea26a99d12a19a7.pdf) introduces BenchForm with five interaction protocols, studies majority size and interaction time, and tests persona/reflection mitigation. It establishes behavioral majority following, rather than isolating the approval mechanism proposed here.
- [Zhong et al., arXiv:2508.14918v1](https://arxiv.org/html/2508.14918v1), §6, uses private/public signals with stated reliability and a Bayesian comparison. Public-signal overweighting is interpreted as normative-like. Caution: overweighting alone can reflect dependence mistakes or evidence-processing errors. Reliability is varied between different domain scenarios, so uncertainty and domain are not fully crossed; the proposed Qwen experiment should cross them within each domain.
- [Vennemeyer et al., arXiv:2509.21305v1](https://arxiv.org/html/2509.21305v1), §§4–6, distinguishes sycophantic agreement, genuine agreement, and praise using geometry and selective held-out steering. This provides a useful methodological precedent for an intervention-by-behavior matrix. It does not establish that peer conformity shares those mechanisms.

## Proposed operational definitions and experiment

The following is a proposed design, not a result of the cited studies.

Let `m` be the majority target, `P` the approval/publicness manipulation, `E` the factual evidence and its known reliability/dependence, `R` the number of repeated assertions, and `T` the template family. Measure both target adoption and `s = log p(m) - log p(alternative)` at a shared answer slot. Use complete answer likelihoods if options span multiple tokens.

1. **Repetition component:** vary `R` in speaker-free assertions, with matched question, target, position, and output format. Compare with irrelevant repeated text and no assertions. This is an assertion/repetition contrast, not an estimate of every non-social effect.
2. **Source-count component:** compare one source repeated with multiple explicitly independent sources, while holding assertion count fixed. Add a condition where multiple sources explicitly copied the same source. Independence creates legitimate new evidence, so a distinct-source effect alone is not normative conformity.
3. **Evidence component:** supply synthetic observations with a known generative model. Cross source accuracy, private-signal accuracy, independence/copying, and whether evidence supports the gold answer or a distractor. This supplies beneficial correction trials as well as harmful ones. Do not rely entirely on remembered MMLU facts to define informational updating.
4. **Approval component:** hold all evidence and assertions fixed; vary whether peers can see the response and whether they approve agreement, independence, or accuracy. Approval direction should reverse across otherwise matched trials. A nonsocial evaluator/administrative recording control distinguishes generic evaluation from peer approval.

Estimate the paired approval effect `E[s(P=1,E,R,T)-s(P=0,E,R,T)]` within evidence and repetition cells, including interactions. A group-versus-nonsocial-evaluator difference-in-differences is useful if the matched contexts support its assumptions. Report visibility, explicit approval contingency, and continuing-relationship effects separately before combining them. Publicness alone need not create an effect, and explicit commands to agree primarily test instruction following.

Use several independently written template families. Keep all counterfactual versions of an item within its item split, and hold out whole template families and item identities for evaluation. For large paraphrase sets, do not treat item×template combinations as independent samples; use crossed item/template effects or resample both units. Counterbalance target letter, option order, source names, and approval direction. Share question, answer schema, and final answer cue across every contrast, and report sensitivity to wrappers and token length instead of assuming token matching establishes semantic equivalence.

## Proposed causal intervention and interpretation

Fit approval contrasts by paired condition differences averaged over balanced evidence, repetition, answer identities, and templates. Also fit evidence, repetition, source-count, and answer-identity contrasts. Estimate a nuisance *subspace* from multiple training contrasts rather than assuming one informational vector exhausts informational processing.

An exploratory orthogonalized candidate is `v_N_perp = (I - U_E U_E^T) v_N` for orthonormal evidence basis `U_E`. Add repetition/format bases only when justified by training data. This guarantees geometric orthogonality under the chosen metric, not functional independence. Compare the original vector, projected candidate, the removed component, and norm-matched random controls on held-out data. A null after projection may indicate shared computation or an inadequate basis, rather than the absence of an approval mechanism.

[LEACE](https://proceedings.neurips.cc/paper_files/paper/2023/hash/d066d21c619d0a78c5b557fa3291a8f4-Abstract-Conference.html) provides a stronger linear erasure guarantee than deleting one selected vector: it prevents linear classifiers from recovering the labeled concept under its assumptions while minimizing an appropriate representation change. It does not guarantee semantic or causal removal from a nonlinear network; later layers can reconstruct information. Use it as an additional intervention with held-out probe retraining and behavioral validation, not as a declaration that “informational conformity was removed.”

Evaluate an intervention-by-outcome matrix: approval effect, evidence-reliability sensitivity, beneficial correction, harmful adoption, repetition effect, no-context accuracy, and format validity. Prefer absolute changes with uncertainty intervals over a selectivity ratio alone. Bidirectional steering supports causal sufficiency for shifting the operational behavior. Conditional projection removal and matched activation patching provide additional evidence relevant to necessity and mediation; none alone identifies a unique complete circuit.

## Repository implications supplied by the concurrent code audit

The README already labels multiple-participant-minus-control DiffMeans as descriptive attribution/distinct-source contrasts, appropriately avoiding a pure-conformity claim. The root agent's audit reports seven-line target adoption of .77 for multiple speakers, .18 for one repeated speaker, .97 for speaker-free XML, .86 for `Answer: B`, and .84 for `Answer: <answer>B</answer>` across the fixed 100 items. These are repository results supplied by the audit, not independently rerun literature results. Their wrapper sensitivity makes a universal social decomposition unsupported.

The same audit reports chat-template extraction with thinking enabled versus raw-prompt behavioral/steering inputs; an `assistant_start` position at the beginning of the header; additive hooks broadcast across token positions; and constant-vector subtraction called ablation. Before causal interpretation, align encoding, exact site, and decoding between extraction and evaluation. Separate prefill-only, answer-slot-only, and generation-time interventions. Constant subtraction is an additive intervention, whereas projection erasure removes each activation's component. Activation-oracle descriptions are hypotheses, not evidence of motive.

## Claim that the proposed evidence could support

“Qwen3-8B exhibits a reproducible response component sensitive to peer-approval cues, beyond specified evidence and repetition manipulations. A targeted activation intervention selectively changes this component while preserving measured evidence use across held-out items and prompt families.”

This claim is contingent on the proposed validation succeeding. It is narrower than a claim of human-like normative motivation or a uniquely isolated conformity circuit.
