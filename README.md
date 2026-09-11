# Reinforcement Learning for Molecular Lead Optimization
[![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/farabhi/rl-lead-optimization/blob/main/rl_lead_optimization.ipynb)

*A reinforcement-learning experiment in computational drug design: can an RL agent learn to improve a candidate drug molecule? This iteration of the experiment says "not yet" — and why it's so is the most interesting part.*

## Introduction

Molecular lead optimization is implemented in the notebook as this: given a known inhibitor of EGFR (a protein that real cancer drugs are designed to block), improve it by adding functional groups one at a time. The modification is guided by a scoring function and an RL policy. Five policies — Random, ε-greedy, triggered ε-greedy, UCB, and Thompson Sampling — try to come up with rules of successful modifications and are different in how they treat immediate reward vs exploration. Each policy has thousands of molecules to experiment with and each policy repeats the experiment several times (exact number depends on configuration). The pool of initial molecules comes from ChEMBL, a public database of measured drug-activity data. Built molecules are evaluated among other things by a QSAR model — a machine-learning model, also built in the notebook, that predicts how strongly a molecule will block the EGFR target.

## What this project is and is not

This is a preliminary study whose main output is a precise account of what has to be true before RL policies can compete with QSAR-like screening and random molecule building.

The experiment in a nutshell is sequential decision-making, not one-shot molecular generation. It is not a competitor to AlphaFold or transformer-based generators, and not a scaled-down attempt at one — those learn chemical representations from millions of compounds and use them to generate a new compound in its final form; this asks whether, given one molecule, a policy can learn which local sequential edits to make to improve it. Tabular Q-learning with linear function approximation is a deliberate choice: the weight matrix is directly inspectable, as opposed to billions of uninterpretable parameters of a black-box LLM. Exploration in this experiment is explicit and tunable, and the whole thing runs on free-tier hardware in 1–3 hours, depending on the experiment's configuration.

## Findings

On the full set of molecules each method retained, QSAR screening and Random modification are indistinguishable — mean score 14.86 against 14.79, closer to each other than any pair of policies. The four learned policies sit slightly but consistently below both, in the range 14.30 to 14.50. No learned policy beats Random anywhere in this comparison, and this holds across hardware, episode budgets, and seed counts.

Top-50 molecules ("right tail") tells a more diverse story. Random might be beaten by learned policies — all of them, some of them, or none of them. The inconsistency is shown to be a mix of reproducibility bug (see below), hardware peculiarities (AVX512 instruction set presence/absence), and policies' sensitivities to what molecules are present in seeds and in what sequence. Top-50 demonstrates a second invariant — Random always beats QSAR there. The fact that randomly attaching functional groups always beats the initial pool, but rule-based attachment is not guaranteed to do so, is the most contentious finding of this version of the notebook. A clear explanation of why it happens is the primary interest for the next iteration of the notebook.

The policies did learn — their weight matrices show distinct, interpretable functional-group preferences. Learning happens; it just doesn't translate into a competitive advantage, for a structural reason given in the next sections.

## Reproducibility

Runs are deterministic on any single machine and not portable across machines with different CPU SIMD support. This turned out to have two distinct causes of very different character, and separating them is the methodological core of the study.

**A)** The first cause of inconsistency was traced back to how the initial pool of EGFR inhibitors is created from the ChEMBL dataset. The pool is built by sorting molecules on measured activity (pchembl score) and then taking a positional slice that cuts off the ends of the score distribution and keeps only mid-scoring compounds. But activity values are 90% tied, and the sort left tied rows in a machine-dependent order. On CPUs with AVX-512, NumPy dispatches a different sort routine than on CPUs without it, ordering the score ties differently, admitting different molecules into the pool, and shifting the desirable threshold from 18.29 to 18.12. This alone moved the apparent success rate from 8/15 to 14/15 on identical data. The sort is deterministic on any single machine, so the bug was invisible until the same code was run across six runtimes. Fix was introduced in later versions: sort on a total key. *(Written up separately at https://farabhi.github.io/2026/09/03/same-code-same-seed-different-answer.html  — it generalizes well beyond chemistry; it's the SQL `ORDER BY … LIMIT` pagination bug in another domain.)*

The fix turned out to be more robust than the defect required. A later run on a different CPU (AMD without AVX-512) under a different NumPy minor version and a different Python minor version reproduced the seed pool exactly: same pool size, same slice hash, same 90th-percentile cutoff, same count of improving steps, same median gain, same derived threshold. The pool now reproduces across every environment it has been tried in, while the experiment reproduces across nothing but an identical host. In that same run Random reproduced bit-for-bit — identical crossing episodes, identical best scores — while every learned policy diverged completely. Random is the control here: it has no weight matrix and no argmax, so it has nothing for a last-bit difference to amplify. (One honest caveat: hardware and software versions moved together in that run, so it corroborates the mechanism rather than isolating AVX-512 as the cause. The six-runtime comparison, where the software stack was pinned, remains the clean evidence for that.)

**B)** Fixing the sort defect made the pool reproducible; it did not make the experiment reproducible. The reinforcement-learning loop is a feedback process, and it amplifies last-bit floating-point differences between runtimes with the AVX512 instruction set present vs the ones where it's absent. A one-ULP gap between two action-values flips an argmax, which changes the action, which changes the learning trajectory of a policy in a given seed permanently. Everything upstream of the loop reproduces; nothing inside it does.

For example, on identical pools and identical seeds, changing only the CPU moved the UCB policy from not significantly better than Random (p≈0.10) to significantly better (p≈0.02). A one-ULP difference in the input flipped a significance verdict in the output. It's important to understand that these inconsistencies make top-50 results inconclusive, not false. Molecules generated by learned policies in winning runs are as legitimate as the ones from failed runs. The instabilities show rather that the constraints in this version of the experiment determine evaluation results, not just facilitate them. Understanding the constraints is the first step to changing them so that they can do the job they were intended for.

Another way to look at these two causes and understand their role as constraints is to distinguish their effect on learned-policy results. Sort inconsistency harbors a small, directional nudge. A total sort key clusters structurally similar molecules, raising the structural turnover of the random episode sequence, which introduced a "whiplash" effect for learned policies. Equivalently, the first, naive version of sort broke reproducibility but was stacked in a small, but detectable favor for learned policies. The measurement was done with Mann Whitney test - approximately 54% of pchamble/total sorting pairs showed higher structural turnover for total sort. 

ULP difference between runtimes is a significant, but non-directional variance source. AVX-512 arithmetic isn't worse, it's different in the last bit; the argmax flips it causes send trajectories to better or worse basins with no bias. It destabilizes the verdict; it does not push it one way.

The notebook prints an environment fingerprint (library versions, CPU SIMD support) and the pool-integrity hash on every run, so a reader can confirm they're on a comparable machine before trusting cross-run numbers, and so any comparison across the AVX-512 boundary is flagged rather than silently trusted.

If you are trying to match a published number, you need to be aware that nothing selectable in Colab pins the hardware. The CPU runtime returns whatever host the scheduler allocates, and AVX-512 support varies across those hosts. The fingerprint is therefore the only ground truth for whether two runs are comparable.

## The architectural constraints

Another reason for the inconclusive performance of learned policies over Random modification and QSAR is an architectural constraint that underlies the experiment.

Each experiment consists of hundreds of episodes — hundreds of molecules from the initial pool. For each molecule the experiment attaches a functional group and has 15 attempts, or steps, to do it. First and foremost, the experiment doesn't control what molecule it will receive next. They are dispensed by a random mechanism. Moreover, each step involves two decisions:

- **What group to add** — the only decision the policy controls.
- **Where to attach the new group** — chosen at random among available molecule parts.

Obviously, this non-determinism in important chemical context introduces another instability for learned policies. However, in order to see why the latter is so detrimental for learned policies and not so much for Random (it also doesn't control where the functional group lands on the molecule), the step-related constraints should be viewed in the light of the architectural decision that underlies the whole experiment in this version of the notebook. A single linear weight matrix maps molecular fingerprint to action-values across every scaffold family in the pool at once. When different scaffolds reward different edits, one matrix cannot represent both — it averages them, which is visible directly in the learned weights (all action-values end up negative; the policy picks the least-bad option rather than a good one). So even the one decision a policy does control is made through a representation too weak to separate one chemical context from another. The representational limit is established; whether adding decision authority on top of it would help is an open question, taken up at the end.


## What this motivates

The core limitation of this experiment is representational: a single linear weight matrix is asked to serve every scaffold family in the pool at once, and it cannot — it averages preferences that should be distinct, and the averaged result favors no action strongly (every learned value ends up negative; the policy picks the least-bad option). So the follow-up has to give the model a way to distinguish scaffold contexts and hold different value estimates for each.

A smoother reward oracle (gradient-boosted or Gaussian-process rather than a piecewise-constant forest) is a separate, cleaner improvement worth folding in regardless: the flat reward landscape both weakens learning and, by keeping action-values near-tied, feeds the reproducibility sensitivity described above.

Also, there should be a way for learned policies to control where to attach the next functional group. Finally, the metrics for performance evaluation should not be just top-50 molecules. Obviously, general population cannot be assigned this role either. Instead, the demonstrated generalization over different scaffold families producing high-scoring molecules should be considered a real sign of learned policies crystallizing proper chemical rules. The next version of the notebook will implement these, or will explain why they can't or shouldn't be built.

## Running it

The fastest route is the Colab badge at the top — it opens the notebook in a fresh runtime, and the first cell installs what it needs.

Locally:

```bash
git clone https://github.com/farabhi/rl-lead-optimization.git
cd rl-lead-optimization
pip install -r requirements.txt
jupyter notebook rl_lead_optimization.ipynb
```

Run the cells in order. A full run at the published settings — five policies × three seeds × 1000 episodes — takes roughly 2.5 hours on a Colab CPU runtime. The accelerator is never used as there is no torch, jax or CUDA in the notebook. The whole experiment runs in NumPy, scikit-learn and RDKit. Selecting a GPU or TPU runtime can still be noticeably faster, since Colab allocates a different host VM whose CPU is quicker — not because any computation moves to the accelerator.

Two checkpoints tell you the pipeline is intact before the long run starts: the pool-integrity cell should report `PASS`, and the dynamic-threshold cell should report a 2,853-compound pool and a threshold of 18.08.


## Data

The dataset ships with the repository, in `data/`. It is a snapshot of **ChEMBL_36** (July 2025 release), fetched March 2026 — see `data/egfr_chembl.provenance.json` for the full provenance note.

It is a committed artifact for a very specific reason — reproducibility. The 6,000-compound working set is drawn with `df_full.sample()`, which samples **by position**. Any new ChEMBL release will change the initial dataset, data sample and the exact results.

The notebook falls back to a live fetch if the file is missing, and records the release it received alongside the data when it does.


## Repository contents

```
rl_lead_optimization.ipynb        the experiment, end to end
data/egfr_chembl.parquet          ChEMBL_36 snapshot (336 KB)
data/egfr_chembl.provenance.json  which release, fetched when, and why committed
requirements.txt                  pinned versions from the run that produced these results
CITATION.cff                      citation metadata
LICENSE                           MIT
```

This is release **v1.0** — the preliminary study described above. The follow-up outlined in *What this motivates* will be developed in this repository as a later version, so v1.0 remains a stable reference point.

## Writing

- [Same code, same seed, different answer](https://farabhi.github.io/2026/09/03/same-code-same-seed-different-answer.html) — the sort defect and what it cost, without the chemistry
- [Taking apart a result that (never) worked](https://farabhi.github.io/2026/09/10/taking-apart-a-result-that-worked.html) — how the result came apart, and what was actually producing it

## License and data

Code in this repository is MIT licensed (see `LICENSE`).

The dataset in `data/` is derived from ChEMBL and is separately licensed **CC BY-SA 3.0** (© EMBL-EBI). The code license does not extend to it.
