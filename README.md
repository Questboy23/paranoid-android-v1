# Paranoid Android v1

A fast, cheap monitor that flags AI agents that cheat a grader. It reads a saved agent run, turns it into word embeddings with a small free model (bge-small-en-v1.5), and adds up seven simple scores. No AI judge is needed.

**Paper:** [paper/paper.pdf](paper/paper.pdf) (6 pages, plain language, all benchmark results in Appendix A).
**One-page diagram:** [paper/tree.pdf](paper/tree.pdf).

## Results in short

| | Paranoid Android v1 | AI judge |
|---|---|---|
| Time per run | about 0.03 s on a laptop | about 23 s |
| One million runs | about 9 hours | about 266 days |
| Terminal Wrench, new jobs, caught at 5% false alarms | 72.9% to 81.3% | 44% (published GPT-5.4 judge) |
| Same kind of cheating, new jobs (AUC): Terminal Wrench / MALT / ImpossibleBench | 0.86 / 0.90 / 0.84 | |
| A benchmark it never trained on (AUC) | 0.49 to 0.71, almost nothing caught | 94% caught on ImpossibleBench |

It works well on new cases of the kinds of cheating it was trained on. It does not carry over to new kinds of misbehaviour, and the 81.3% is an upper bound (the design was chosen on the same runs). The paper lists every limit.

Version 2 adds an AI judge and rule-based checks on top of this layer.

## Files

| File | What it does |
|---|---|
| `build.py` | Builds `data/runs.jsonl` from Terminal Wrench (set `TW_TASKS` to the task folder). |
| `monitor.py` | Step cleaning, word counts, the meaning-bundle reader. |
| `run.py` | Embeds runs with bge-small and fine-tunes it. |
| `v4_crossfit.py` | Meaning-bundle reader, scored out-of-fold by job. |
| `v5_cache.py`, `v5_lab.py`, `v5_e2e.py` | Per-step embedding cache and the other readers. |
| `v5_combine.py` | Stacks the seven readers into one score. |
| `v5_fair.py`, `v5_fair_more.py`, `v5_fair_full.py` | Learning curve, new-agent and no-label tests. |
| `malt.py` | MALT test. |
| `cross_bench.py` | Train on one or two benchmarks, test on another. Uses the version 2 data loaders. |
| `judge.py` | AI-judge baseline. |
| `results/*.json` | Every number in the paper. |
| `paper/gen_paper.py` | Rebuilds the paper from `results/`: `python3 paper/gen_paper.py && cd paper && tectonic paper.tex` |

Every score comes from a model that never trained on that run's job (5 job groups, seed 0, `splits.json`).

## Data

Not included. Terminal Wrench (Apache-2.0), MALT (`metr-evals/malt-public`, gated), ImpossibleBench rollouts. Download them separately.

## Setup

```bash
python3 -m venv .venv && .venv/bin/pip install -r requirements.txt
```

## License

MIT
