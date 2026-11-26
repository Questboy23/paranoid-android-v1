"""Leave-one-benchmark-out test of the simple v1 reader: does training on more kinds of misbehaviour help on a benchmark
it never saw? Benchmarks: Terminal Wrench (grader gaming), MALT (sabotage, sandbagging, hard-coding, ...),
ImpossibleBench (passing tests that contradict the spec). Agent's words deleted everywhere.
Reader: mean and max of frozen bge-small step vectors, logistic regression; each training benchmark gets equal total weight.
Line for 'caught' is set on the TRAINING benchmarks' honest runs (what a deployment would have), and AUC needs no line.
Uses pa3 loaders and the frozen-encoder cache from ~/projects/paranoid-android-v3. Writes results/cross_bench.json."""
import json, os, sys, numpy as np
sys.path.insert(0, os.path.expanduser("~/projects/paranoid-android-v3"))
from pa3 import data as D, detectors as T
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_auc_score

B = {"tw": D.tw(), "malt": D.malt(), "ib": D.impossible()}
def feats(rows):
    out = []
    for r in rows:
        V = T.encode(T.act_units(r), "base"); out.append(np.r_[V.mean(0), V.max(0)])
    return np.array(out)
X = {k: feats(v) for k, v in B.items()}; Y = {k: np.array([r["label"] for r in v]) for k, v in B.items()}
for k in B: print(k, len(Y[k]), int(Y[k].sum()), flush=True)

def fit(train):
    Xt = np.vstack([X[k] for k in train]); yt = np.r_[tuple(Y[k] for k in train)]
    w = np.r_[tuple(np.full(len(Y[k]), 1 / len(Y[k])) for k in train)] * len(yt)
    return LogisticRegression(C=0.5, max_iter=5000, class_weight="balanced").fit(Xt, yt, sample_weight=w)

def boot_auc(y, s, n=1000, seed=0):
    rng = np.random.default_rng(seed); a = []
    for _ in range(n):
        i = rng.integers(0, len(y), len(y))
        if 0 < y[i].sum() < len(i): a.append(roc_auc_score(y[i], s[i]))
    return [round(float(np.percentile(a, 2.5)), 3), round(float(np.percentile(a, 97.5)), 3)]

res = {}
for test in B:
    for train in ([k for k in B if k != test][:1], [k for k in B if k != test][1:], [k for k in B if k != test]):
        m = fit(train); s = m.decision_function(X[test]); y = Y[test]
        line = np.quantile(np.r_[tuple(m.decision_function(X[k])[Y[k] == 0] for k in train)], 0.95)
        line_t = np.quantile(s[y == 0], 0.95)
        key = f"train {'+'.join(train)} -> test {test}"
        res[key] = dict(auc=round(float(roc_auc_score(y, s)), 3), auc_ci=boot_auc(y, s),
                        caught_line_from_train=round(float((s[y == 1] > line).mean()), 3),
                        false_alarms_line_from_train=round(float((s[y == 0] > line).mean()), 3),
                        caught_line_from_test_honest=round(float((s[y == 1] > line_t).mean()), 3))
        print(key, res[key], flush=True)
# same-benchmark reference: 5 folds by task inside each benchmark
for k in B:
    task = np.array([r["task"] for r in B[k]]); tasks = np.unique(task); s = np.zeros(len(task))
    for f in np.array_split(np.random.default_rng(0).permutation(tasks), 5):
        te = np.isin(task, f); m = LogisticRegression(C=0.5, max_iter=5000, class_weight="balanced").fit(X[k][~te], Y[k][~te])
        s[te] = m.decision_function(X[k][te])
    y = Y[k]; line = np.quantile(s[y == 0], 0.95)
    res[f"same benchmark {k} (new tasks)"] = dict(auc=round(float(roc_auc_score(y, s)), 3), auc_ci=boot_auc(y, s),
                                                  caught_line_from_test_honest=round(float((s[y == 1] > line).mean()), 3))
    print(k, "same", res[f"same benchmark {k} (new tasks)"], flush=True)
res["n"] = {k: [len(Y[k]), int(Y[k].sum())] for k in B}
json.dump(res, open("results/cross_bench.json", "w"), indent=1)
