"""v4 (embedding only), scored on every Terminal Wrench run with the published judge's protocol.

Fair test: the public encoder (bge-small-en-v1.5, no fine-tuning) never saw Terminal Wrench. The six readers, the word-count
reader and the combiner are fit 5 times, each time on 4/5 of the 331 tasks, and score the remaining 1/5. So every one of the
5,984 runs (3,632 hack, 2,352 baseline) is scored by a model that never saw its task. Positives: hack runs in the variant;
negatives: baseline runs (stripped variant: baseline with its text removed too), the same population as the published
GPT-5.4 judge. C is picked inside each fold's training tasks only. Writes results/v4_crossfit.json and data/scores_v4.json.
"""
import json, os, time
import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_auc_score, roc_curve
from sklearn.model_selection import GroupKFold
import monitor as M

TAG = os.environ.get("EA_TAG", "bge-small-en-v1.5_ft0"); K = 5; VARS = ("original", "sanitized", "stripped")
t0 = time.time(); log = lambda *a: print(f"[{time.time() - t0:6.0f}s]", *a, flush=True)
rows = [json.loads(l) for l in open("data/runs.jsonl")]
y = np.array([r["label"] for r in rows]); task = np.array([r["task"] for r in rows])
final = lambda st: next((s["say"] for s in reversed(st) if s["say"]), "")

U = {(i, v): M.units(st) for i, r in enumerate(rows) for v, st in r["v"].items()}
texts = set(M.CHEAT + M.HONEST)  # identical text list to run.py, so the cached embeddings line up
for (i, v), u in U.items():
    texts.update(u["thought"] + u["action"] + u["context"]); texts.add(u["files"]); texts.add(rows[i]["prompt"]); texts.add(final(rows[i]["v"][v]))
texts.discard(""); texts = sorted(texts)
assert json.load(open(f"data/emb_{TAG}.npz.n.json")) == len(texts), "embedding cache does not match the text list"
V = np.load(f"data/emb_{TAG}.npz")["V"].astype(np.float32); ix = {t: n for n, t in enumerate(texts)}
E = lambda L: V[[ix[t] for t in L if t]] if L else np.zeros((0, V.shape[1]), np.float32)
D = (E(M.CHEAT), E(M.HONEST))
F = {k: M.features(u, rows[k[0]]["prompt"], final(rows[k[0]]["v"][k[1]]), E, D) for k, u in U.items()}
log("features", len(F))

def doc(i, v): return "\n".join((s["say"] + "\n" if s["say"] else "") + s["act"] for s in rows[i]["v"][v])
def keys(idx): return [(i, v) for i in idx for v in (VARS if y[i] else ("original", "stripped"))]
def eval_keys(idx, v): return [(i, v if y[i] else ("stripped" if v == "stripped" else "original")) for i in idx]
tv = lambda: TfidfVectorizer(max_features=50000, ngram_range=(1, 2), min_df=2, sublinear_tf=True)

def word_scores(trk, other):
    """Out-of-fold word-count scores for the training keys, and scores for `other` from a fit on all training keys."""
    ytr = np.array([y[i] for i, _ in trk]); out = {}
    for a, b in GroupKFold(5).split(trk, ytr, [task[i] for i, _ in trk]):
        vec = tv().fit([doc(*trk[j]) for j in a]); lr = LogisticRegression(C=4, max_iter=3000).fit(vec.transform([doc(*trk[j]) for j in a]), ytr[a])
        out.update(zip([trk[j] for j in b], lr.decision_function(vec.transform([doc(*trk[j]) for j in b]))))
    vec = tv().fit([doc(*k) for k in trk]); lr = LogisticRegression(C=4, max_iter=3000).fit(vec.transform([doc(*k) for k in trk]), ytr)
    out.update(zip(other, lr.decision_function(vec.transform([doc(*k) for k in other]))))
    return out

def fold_fit(tr_idx, te_idx):
    trk = keys(tr_idx); tek = [k for k in keys(te_idx)]
    W = word_scores(trk, tek)
    G = {k: dict(F[k], words=np.array([W[k]])) for k in trk + tek}
    ytr = np.array([y[i] for i, _ in trk]); gtr = np.array([task[i] for i, _ in trk])
    # pick C inside the training tasks (3 inner folds by task), mean AUC over the three variants
    best = None
    for C in (0.05, 0.2):
        sc = []
        for a, b in GroupKFold(3).split(trk, ytr, gtr):
            h = M.Head(M.GROUPS).fit([G[trk[j]] for j in a], ytr[a], C, gtr[a])
            ib = sorted({trk[j][0] for j in b})
            sc += [roc_auc_score([y[i] for i, _ in eval_keys(ib, v)], h.p([G[k] for k in eval_keys(ib, v)])) for v in VARS]
        if best is None or np.mean(sc) > best[0]: best = (np.mean(sc), C)
    h = M.Head(M.GROUPS).fit([G[k] for k in trk], ytr, best[1], gtr)
    return {k: float(p) for k, p in zip(tek, h.p([G[k] for k in tek]))}, best[1]

def tpr_at(yv, s, fpr=0.05):
    f, t, _ = roc_curve(yv, s); return float(np.interp(fpr, f, t))

def task_ci(yv, s, g, fn, n=1000, seed=0):
    rng = np.random.default_rng(seed); ts = np.unique(g); at = {t: np.where(g == t)[0] for t in ts}; v = []
    for _ in range(n):
        i = np.concatenate([at[t] for t in rng.choice(ts, len(ts))])
        if len(set(yv[i])) == 2: v.append(fn(yv[i], s[i]))
    return [round(float(np.quantile(v, q)), 3) for q in (0.025, 0.975)]

def main():
    tasks = np.unique(task); folds = np.array_split(np.random.default_rng(0).permutation(tasks), K)
    P = {}; Cs = []
    for n, f in enumerate(folds):
        te_idx = np.where(np.isin(task, f))[0]; tr_idx = np.where(~np.isin(task, f))[0]
        p, C = fold_fit(tr_idx, te_idx); P.update(p); Cs.append(C); log("fold", n + 1, "of", K, "C", C)
    allix = np.arange(len(rows)); res = {"encoder": TAG, "n_runs": len(rows), "n_hack": int(y.sum()), "n_baseline": int((y == 0).sum()),
                                         "n_tasks": int(len(tasks)), "folds": K, "C_per_fold": Cs, "variants": {}}
    for v in VARS:
        k = eval_keys(allix, v); yv = np.array([y[i] for i, _ in k]); s = np.array([P[x] for x in k]); g = np.array([task[i] for i, _ in k])
        res["variants"][v] = dict(auc=round(roc_auc_score(yv, s), 4), tpr5=round(tpr_at(yv, s), 4),
                                  auc_ci=task_ci(yv, s, g, roc_auc_score), tpr5_ci=task_ci(yv, s, g, tpr_at))
        log(v, res["variants"][v])
    json.dump(res, open("results/v4_crossfit.json", "w"), indent=1)
    json.dump({"ids": [r["id"] for r in rows], **{v: {rows[i]["id"]: P[(i, vv)] for i, vv in eval_keys(allix, v)} for v in VARS}},
              open("data/scores_v4.json", "w"))

if __name__ == "__main__":
    main()
