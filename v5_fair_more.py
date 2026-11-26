"""Learning curve past 50 labelled tasks, simple reader from v5_fair.py (step mean/max embeddings + word counts, logistic
regression). k labelled tasks picked at random (10 draws), scored on all other tasks. k = 265 is the 5-fold cross-fit.
Writes results/v5_fair_more.json."""
import json, pickle, warnings, numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_curve
from scipy.sparse import hstack, csr_matrix
warnings.filterwarnings("ignore")
D = pickle.load(open("data/v5_steps.pkl", "rb")); S, y, task = D["steps"], D["y"], D["task"]; N = len(y)
rows = [json.loads(l) for l in open("data/runs.jsonl")]
def tpr(yv, s): f, t, _ = roc_curve(yv, s); return float(np.interp(0.05, f, t))
act = [S[(i, "stripped")][:, :384].astype(np.float32) for i in range(N)]
X_emb = np.stack([np.r_[a.mean(0), a.max(0)] for a in act])
docs = ["\n".join(s["act"] for s in r["v"]["stripped"]) for r in rows]
def fit_pred(tr, te):
    v = TfidfVectorizer(max_features=50000, ngram_range=(1, 2), min_df=2, sublinear_tf=True).fit([docs[i] for i in tr])
    Xs = lambda idx: hstack([csr_matrix(X_emb[idx]), v.transform([docs[i] for i in idx])]).tocsr()
    return LogisticRegression(C=0.5, max_iter=3000).fit(Xs(tr), y[tr]).decision_function(Xs(te))
tasks = np.unique(task); rng = np.random.default_rng(1); res = {}
for k in (100, 150, 200):
    out = []
    for _ in range(10):
        pick = rng.choice(tasks, k, replace=False); tr = np.where(np.isin(task, pick))[0]; te = np.where(~np.isin(task, pick))[0]
        out.append(tpr(y[te], fit_pred(tr, te)))
    res[k] = dict(caught_median=round(float(np.median(out)), 4), caught_min=round(float(min(out)), 4), draws=10)
    print(k, res[k], flush=True); json.dump(res, open("results/v5_fair_more.json", "w"), indent=1)
FOLDS = np.array_split(np.random.default_rng(0).permutation(tasks), 5); s = np.zeros(N)
for f in FOLDS:
    te = np.where(np.isin(task, f))[0]; tr = np.where(~np.isin(task, f))[0]; s[te] = fit_pred(tr, te)
res[265] = dict(caught=round(tpr(y, s), 4), note="5-fold cross-fit, about 265 tasks labelled per fold"); print(265, res[265], flush=True)
json.dump(res, open("results/v5_fair_more.json", "w"), indent=1)
