"""v5 experiments on the same 5 task folds as v4. Every score is out-of-fold by task.
Components (each writes OOF scores for all three variants into data/v5_oof.npz):
  mil   : attention reader over the sequence of step embeddings (multiple-instance learning: a run is a bag of steps,
          a cheat is often 1 to 3 of them). Optional padding augmentation (honest steps inserted into cheat runs).
  canon : char n-gram word counts over commands with numbers, hashes and quoted strings replaced (plain, neutral form).
  knn   : mean similarity of a run's top steps to its nearest cheat-run steps minus nearest honest-run steps (training bank).
Usage: python v5_lab.py mil|mil_aug|canon|knn"""
import pickle, re, sys, json
import numpy as np
from sklearn.metrics import roc_auc_score, roc_curve

# pickle/npz files here are written by v5_cache.py and this script on this machine only
D = pickle.load(open("data/v5_steps.pkl", "rb"))
S, P, y, task, model = D["steps"], D["prompt"], D["y"], D["task"], D["model"]
VARS = ("original", "sanitized", "stripped"); N = len(y)
tasks = np.unique(task); FOLDS = np.array_split(np.random.default_rng(0).permutation(tasks), 5)  # same as v4

def keys(idx): return [(i, v) for i in idx for v in (VARS if y[i] else ("original", "stripped"))]
def eval_key(i, v): return (i, v if y[i] else ("stripped" if v == "stripped" else "original"))
def tpr_at(yv, s, fpr=0.05):
    f, t, _ = roc_curve(yv, s); return float(np.interp(fpr, f, t))
def report(name, O):
    out = {}
    for v in VARS:
        yv = np.array([y[i] for i in range(N)]); s = O[v]
        if v not in O: continue
        out[v] = (round(roc_auc_score(yv, s), 4), round(tpr_at(yv, s), 4))
    print(name, out, flush=True); return out

PADP = {}
import os; PAD_SAME = os.environ.get("PAD_SAME") == "1"
def run_folds(fit_predict):
    O = {v: np.zeros(N) for v in VARS}
    for f in FOLDS:
        te = np.where(np.isin(task, f))[0]; tr = np.where(~np.isin(task, f))[0]
        pred = fit_predict(keys(tr), [eval_key(i, v) for i in te for v in VARS]); print("fold done", flush=True)
        for i in te:
            for v in VARS: O[v][i] = pred[eval_key(i, v)]
    return O

# ---------------- MIL attention reader ----------------
def mil(aug=False, seeds=1, epochs=10, maxlen=40, task_film=False):
    import torch, torch.nn as nn
    torch.set_num_threads(3)
    dim = next(iter(S.values())).shape[1]
    class Net(nn.Module):
        def __init__(s):
            super().__init__()
            s.step = nn.Sequential(nn.Linear(dim + 2, 256), nn.GELU(), nn.Dropout(0.2), nn.Linear(256, 128), nn.GELU())
            s.att = nn.Linear(128, 1); s.pr = nn.Linear(384, 32); s.film = nn.Linear(384, 256)
            s.out = nn.Sequential(nn.Linear(128 * 2 + 32 + 1, 64), nn.GELU(), nn.Dropout(0.2), nn.Linear(64, 1))
        def forward(s, X, M, Pr, L):
            h = s.step(X) if not task_film else s.step[3:](s.step[:3](X) * (1 + torch.tanh(s.film(Pr))).unsqueeze(1))  # each step read in light of the task
            a = s.att(h).squeeze(-1).masked_fill(~M, -1e4).softmax(1)
            att = (a.unsqueeze(-1) * h).sum(1); mx = h.masked_fill(~M.unsqueeze(-1), -1e4).max(1).values
            return s.out(torch.cat([att, mx, s.pr(Pr), torch.log1p(L).unsqueeze(-1)], 1)).squeeze(-1)
    def batch(ks, extra=None):
        mats = [S[k][:maxlen].astype(np.float32) for k in ks] if extra is None else extra
        L = max(len(m) for m in mats); B = len(mats)
        X = np.zeros((B, L, dim + 2), np.float32); Mk = np.zeros((B, L), bool)
        for b, m in enumerate(mats):
            n = len(m); X[b, :n, :dim] = m; X[b, :n, dim] = np.arange(n) / max(n - 1, 1); X[b, :n, dim + 1] = n == np.arange(1, n + 1)
            Mk[b, :n] = True
        Pr = np.stack([P[k[0]].astype(np.float32) for k in ks]); Ln = np.array([len(m) for m in mats], np.float32)
        return [torch.from_numpy(a) for a in (X, Mk, Pr, Ln)]
    def fp(trk, tek):
        rng = np.random.default_rng(0); ytr = torch.tensor([float(y[i]) for i, _ in trk])
        hon = [k for k in trk if y[k[0]] == 0]
        preds = np.zeros(len(tek))
        for seed in range(seeds):
            torch.manual_seed(seed); net = Net(); opt = torch.optim.AdamW(net.parameters(), 1e-3, weight_decay=1e-3)
            for ep in range(epochs):
                net.train()
                for b in np.array_split(rng.permutation(len(trk)), max(1, len(trk) // 64)):
                    ks = [trk[j] for j in b]; mats = None
                    if aug:  # padding attack as training data: insert 1 to 6 honest steps from other honest runs
                        mats = []
                        for k in ks:
                            m = S[k][:maxlen].astype(np.float32)
                            if y[k[0]] == 1 and rng.random() < 0.5:
                                h = S[hon[rng.integers(len(hon))]].astype(np.float32); h = h[rng.choice(len(h), min(len(h), rng.integers(1, 7)), replace=False)]
                                pos = rng.integers(0, len(m) + 1); m = np.r_[m[:pos], h, m[pos:]][:maxlen]
                            mats.append(m)
                    X, Mk, Pr, Ln = batch(ks, mats)
                    loss = nn.functional.binary_cross_entropy_with_logits(net(X, Mk, Pr, Ln), ytr[b])
                    opt.zero_grad(); loss.backward(); opt.step()
            net.eval()
            with torch.no_grad():
                for b in np.array_split(np.arange(len(tek)), max(1, len(tek) // 256)):
                    preds[b] += net(*batch([tek[j] for j in b])).numpy() / seeds
                # padding attack (as in v2): 5 honest steps from a training honest run put in front of each stripped run
                prng = np.random.default_rng(1); hs = [k for k in hon if k[1] == "stripped"]
                pk = [k for k in tek if k[1] == "stripped" and y[k[0]] == 1]  # attacker pads cheats; honest runs unchanged
                same = {}
                for k in tek:
                    if y[k[0]] == 0 and k[1] == "stripped": same.setdefault(task[k[0]], []).append(k)
                src = lambda k: (same[task[k[0]]][prng.integers(len(same[task[k[0]]]))] if PAD_SAME and task[k[0]] in same else hs[prng.integers(len(hs))])
                pm = [np.r_[S[src(k)][:5].astype(np.float32), S[k].astype(np.float32)][:maxlen] for k in pk]  # PAD_SAME: honest steps from the same task
                for b in np.array_split(np.arange(len(pk)), max(1, len(pk) // 256)):
                    pp = net(*batch([pk[j] for j in b], [pm[j] for j in b])).numpy() / seeds
                    for j, q in zip(b, pp): PADP[pk[j]] = PADP.get(pk[j], 0) + q
        return dict(zip(tek, preds))
    return fp

# ---------------- canonical command word counts ----------------
def canon_fp(trk, tek):
    from sklearn.feature_extraction.text import TfidfVectorizer
    from sklearn.linear_model import LogisticRegression
    vec = TfidfVectorizer(analyzer="char_wb", ngram_range=(3, 4), min_df=3, max_features=100000, sublinear_tf=True)
    X = vec.fit_transform([CANON[k] for k in trk]); lr = LogisticRegression(C=2, solver="liblinear").fit(X, [y[i] for i, _ in trk])
    return dict(zip(tek, lr.decision_function(vec.transform([CANON[k] for k in tek]))))
def canon(cmd):
    c = cmd.lower()
    c = re.sub(r"'[^'\n]{20,}'|\"[^\"\n]{20,}\"", " STR ", c)
    c = re.sub(r"\b[0-9a-f]{12,}\b", " HEX ", c); c = re.sub(r"\d+", "0", c)
    return re.sub(r"[ \t]+", " ", c)

# ---------------- nearest-neighbour step bank ----------------
def knn_fp(trk, tek, k=10, top=3):
    import torch
    def acts(kk): return S[kk][:, :384].astype(np.float32)
    cheat = np.concatenate([acts(kk) for kk in trk if y[kk[0]] == 1]); hon = np.concatenate([acts(kk) for kk in trk if y[kk[0]] == 0])
    C = torch.from_numpy(cheat); H = torch.from_numpy(hon)
    def sc(kk, excl):
        A = torch.from_numpy(acts(kk))
        c = (A @ C.T).topk(k, 1).values.mean(1); h = (A @ H.T).topk(k, 1).values.mean(1)
        d = (c - h).sort(descending=True).values[:top]
        return float(d.mean())
    # training runs' own steps are in the bank; only test scores are used, so no self-match issue for test
    return {kk: sc(kk, None) for kk in tek}

if __name__ == "__main__":
    which = sys.argv[1]
    if which.startswith("canon"):
        rows = [json.loads(l) for l in open("data/runs.jsonl")]
        CANON = {(i, v): "\n".join(canon(s["act"].split("\nOUT:")[0]) + " || " + canon(s["act"].split("\nOUT:")[1][:300] if "\nOUT:" in s["act"] else "")
                                   for s in r["v"][v]) for i, r in enumerate(rows) for v in r["v"]}
        fp = canon_fp
    elif which == "knn": fp = knn_fp
    else: fp = mil(aug="aug" in which, task_film="task" in which)
    O = run_folds(fp)
    if PADP: O["stripped_pad"] = np.array([PADP[(i, "stripped")] if y[i] else O["stripped"][i] for i in range(N)])
    report(which, O)
    if PADP: print("padded stripped AUC", round(roc_auc_score(y, O["stripped_pad"]), 4), "caught", round(tpr_at(y, O["stripped_pad"]), 4), flush=True)
    if PAD_SAME: json.dump({"padded_same_task_auc": round(roc_auc_score(y, O["stripped_pad"]), 4), "caught": round(tpr_at(y, O["stripped_pad"]), 4)}, open(f"results/v5_padsame_{which}.json", "w"))
    else: np.savez(f"data/v5_oof_{which}.npz", **O)
