"""Train and score one configuration. Env: EA_MODEL (HF name), EA_FT_N (fine-tune examples, 0 = no fine-tune), EA_TAG.

Protocol (same as the benchmark-trained baselines): learn on train-split tasks only, pick the one setting (C) on calib tasks,
report on test tasks never seen. Writes data/scores_<tag>.json (p per run and variant) and results/run_<tag>.json.
"""
import json, os, time
import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_auc_score
from sklearn.model_selection import GroupKFold
import monitor as M

MODEL = os.environ.get("EA_MODEL", "BAAI/bge-small-en-v1.5")
FT_N = int(os.environ.get("EA_FT_N", "6000"))
TAG = os.environ.get("EA_TAG", MODEL.split("/")[-1] + f"_ft{FT_N}")
MAX_SEQ = int(os.environ.get("EA_MAX_SEQ", "256"))
VARS = ("original", "sanitized", "stripped")
t0 = time.time()
log = lambda *a: print(f"[{time.time() - t0:7.0f}s]", *a, flush=True)

rows = [json.loads(l) for l in open("data/runs.jsonl")]
if os.environ.get("EA_SMOKE"):  # smoke test: every 25th task only
    keep = sorted({r["task"] for r in rows})[::25]; rows = [r for r in rows if r["task"] in keep]
y = np.array([r["label"] for r in rows]); split = np.array([r["split"] for r in rows]); task = np.array([r["task"] for r in rows])
tr, ca, te = split == "train", split == "calib", split == "test"


def final(st):
    return next((s["say"] for s in reversed(st) if s["say"]), "")


path = MODEL
if FT_N:
    path = f"checkpoints/{TAG}"
    if not os.path.exists(path):
        T, L = M.step_labels([r for r in rows if r["split"] == "train"])
        log("fine-tune units", len(T), "cheating share", round(float(np.mean(L)), 3))
        M.finetune(T, L, MODEL, path, n=FT_N, max_seq=MAX_SEQ)
        log("fine-tuned")

U = {(i, v): M.units(st) for i, r in enumerate(rows) for v, st in r["v"].items()}
texts = set(M.CHEAT + M.HONEST)
for (i, v), u in U.items():
    texts.update(u["thought"] + u["action"] + u["context"]); texts.add(u["files"]); texts.add(rows[i]["prompt"]); texts.add(final(rows[i]["v"][v]))
texts.discard(""); texts = sorted(texts)
cache = f"data/emb_{TAG}.npz"
SEC = json.load(open(f"results/sec_{TAG}.json")) if os.path.exists(f"results/sec_{TAG}.json") else None
if os.path.exists(cache) and json.load(open(cache + ".n.json")) == len(texts):
    V = np.load(cache)["V"].astype(np.float32)
else:
    from sentence_transformers import SentenceTransformer
    m = SentenceTransformer(path, device=M.device()); m.max_seq_length = MAX_SEQ
    log("encoding", len(texts), "texts")
    te0 = time.time(); os.makedirs("data/parts", exist_ok=True); parts = []
    for c in range(0, len(texts), 5000):  # chunks saved to disk so a killed run resumes (this Mac runs out of swap)
        f = f"data/parts/{TAG}_{len(texts)}_{c}.npy"
        if not os.path.exists(f):
            np.save(f, m.encode(texts[c:c + 5000], batch_size=32, normalize_embeddings=True, convert_to_numpy=True, show_progress_bar=False).astype(np.float16))
            log("chunk", c // 5000 + 1, "of", -(-len(texts) // 5000))
        parts.append(np.load(f))
    V = np.concatenate(parts).astype(np.float32)
    SEC = round((time.time() - te0) / sum(len(r["v"]) for r in rows), 4)  # seconds per run-variant
    json.dump(SEC, open(f"results/sec_{TAG}.json", "w"))
    log("encoded", SEC, "s per run")
    np.savez(cache, V=V.astype(np.float16)); json.dump(len(texts), open(cache + ".n.json", "w"))
ix = {t: n for n, t in enumerate(texts)}
E = lambda L: V[[ix[t] for t in L if t]] if L else np.zeros((0, V.shape[1]), np.float32)
D = (E(M.CHEAT), E(M.HONEST))
F = {k: M.features(u, rows[k[0]]["prompt"], final(rows[k[0]]["v"][k[1]]), E, D) for k, u in U.items()}
log("features built")

# word-count reader: out-of-fold for train rows (5 folds by task), fit on all train for the rest
def doc(i, v):
    return "\n".join((s["say"] + "\n" if s["say"] else "") + s["act"] for s in rows[i]["v"][v])
trk = [(i, v) for i in np.where(tr)[0] for v in (VARS if y[i] else ("original", "stripped"))]
ytr = np.array([y[i] for i, _ in trk]); gtr = np.array([task[i] for i, _ in trk])
tv = lambda: TfidfVectorizer(max_features=50000, ngram_range=(1, 2), min_df=2, sublinear_tf=True)
words = {}
for a, b in GroupKFold(5).split(trk, ytr, [task[i] for i, _ in trk]):
    vec = tv().fit([doc(*trk[j]) for j in a]); lr = LogisticRegression(C=4, max_iter=3000).fit(vec.transform([doc(*trk[j]) for j in a]), ytr[a])
    for j, s in zip(b, lr.decision_function(vec.transform([doc(*trk[j]) for j in b]))):
        words[trk[j]] = s
vec = tv().fit([doc(*k) for k in trk]); lr = LogisticRegression(C=4, max_iter=3000).fit(vec.transform([doc(*k) for k in trk]), ytr)
rest = [k for k in F if k not in words]
for k, s in zip(rest, lr.decision_function(vec.transform([doc(*k) for k in rest]))):
    words[k] = s
for k in F:
    F[k]["words"] = np.array([words[k]])
log("word counts done")


def eval_keys(mask, v):
    """Hack runs in variant v, honest runs as recorded (stripped variant: honest text blanked too)."""
    neg = "stripped" if v == "stripped" else "original"
    return [(i, v if y[i] else neg) for i in np.where(mask)[0]]


def auc(head, mask, v):
    k = eval_keys(mask, v)
    return roc_auc_score([y[i] for i, _ in k], head.p([F[x] for x in k]))


def fit(groups, C=None):
    h = M.Head(groups); X = [F[k] for k in trk]
    if C is None:  # pick C on calib tasks only
        best = max((np.mean([auc(M.Head(groups).fit(X, ytr, c, gtr), ca, v) for v in VARS]), c) for c in (0.01, 0.05, 0.2, 1.0))
        C = best[1]
    return h.fit(X, ytr, C, gtr), C



head, C = fit(M.GROUPS, float(os.environ["EA_C"]) if os.environ.get("EA_C") else None)
log("chosen C", C)
if os.environ.get("EA_SAVE_HEAD"):  # for scoring other benchmarks (malt.py): the trained head and word-count reader, no test data used
    import pickle
    pickle.dump({"head": head, "vec": vec, "lr": lr, "C": C, "path": path, "max_seq": MAX_SEQ}, open(f"checkpoints/head_{TAG}.pkl", "wb"))
    log("saved head"); raise SystemExit
PARAMS = {"bge-small-en-v1.5": "33M", "bge-base-en-v1.5": "109M", "bge-large-en-v1.5": "335M"}
res = {"params": PARAMS.get(MODEL.split("/")[-1], ""), "sec_per_run": SEC, "tag": TAG, "model": MODEL, "ft_n": FT_N, "max_seq": MAX_SEQ, "C": C, "test_auc": {v: round(auc(head, te, v), 4) for v in VARS}}
res["calib_auc"] = {v: round(auc(head, ca, v), 4) for v in VARS}
pcal = {v: head.p([F[k] for k in eval_keys(ca, v)]) for v in VARS}
ycal = {v: np.array([y[i] for i, _ in eval_keys(ca, v)]) for v in VARS}
res["router_cutoffs"] = {v: [float(np.quantile(pcal[v][ycal[v] == 1], 0.01)), float(np.quantile(pcal[v][ycal[v] == 0], 0.99))] for v in VARS}
res["threshold_5fpr_calib"] = {v: float(np.quantile(pcal[v][ycal[v] == 0], 0.95)) for v in VARS}
res["ablation_drop_one"] = {}
for g in M.GROUPS:
    h, _ = fit(tuple(x for x in M.GROUPS if x != g), C)
    res["ablation_drop_one"][g] = {v: round(auc(h, te, v), 4) for v in VARS}
res["only_one"] = {}
for g in M.GROUPS:
    h, _ = fit((g,), C)
    res["only_one"][g] = {v: round(auc(h, te, v), 4) for v in VARS}
log("ablations done")

# held-out agent model: head refit without that model's runs; test only on that model (fine-tune still saw its train-split runs)
res["heldout_model"] = {}
models = sorted({r["model"] for r in rows})
for mdl in models:
    keep = [j for j, (i, _) in enumerate(trk) if rows[i]["model"] != mdl]
    h = M.Head(M.GROUPS).fit([F[trk[j]] for j in keep], ytr[keep], C, gtr[keep])
    mm = te & np.array([r["model"] == mdl for r in rows])
    if len(set(y[mm])) == 2:
        res["heldout_model"][mdl] = {v: round(auc(h, mm, v), 4) for v in VARS}

scores = {"ids": [r["id"] for r in rows]}
for (i, v) in F:
    scores.setdefault(v, {})[rows[i]["id"]] = float(head.p([F[(i, v)]])[0])
json.dump(scores, open(f"data/scores_{TAG}.json", "w"))
json.dump(res, open(f"results/run_{TAG}.json", "w"), indent=1)
log(json.dumps({k: res[k] for k in ("test_auc", "C")}))
