"""v5 end-to-end reader: the public bge-small encoder is fine-tuned together with an attention layer over a run's steps,
using only whole-run labels (cheat or honest). Earlier fine-tuning (v2) used noisy per-step labels and a triplet loss.
Same 5 task folds as v4; each fold's model never sees its test tasks. Variant: stripped (actions and outputs only).
Usage: python v5_e2e.py [probe]   writes data/v5_oof_e2e.npz and results/v5_e2e.json"""
import json, sys, time
import numpy as np, torch, torch.nn as nn
from transformers import AutoTokenizer, AutoModel
from sklearn.metrics import roc_auc_score, roc_curve

VAR = "stripped"; MAXTOK = 160; MAXSTEPS = 16; EPOCHS = 2; LR = 3e-5; SEEDS = 1
dev = "mps" if torch.backends.mps.is_available() else "cpu"
rows = [json.loads(l) for l in open("data/runs.jsonl")]
y = np.array([r["label"] for r in rows]); task = np.array([r["task"] for r in rows]); N = len(rows)
tasks = np.unique(task); FOLDS = np.array_split(np.random.default_rng(0).permutation(tasks), 5)  # same as v4
tok = AutoTokenizer.from_pretrained("BAAI/bge-small-en-v1.5")

def steps(i):
    st = rows[i]["v"][VAR]
    st = st if len(st) <= MAXSTEPS else st[: MAXSTEPS // 2] + st[-MAXSTEPS // 2:]  # keep start and end
    return [s["act"][:2000] or "(empty)" for s in st] or ["(empty)"]
ENC = {i: tok(steps(i), truncation=True, max_length=MAXTOK) for i in range(N)}

class Net(nn.Module):
    def __init__(s):
        super().__init__()
        s.enc = AutoModel.from_pretrained("BAAI/bge-small-en-v1.5"); d = s.enc.config.hidden_size
        s.att = nn.Sequential(nn.Linear(d, 128), nn.Tanh(), nn.Linear(128, 1)); s.out = nn.Linear(2 * d, 1)
    def forward(s, ids, mask, bag):
        h = s.enc(input_ids=ids, attention_mask=mask).last_hidden_state[:, 0]  # CLS per step
        outs = []
        for b in range(int(bag.max()) + 1):
            hb = h[bag == b]; a = s.att(hb).softmax(0)
            outs.append(torch.cat([(a * hb).sum(0), hb.max(0).values]))
        return s.out(torch.stack(outs)).squeeze(-1)

def collate(idx):
    ids, bag = [], []
    for b, i in enumerate(idx):
        ids += ENC[i]["input_ids"]; bag += [b] * len(ENC[i]["input_ids"])
    L = max(len(x) for x in ids); I = torch.zeros(len(ids), L, dtype=torch.long); M = torch.zeros(len(ids), L, dtype=torch.long)
    for n, x in enumerate(ids): I[n, :len(x)] = torch.tensor(x); M[n, :len(x)] = 1
    return I.to(dev), M.to(dev), torch.tensor(bag).to(dev)

def fit_predict(tr, te, seed=0, probe=False):
    torch.manual_seed(seed); rng = np.random.default_rng(seed)
    net = Net().to(dev); opt = torch.optim.AdamW([{"params": net.enc.parameters(), "lr": LR},
                                                  {"params": list(net.att.parameters()) + list(net.out.parameters()), "lr": 1e-3}], weight_decay=0.01)
    bs = 8; nb = len(tr) // bs; total = EPOCHS * nb; step = 0; t0 = time.time()
    sched = torch.optim.lr_scheduler.LambdaLR(opt, lambda k: min(1, (k + 1) / 50) * max(0.0, 1 - k / total))
    for ep in range(EPOCHS):
        net.train()
        for b in np.array_split(rng.permutation(tr), nb):
            loss = nn.functional.binary_cross_entropy_with_logits(net(*collate(b)), torch.tensor(y[b], dtype=torch.float32, device=dev))
            opt.zero_grad(); loss.backward(); nn.utils.clip_grad_norm_(net.parameters(), 1.0); opt.step(); sched.step(); step += 1
            if probe and step == 30: print(f"{(time.time() - t0) / 30:.2f}s per batch of {bs}; est {total * (time.time() - t0) / 30 / 60:.0f} min per fold", flush=True); return None
            if step % 200 == 0: print(f"  step {step}/{total} loss {loss.item():.3f} {time.time() - t0:.0f}s", flush=True)
    net.eval(); p = []
    with torch.no_grad():
        for b in np.array_split(te, max(1, len(te) // 16)): p += net(*collate(b)).float().cpu().tolist()
    return np.array(p)

def tpr_at(yv, s, fpr=0.05):
    f, t, _ = roc_curve(yv, s); return float(np.interp(fpr, f, t))

if __name__ == "__main__":
    probe = len(sys.argv) > 1 and sys.argv[1] == "probe"
    O = np.zeros(N)
    import os; only = os.environ.get("FOLD")
    for n, f in enumerate(FOLDS):
        if only is not None and n != int(only): continue
        te = np.where(np.isin(task, f))[0]; tr = np.where(~np.isin(task, f))[0]
        p = fit_predict(tr, te, probe=probe)
        if probe: break
        O[te] = p; print("fold", n + 1, "AUC", round(roc_auc_score(y[te], p), 4), flush=True)
        json.dump({rows[i]["id"]: float(q) for i, q in zip(te, p)}, open(f"data/e2e_fold{n}.json", "w"))
    if only is not None: sys.exit(0)
    if not probe:
        res = {"variant": VAR, "auc": round(roc_auc_score(y, O), 4), "tpr5": round(tpr_at(y, O), 4)}
        print(res); json.dump(res, open("results/v5_e2e.json", "w")); np.savez("data/v5_oof_e2e.npz", stripped=O)
