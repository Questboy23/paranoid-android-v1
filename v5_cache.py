"""Cache per-step embedding sequences for every (run, variant) so v5 experiments skip the slow v4 load.
Each key gets a float16 matrix [steps, 3*384+1]: action, 3-step context, thought (zeros if none), has-thought flag,
plus the task prompt vector. Writes data/v5_steps.pkl."""
import pickle
import numpy as np
import v4_crossfit as B

z = np.zeros(B.V.shape[1], np.float32)
out = {}
for (i, v), u in B.U.items():
    st = B.rows[i]["v"][v]
    ctx = u["context"] if u["context"] else [None] * len(st)
    S = []
    for n, s in enumerate(st):
        a = B.V[B.ix[s["act"]]] if s["act"] else z
        c = B.V[B.ix[ctx[n]]] if ctx[n] else a
        t = B.V[B.ix[f"THOUGHT: {s['say'][:1200]}"]] if s["say"] else z
        S.append(np.r_[a, c, t, float(bool(s["say"]))])
    out[(i, v)] = np.array(S, np.float16) if S else np.zeros((1, 3 * len(z) + 1), np.float16)
P = {i: B.V[B.ix[r["prompt"]]].astype(np.float16) for i, r in enumerate(B.rows)}
pickle.dump({"steps": out, "prompt": P, "y": B.y, "task": B.task, "model": np.array([r["model"] for r in B.rows]),
             "ids": [r["id"] for r in B.rows]}, open("data/v5_steps.pkl", "wb"))
print("cached", len(out))
