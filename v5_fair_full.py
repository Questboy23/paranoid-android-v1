"""Few-label test for the full v4 reader stack (six readers, stacked), the main part of v5: labels from only k tasks,
tested on all other tasks, stripped variant, 10 random draws per k. Writes results/v5_fair_full.json."""
import json, warnings
import numpy as np
warnings.filterwarnings("ignore")
import v4_crossfit as B
res = {}; rng = np.random.default_rng(0); tasks = np.unique(B.task)
import sys
for k in [int(a) for a in sys.argv[1:]] or (10, 50):
    out = []
    for _ in range(10):
        pick = rng.choice(tasks, k, replace=False); tr = np.where(np.isin(B.task, pick))[0]; te = np.where(~np.isin(B.task, pick))[0]
        if len(set(B.y[tr])) < 2: continue
        try: p, _ = B.fold_fit(tr, te)
        except ValueError: continue  # too few tasks for the inner folds
        k2 = B.eval_keys(te, "stripped"); out.append(B.tpr_at(np.array([B.y[i] for i, _ in k2]), np.array([p[x] for x in k2])))
    res[k] = dict(caught_median=round(float(np.median(out)), 4), caught_min=round(float(min(out)), 4), draws=len(out)); print(k, res[k], flush=True)
json.dump(res, open(f"results/v5_fair_full_{'_'.join(sys.argv[1:])}.json", "w"), indent=1)
