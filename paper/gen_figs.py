"""Charts for the diagram paper, all read from results/*.json. Run from repo root: python3 paper/gen_figs.py"""
import json
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

J = lambda p: json.load(open(p))
F, M, FIN, X = J("results/v5_fair.json"), J("results/v5_fair_more.json"), J("results/v5_final_stripped.json"), J("results/cross_bench.json")
F10, F50 = J("results/v5_fair_full_10.json"), J("results/v5_fair_full_50.json")
V3, VAL = J("results/v2_final.json"), J("results/v2_base_val.json")["v5_stack"]
HOLD, IB = V3["A_hold"]["v2_embed"], V3["E_impossiblebench"]
ENC_S, JUDGE_S = 0.0321, 23  # RESULTS.md section 6 of the dev repo: encode seconds per run; AI judge about 23 s per run

BLUE, GREY, RED, INK, GOLD = "#2D6CB0", "#9AA3AE", "#C0392B", "#1B1F27", "#B87600"
plt.rcParams.update({"font.family": ["Helvetica", "DejaVu Sans"], "font.size": 13, "axes.spines.top": False,
                     "axes.spines.right": False, "axes.edgecolor": "#555", "axes.titleweight": "bold", "axes.titlesize": 15})
W, H = 10.5, 5.6

def save(fig, name):
    fig.tight_layout(); fig.savefig(f"paper/fig_{name}.pdf"); plt.close(fig)

# 1. speed
fig, (a, b) = plt.subplots(1, 2, figsize=(W, H))
a.bar(["Paranoid Android v1", "AI judge"], [ENC_S, JUDGE_S], color=[BLUE, GREY], width=.55)
a.set_title("Seconds to check one run"); a.set_ylabel("seconds")
a.text(0, ENC_S + .5, f"{ENC_S:.2f} s", ha="center", fontsize=16, fontweight="bold", color=BLUE)
a.text(1, JUDGE_S + .5, f"about {JUDGE_S} s", ha="center", fontsize=16, fontweight="bold", color=INK)
hrs, days = 1e6 * ENC_S / 3600, 1e6 * JUDGE_S / 86400
b.bar(["Paranoid Android v1", "AI judge"], [hrs / 24, days], color=[BLUE, GREY], width=.55)
b.set_title("Days to check one million runs"); b.set_ylabel("days, one machine")
b.text(0, days * .03, f"about {hrs:.0f} hours", ha="center", fontsize=16, fontweight="bold", color=BLUE)
b.text(1, days + 4, f"about {days:.0f} days", ha="center", fontsize=16, fontweight="bold", color=INK)
for ax in (a, b): ax.margins(y=.12)
save(fig, "speed")

# 2. catch rate on Terminal Wrench
rows = [("Full system\n(upper bound)", FIN["tpr5"], FIN["tpr5_ci"], BLUE),
        ("Full system, retrained\nwith 33 jobs locked away", VAL["catch"], VAL["catch_ci"], BLUE),
        ("Simple reader", M["265"]["caught"], None, BLUE),
        ("Earlier version on\n629 never-used runs", HOLD["catch5"], HOLD["catch5_ci"], BLUE),
        ("GPT-5.4 judge\n(published)", .44, None, GREY),
        ("No labels at all", F["zero_shot"]["caught"], None, "#8FB3DA")]
fig, ax = plt.subplots(figsize=(W, H))
y = np.arange(len(rows))[::-1]
for yi, (lab, v, c, col) in zip(y, rows):
    ax.barh(yi, 100 * v, color=col, height=.62)
    if c: ax.errorbar(100 * v, yi, xerr=[[100 * (v - c[0])], [100 * (c[1] - v)]], color=INK, capsize=4, lw=1.3)
    ax.text(100 * (c[1] if c else v) + 1.5, yi, f"{100 * v:.0f}%", va="center", fontsize=14, fontweight="bold")
ax.set_yticks(y, [r[0] for r in rows]); ax.set_xlim(0, 100); ax.set_xlabel("cheating runs caught (%), with 5 in 100 honest runs flagged")
ax.set_title("Terminal Wrench: cheating caught on jobs it never trained on", loc="left")
save(fig, "caught")

# 3. learning curve
ks = [0, 5, 10, 25, 50, 100, 150, 200]
med = [F["zero_shot"]["caught"]] + [F["few_tasks"][str(k)]["caught_median"] for k in ks[1:5]] + [M[str(k)]["caught_median"] for k in ks[5:]]
low = [F["zero_shot"]["caught"]] + [F["few_tasks"][str(k)]["caught_min"] for k in ks[1:5]] + [M[str(k)]["caught_min"] for k in ks[5:]]
fig, ax = plt.subplots(figsize=(W, H))
ax.fill_between(ks + [265], 100 * np.r_[low, M["265"]["caught"]], 100 * np.r_[med, M["265"]["caught"]], color=BLUE, alpha=.15, lw=0, label="worst of 10 random picks")
ax.plot(ks + [265], 100 * np.r_[med, M["265"]["caught"]], "o-", color=BLUE, lw=2.5, label="simple reader (median of 10 picks)")
ax.plot([10, 50, 265], [100 * F10["10"]["caught_median"], 100 * F50["50"]["caught_median"], 100 * FIN["tpr5"]], "s--", color=INK, lw=1.8, label="full system")
ax.axhline(44, color=GREY, lw=2, ls=":"); ax.text(268, 45.5, "GPT-5.4 judge 44%", color="#666", ha="right", fontsize=12)
ax.set_xlabel("jobs with labelled runs (about 18 runs each)"); ax.set_ylabel("cheating runs caught (%)"); ax.set_ylim(0, 90); ax.set_xlim(-5, 275)
ax.set_title("More labelled jobs, more cheating caught. Most of the gain comes in the first 25.", loc="left")
ax.legend(loc="lower right", frameon=False)
save(fig, "curve")

# 4. transfer matrix
B = [("tw", "Terminal\nWrench"), ("malt", "MALT"), ("ib", "Impossible\nBench")]
Mx = np.zeros((3, 3))
for i, (tr, _) in enumerate(B):
    for j, (te, _) in enumerate(B):
        Mx[i, j] = X[f"same benchmark {te} (new tasks)"]["auc"] if tr == te else X[f"train {tr} -> test {te}"]["auc"]
fig, (a, b) = plt.subplots(1, 2, figsize=(W, H), gridspec_kw={"width_ratios": [1.25, 1]})
im = a.imshow(Mx, cmap="RdYlBu", vmin=.3, vmax=1)
for i in range(3):
    for j in range(3):
        a.text(j, i, f"{Mx[i, j]:.2f}" + ("\nsame kind,\nnew jobs" if i == j else ""), ha="center", va="center", fontsize=13 if i == j else 15, fontweight="bold")
a.set_xticks(range(3), [b_[1] for b_ in B]); a.set_yticks(range(3), [b_[1] for b_ in B])
a.set_xlabel("tested on"); a.set_ylabel("trained on"); a.set_title("Score (AUC): 1 = perfect, 0.5 = guessing", fontsize=13)
for s in a.spines.values(): s.set_visible(False)
two = [("MALT + IB\n→ TW", X["train malt+ib -> test tw"]["auc"]), ("TW + IB\n→ MALT", X["train tw+ib -> test malt"]["auc"]), ("TW + MALT\n→ IB", X["train tw+malt -> test ib"]["auc"])]
b.bar([t[0] for t in two], [t[1] for t in two], color=RED, width=.55); b.axhline(.5, color=INK, ls=":", lw=1.5)
b.text(.5, .455, "guessing", ha="center", fontsize=10); b.set_ylim(0, 1); b.set_title("Trained on two, tested on the third", fontsize=13)
for i, t in enumerate(two): b.text(i, t[1] + .02, f"{t[1]:.2f}", ha="center", fontweight="bold")
save(fig, "transfer")

# 5. new agents and other checks
items = [("Claude Opus 4.6", F["new_agent"]["claude-opus-4.6"]["caught"]), ("Gemini 3.1 Pro", F["new_agent"]["gemini-3.1-pro"]["caught"]),
         ("GPT-5.4", F["new_agent"]["gpt-5.4"]["caught"])]
fig, ax = plt.subplots(figsize=(W, H * .8))
ax.barh(range(3)[::-1], [100 * v for _, v in items], color=GOLD, height=.6)
for yi, (_, v) in zip(range(3)[::-1], items): ax.text(100 * v + 1.5, yi, f"{100 * v:.0f}%", va="center", fontweight="bold", fontsize=14)
ax.set_yticks(range(3)[::-1], [n for n, _ in items]); ax.set_xlim(0, 100)
ax.set_xlabel("cheating runs caught (%), trained only on the other two agents")
ax.set_title("An AI agent it never learned from: partly works", loc="left")
save(fig, "agents")
print("figs ok")
