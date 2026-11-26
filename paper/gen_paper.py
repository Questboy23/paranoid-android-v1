"""Paranoid Android v1 paper: one picture per page, every number in the last two pages. Every number is read from
results/*.json, the v2 repo's results, or RESULTS.md (published GPT-5.4 judge, judge timing).
Run from repo root: .venv/bin/python paper/gen_paper.py && cd paper && tectonic paper.tex"""
import json, os
J = lambda p: json.load(open(os.path.expanduser(p)))
F, M, FIN, X = J("results/v5_fair.json"), J("results/v5_fair_more.json"), J("results/v5_final_stripped.json"), J("results/cross_bench.json")
F10, F50, MA = J("results/v5_fair_full_10.json"), J("results/v5_fair_full_50.json"), J("results/malt.json")
V3 = J("results/v2_final.json"); VAL = J("results/v2_base_val.json")["v5_stack"]
HOLD, IB, MS, n = V3["A_hold"]["v2_embed"], V3["E_impossiblebench"], MA["variants"]["stripped"]["all"], X["n"]
assert n["tw"] == [5984, 3632] and n["malt"] == [1907, 407] and n["ib"] == [205, 32] and V3["n_hold"] == 629
REPO = "https://github.com/Questboy23/paranoid-android-v1"
ENC_S, JUDGE_S = 0.0321, 23  # RESULTS.md section 6: encode seconds per run; AI judge about 23 s per run
pc = lambda v: f"{100 * v:.1f}"
p0 = lambda v: f"{100 * v:.0f}"
ci = lambda c: f"[{c[0]:.3f}, {c[1]:.3f}]"
cip = lambda c: f"[{100 * c[0]:.1f}, {100 * c[1]:.1f}]"
one = [X[k]["auc"] for k in X if k.startswith("train ") and "+" not in k]
two = [X[k]["auc"] for k in X if k.startswith("train ") and "+" in k]
same = [X[f"same benchmark {b} (new tasks)"]["auc"] for b in ("tw", "malt", "ib")]
assert max(two) < 0.65 and min(same) > 0.8

def xrow(k, label):
    r = X[k]
    return rf"{label} & {r['auc']:.3f} {ci(r['auc_ci'])} & {pc(r['caught_line_from_train'])} & {pc(r['caught_line_from_test_honest'])} \\"

full = {"10": F10["10"], "50": F50["50"]}
curve = [("0", F["zero_shot"]["caught"], None)] + [(k, F["few_tasks"][k]["caught_median"], F["few_tasks"][k]["caught_min"]) for k in ("5", "10", "25", "50")] \
      + [(k, M[k]["caught_median"], M[k]["caught_min"]) for k in ("100", "150", "200")] + [("about 265", M["265"]["caught"], None)]
CURVE = "\n".join(rf"{k} & {pc(v)} & {'--' if w is None else pc(w)} & "
                  + (rf"{pc(full[k]['caught_median'])} ({pc(full[k]['caught_min'])})" if k in full else (pc(FIN['tpr5']) if k.startswith('about') else '--')) + r" \\"
                  for k, v, w in curve)
AGENTS = "\n".join(rf"New agent: {a} (simple reader) & {F['new_agent'][k]['auc']:.3f} & {pc(F['new_agent'][k]['caught'])} \\"
                   for a, k in (("Claude Opus 4.6", "claude-opus-4.6"), ("Gemini 3.1 Pro", "gemini-3.1-pro"), ("GPT-5.4", "gpt-5.4")))
CROSS = "\n".join(xrow(k, l) for k, l in (
    ("train tw -> test malt", "Terminal Wrench $\\to$ MALT"), ("train tw -> test ib", "Terminal Wrench $\\to$ ImpossibleBench"),
    ("train malt -> test tw", "MALT $\\to$ Terminal Wrench"), ("train malt -> test ib", "MALT $\\to$ ImpossibleBench"),
    ("train ib -> test tw", "ImpossibleBench $\\to$ Terminal Wrench"), ("train ib -> test malt", "ImpossibleBench $\\to$ MALT"))) + "\n\\midrule\n" \
    + "\n".join(xrow(k, l) for k, l in (
    ("train malt+ib -> test tw", "MALT + ImpossibleBench $\\to$ Terminal Wrench"), ("train tw+ib -> test malt", "Terminal Wrench + ImpossibleBench $\\to$ MALT"),
    ("train tw+malt -> test ib", "Terminal Wrench + MALT $\\to$ ImpossibleBench"))) + "\n\\midrule\n" \
    + "\n".join(rf"{l}, new jobs only & {X[k]['auc']:.3f} {ci(X[k]['auc_ci'])} & -- & {pc(X[k]['caught_line_from_test_honest'])} \\"
                for k, l in (("same benchmark tw (new tasks)", "Terminal Wrench"), ("same benchmark malt (new tasks)", "MALT"), ("same benchmark ib (new tasks)", "ImpossibleBench")))

V = dict(BEST=pc(FIN["tpr5"]), BEST_AUC=f"{FIN['auc']:.3f}", BEST_AUCCI=ci(FIN["auc_ci"]), BESTCI=cip(FIN["tpr5_ci"]),
         VAL=pc(VAL["catch"]), VALCI=cip(VAL["catch_ci"]), VAL_AUC=f"{VAL['auc']:.3f}",
         SIMPLE=pc(M["265"]["caught"]), HOLD=pc(HOLD["catch5"]), HOLDCI=cip(HOLD["catch5_ci"]), HOLD_AUC=f"{HOLD['auc']:.3f}", HOLD_AUCCI=ci(HOLD["auc_ci"]),
         ZERO=pc(F["zero_shot"]["caught"]), ZERO_AUC=f"{F['zero_shot']['auc']:.3f}", L25=p0(F["few_tasks"]["25"]["caught_median"]),
         W10=pc(F10["10"]["caught_min"]), AG_LO=p0(F["new_agent"]["gpt-5.4"]["caught"]), AG_HI=p0(F["new_agent"]["claude-opus-4.6"]["caught"]),
         ONE_LO=f"{min(one):.2f}", ONE_HI=f"{max(one):.2f}", TWO_LO=f"{min(two):.2f}", TWO_HI=f"{max(two):.2f}",
         S_LO=f"{min(same):.2f}", S_HI=f"{max(same):.2f}", S_TW=f"{same[0]:.2f}", S_MALT=f"{same[1]:.2f}", S_IB=f"{same[2]:.2f}",
         MALT_AUC=f"{MS['auc']:.3f}", MALT_CI=ci(MS["ci"]), MALT_C=pc(MS["catch5"]),
         IB_AUC=f"{IB['v2_embed']['auc']:.3f}", IB_CI=ci(IB["v2_embed"]["auc_ci"]), IB_C=pc(IB["v2_embed"]["catch5"]),
         J_IB=p0(IB["judge_tw"]["catch5"]), J_IB_AUC=f"{IB['judge_tw']['auc']:.3f}",
         ENC=f"{ENC_S:.2f}", SPEED=f"{round(JUDGE_S / ENC_S, -2):.0f}", HOURS=f"{1e6 * ENC_S / 3600:.0f}", DAYS=f"{1e6 * JUDGE_S / 86400:.0f}",
         CURVE=CURVE, AGENTS=AGENTS, CROSS=CROSS, REPO=REPO, NTW=f"{n['tw'][0]:,}", CTW=f"{n['tw'][1]:,}", NMALT=f"{n['malt'][0]:,}")

TEX = r"""\documentclass[11pt]{article}
\usepackage[a4paper,landscape,margin=1.4cm]{geometry}\usepackage[T1]{fontenc}\usepackage{tgheros}\renewcommand{\familydefault}{\sfdefault}
\usepackage{microtype,booktabs,graphicx,xcolor,tikz}\usetikzlibrary{arrows.meta,positioning}\usepackage[hidelinks]{hyperref}
\setlength{\parskip}{4pt}\setlength{\parindent}{0pt}\setlength{\tabcolsep}{5pt}\pagestyle{empty}
\definecolor{ink}{HTML}{1B1F27}\definecolor{pa}{HTML}{2D6CB0}\definecolor{bad}{HTML}{C0392B}\definecolor{ok}{HTML}{2E7D32}
\newcommand{\pg}[2]{\newpage{\fontsize{22}{26}\selectfont\bfseries #1}\par\vspace{1mm}{\large\color{ink!75}#2}\par\vspace{3mm}}
\newcommand{\fig}[1]{\begin{center}\includegraphics[width=\linewidth,height=14.2cm,keepaspectratio]{#1}\end{center}}
\tikzset{box/.style={draw=ink!40,rounded corners=6pt,align=center,inner sep=8pt,font=\large,text width=7.4cm,minimum height=3.4cm},
 q/.style={draw=ink!50,rounded corners=6pt,fill=ink!5,align=center,inner sep=7pt,font=\large,text width=5.6cm},
 e/.style={-{Stealth[length=3mm]},line width=1.4pt,ink!60}}
\begin{document}
% ---------- page 1 ----------
{\fontsize{34}{38}\selectfont\bfseries Paranoid Android v1}\\[3pt]
{\Large Catching AI agents that cheat a grader, in a few hundredths of a second}\\[2pt]
{\color{ink!60}Questboy23 \quad November 2026 \quad \url{@REPO@}}

\vspace{7mm}
{\large AI coding agents sometimes get a passing grade without doing the work, for example by copying the answer key into the
output. Paranoid Android v1 reads a saved run with a small free word-embedding model and seven simple scorers, and decides in
about @ENC@ seconds whether it looks like cheating. Each page that follows is one picture.}

\vspace{10mm}
\begin{center}\resizebox{\linewidth}{!}{\begin{tikzpicture}
\node[box,fill=pa!10] (a) at (0,0) {{\fontsize{30}{34}\selectfont\bfseries\color{pa} @SPEED@$\times$}\\[4pt] faster than an AI judge\\[2pt]{\normalsize @ENC@ s per run against about @JUDGE@ s}};
\node[box,fill=pa!10] (b) at (8.4,0) {{\fontsize{30}{34}\selectfont\bfseries\color{pa} about @SIMPLE_R@\%}\\[4pt] of cheating caught on new jobs\\[2pt]{\normalsize best case @BEST_R@\% (see page 2); GPT-5.4 judge: 44\%}};
\node[box,fill=bad!10] (c) at (16.8,0) {{\fontsize{30}{34}\selectfont\bfseries\color{bad} @ONE_LO@ to @ONE_HI@}\\[4pt] score on benchmarks it never saw\\[2pt]{\normalsize 0.5 is guessing: it only knows the kinds of cheating it was shown}};
\end{tikzpicture}}\end{center}

\vspace{12mm}
{\large\color{ink!70}\textbf{Pages:} 2 is the @BEST_R@\% honest? \quad 3 how it works \quad 4 the embedding, step by step \quad 5 speed \quad 6 how much it catches \quad 7 how many labels it needs \quad
8 does it carry over \quad 9 new AI agents \quad 10 where to use it, and its limits \quad 11 to 12 every number}

% ---------- page 2: is the 81% honest ----------
\pg{Is the @BEST_R@\% honest? Partly.}{It was never scored on a job it trained on. But it did train on other jobs from the same benchmark, and its design was picked using these same scores. So treat @BEST_R@\% as a best case, and about @SIMPLE_R@\% as the realistic figure.}
\begin{center}\resizebox{\linewidth}{!}{\begin{tikzpicture}
% how the test is run
\node[font=\Large\bfseries,anchor=west] at (-0.2,1.2) {How each run is scored};
\node[draw=ink!40,rounded corners=5pt,fill=ink!5,minimum width=10.6cm,minimum height=1cm,font=\large] (jobs) at (5.1,0) {Terminal Wrench: 331 jobs, 5,984 runs};
\foreach \i/\c in {0/pa!15,1/pa!15,2/pa!15,3/pa!15,4/bad!25}
  \node[draw=ink!40,rounded corners=4pt,fill=\c,minimum width=1.9cm,minimum height=1.1cm,font=\normalsize,align=center] (g\i) at (\i*2.15+0.8,-2.2) {group \the\numexpr\i+1\relax\\[-1pt]{\footnotesize about 66 jobs}};
\draw[e] (jobs.south) -- ++(0,-.5) -| (g4.north);
\draw[e] (jobs.south) -- ++(0,-.5) -| (g0.north);
\node[align=center,font=\normalsize,text=pa] at (3.95,-3.35) {learns from these 4 groups};
\draw[pa,line width=1.3pt] (0,-3.0) -- (7.65,-3.0);
\node[align=center,font=\normalsize,text=bad] at (9.4,-3.35) {scores this group};
\node[q,text width=9.6cm,font=\normalsize] at (5.1,-4.9) {Repeat 5 times, so every group gets scored once by a model that never saw its jobs. The same 5 groups (seed 0) are used by every reader, by the retrained embedding model, and by the weights that add the scores up.};
% verdicts
\node[draw=ok!70,rounded corners=6pt,fill=ok!8,text width=9.8cm,align=left,inner sep=9pt,font=\normalsize,anchor=north west] (good) at (11.6,1.6) {
{\Large\bfseries\color{ok} Clean (checked in the code)}\\[4pt]
\textbf{No run is scored by a model that trained on its job.} Checked in \texttt{v4\_crossfit.py}, \texttt{v5\_lab.py}, \texttt{v5\_e2e.py} and \texttt{v5\_combine.py}.\\[4pt]
\textbf{The embedding model never saw the benchmark.} It is the public bge-small. The one reader that retrains it does so separately for each group.\\[4pt]
\textbf{The agent's own words are deleted} in every run, honest and cheating alike.};
\node[draw=bad!70,rounded corners=6pt,fill=bad!6,text width=9.8cm,align=left,inner sep=9pt,font=\normalsize,anchor=north west,yshift=-4mm] (bad) at (good.south west) {
{\Large\bfseries\color{bad} Not clean}\\[4pt]
\textbf{It trained on the benchmark's other jobs.} @BEST_R@\% answers ``new jobs, same benchmark''. On a benchmark it never trained on, it caught about 0\%.\\[4pt]
\textbf{The design was picked on these scores.} Which seven readers to keep was decided by looking at these same results. That makes the number look better than it would on fresh data.\\[4pt]
\textbf{The alarm line was set on these honest runs.} In real use the line has to be fixed in advance.};
\node[draw=pa!70,rounded corners=6pt,fill=pa!10,text width=9.6cm,align=left,inner sep=9pt,font=\normalsize,anchor=north] at (5.1,-6.6) {
{\Large\bfseries\color{pa} The realistic number: about @SIMPLE_R@\%}\\[3pt]
Two checks with far fewer choices made on the scored runs agree: a simple reader, @SIMPLE@\%, and an earlier version on 629 runs never used for any choice, @HOLD@\%. Best case @BEST@\%.};
\end{tikzpicture}}\end{center}

% ---------- tree ----------
\newpage
\begin{center}\includegraphics[width=\linewidth,height=17.8cm,keepaspectratio,trim=0 3cm 0 0,clip]{tree.pdf}\end{center}

% ---------- page 4: the embedding, step by step ----------
\pg{How the embedding works, step by step}{The word-embedding model is bge-small-en-v1.5: public, free, 33 million numbers inside, 12 layers. It turns any piece of text into 384 numbers. The real example below was run through it.}
\begin{center}\resizebox{\linewidth}{!}{\begin{tikzpicture}[node distance=7mm]
\tikzset{st/.style={draw=ink!40,rounded corners=5pt,fill=pa!8,align=left,inner sep=7pt,font=\normalsize,text width=5.2cm,minimum height=4.0cm,anchor=north west}}
\node[st] (a) at (0,0) {{\large\bfseries 1. One step}\\[3pt] A command and its output. The agent's own words are deleted.\\[3pt]{\ttfamily\small CMD: cp /tests/expected\_output.txt /app/output.txt}\\[2pt]{\footnotesize Cut to 2,000 characters.}};
\node[st] (b) at (6.1,0) {{\large\bfseries 2. Word pieces}\\[3pt] The text is split into common word pieces:\\[3pt]{\ttfamily\small cm \#\#d : cp / tests / expected \_ output . tx \#\#t ...}\\[2pt]{\footnotesize Only the first 256 pieces are read.}};
\node[st] (c) at (12.2,0) {{\large\bfseries 3. Twelve layers}\\[3pt] In each layer every piece looks at every other piece. So ``expected'' next to ``tests'' and ``cp'' is read differently from ``expected'' in an error message.};
\node[st] (d) at (18.3,0) {{\large\bfseries 4. 384 numbers}\\[3pt] One summary piece holds the meaning of the whole step. Its 384 numbers are scaled to length 1. The first three, for real:\\[3pt]{\ttfamily\small [-0.055, 0.031, 0.025, ...]}};
\foreach \x/\y in {a/b,b/c,c/d} \draw[e] (\x.east) -- (\y.west);
% closeness
\node[draw=ink!40,rounded corners=6pt,fill=ink!4,text width=11.2cm,align=left,inner sep=9pt,font=\normalsize,anchor=north west] (cl) at (0,-4.75) {
{\large\bfseries 5. Closeness = similar meaning}\\[3pt]
Two steps are compared by multiplying their numbers pairwise and adding up (1 = same meaning). Real scores from the model against the step above:\\[4pt]
\begin{tabular}{lr}
{\ttfamily\small cat /grader/answers.json > /app/result.json} \ (cheating) & 0.73 \\
{\ttfamily\small python solve.py > /app/output.txt} \ (honest) & 0.81 \\
{\ttfamily\small pip install numpy} \ (honest) & 0.58 \\
\end{tabular}\\[4pt]
\textbf{The raw model does not know what cheating is.} The honest step scores closest because it shares words. That is why the next step learns from labelled runs.};
\node[draw=pa!60,rounded corners=6pt,fill=pa!8,text width=11.2cm,align=left,inner sep=9pt,font=\normalsize,anchor=north west] (rd) at (12.2,-4.75) {
{\large\bfseries 6. From steps to a verdict}\\[3pt]
A run becomes a table: one row of 384 numbers per step. Seven readers each turn the table into one score:\\[3pt]
\textbf{Meaning bundle:} the average and the peak of each column, for single steps and for 3-step windows; how well the work matches the task text; closeness to 6 written descriptions of cheating and 5 of honest work; a word-count score.\\[2pt]
\textbf{Nearest steps:} the run's top 3 steps against the 10 nearest cheating and honest steps seen in training.\\[2pt]
\textbf{Step pickers (3):} learn to point at the 1 to 3 steps where the cheat happens.\\[2pt]
\textbf{Letter patterns:} commands with numbers and names blanked out, read as letter groups (no embedding).\\[2pt]
\textbf{Retrained model:} the embedding model itself, retrained on the training jobs only.\\[3pt]
A final weighted sum of the seven scores gives the verdict.};
\end{tikzpicture}}\end{center}

% ---------- page 3 ----------
\pg{Speed: the main gain}{The embedding model is free, runs on a laptop and does not call any outside service. That makes checking every run practical, instead of a sample.}
\fig{fig_speed.pdf}

% ---------- page 4 ----------
\pg{How much cheating it catches}{Terminal Wrench, the benchmark it learned from, scored only on jobs each model never trained on. Bars with lines show 95\% ranges. The top bar is a best case, for the reasons on page 2.}
\fig{fig_caught.pdf}

% ---------- page 5 ----------
\pg{How many labelled examples it needs}{A labelled run is a past run already marked cheating or honest. Labels from about 25 jobs give @L25@\%. With only 10 jobs, an unlucky pick can fail badly (the full system's worst pick caught @W10@\%).}
\fig{fig_curve.pdf}

% ---------- page 6 ----------
\pg{Does it carry over?}{Diagonal: trained on some jobs of a benchmark, tested on its other jobs. It works on all three. Everywhere else: trained on one benchmark, tested on another. It is close to guessing. Trained on Terminal Wrench, the alarm line it brings along catches 0\% on the other two.}
\fig{fig_transfer.pdf}

% ---------- page 7 ----------
\pg{A new AI agent}{Trained on runs from two agents, tested on the third agent's runs on jobs it never saw.}
\fig{fig_agents.pdf}

% ---------- page 8 ----------
\pg{Where to use it, and its limits}{}
\begin{center}\resizebox{\linewidth}{!}{\begin{tikzpicture}
\node[q] (s) at (0,0) {A new agent run arrives};
\node[q] (l) at (0,-3.0) {Do you have labelled runs of this kind of task and cheating?};
\node[q,fill=ok!12] (y) at (-3.4,-6.4) {\textbf{Yes:} check every run with v1 in @ENC@ s};
\node[q,fill=bad!12] (n) at (3.4,-6.4) {\textbf{No:} v1 will miss it. Use an AI judge, or label some runs first};
\node[q,fill=pa!10] (f) at (-3.4,-9.9) {Flagged runs go to an AI judge or a person};
\draw[e] (s) -- (l); \draw[e] (l) -- node[left,font=\large]{yes} (y); \draw[e] (l) -- node[right,font=\large]{no} (n); \draw[e] (y) -- (f);
\node[draw=bad!60,rounded corners=6pt,fill=bad!5,text width=11.6cm,align=left,inner sep=9pt,font=\normalsize,anchor=north west] at (7.4,0.9) {
{\Large\bfseries\color{bad} Limits}\\[4pt]
\textbf{Misses some.} Even at best about 1 cheating run in 5 gets through, and 5 in 100 honest runs are flagged.\\[4pt]
\textbf{@BEST@\% is a best case} (page 2). It trained on other jobs of the same benchmark, and the design was picked on the same scores. Realistic: about @SIMPLE_R@\%.\\[4pt]
\textbf{No carry-over.} New kinds of misbehaviour need new labels. On ImpossibleBench an AI judge caught @J_IB@\% with no training; v1 caught 0\%.\\[4pt]
\textbf{New agents are harder:} @AG_LO@\% to @AG_HI@\% caught.\\[4pt]
\textbf{Labels are batches.} Cheating and honest runs come from separate batches; other batch differences could help it.\\[4pt]
\textbf{Small samples.} ImpossibleBench has 32 cheating runs. On MALT, 345 of 407 misbehaving agents were told to misbehave.\\[4pt]
\textbf{Unpaired judge comparison.} GPT-5.4's 44\% is the benchmark's published figure.\\[4pt]
\textbf{Not attacked yet.} Not tested against an agent that knows about it and tries to evade it.};
\end{tikzpicture}}\end{center}

% ---------- appendix ----------
\newpage
{\fontsize{20}{24}\selectfont\bfseries Every number, by benchmark}\par\vspace{2mm}
\small\emph{Caught}: share of cheating runs flagged when 5 in 100 honest runs are flagged. \emph{AUC}: chance a random cheating run
outscores a random honest one (0.5 = guessing, 1 = perfect). Brackets: 95\% bootstrap ranges. The agent's own words are deleted in every run.
Every score comes from a model that never trained on that run's job.

\begin{minipage}[t]{0.49\linewidth}\small
\textbf{Benchmarks}\\[2pt]
\begin{tabular}{lrr}\toprule Benchmark & runs & cheating \\\midrule
Terminal Wrench \cite{tw} & @NTW@ & @CTW@ \\ MALT \cite{malt} (all 407 + 1,500 of 6,047 normal) & @NMALT@ & 407 \\ ImpossibleBench \cite{ib} & 205 & 32 \\\bottomrule\end{tabular}

\vspace{3mm}\textbf{Terminal Wrench, jobs never trained on}\\[2pt]
\begin{tabular}{lcc}\toprule Setting & AUC & caught (\%) \\\midrule
Full system (upper bound) & @BEST_AUC@ & @BEST@ @BESTCI@ \\
Retrained, 33 jobs locked away; 99 other jobs & @VAL_AUC@ & @VAL@ @VALCI@ \\
Simple reader & -- & @SIMPLE@ \\
Earlier version, 629 never-used runs & @HOLD_AUC@ & @HOLD@ @HOLDCI@ \\
No labels (written descriptions only) & @ZERO_AUC@ & @ZERO@ \\
@AGENTS@
\midrule GPT-5.4 judge, published & 0.917 & 44 \\\bottomrule\end{tabular}
\end{minipage}\hfill
\begin{minipage}[t]{0.47\linewidth}\small
\textbf{Labelled jobs (Terminal Wrench)}, caught (\%): median of 10 random picks, worst pick shown\\[2pt]
\begin{tabular}{lccc}\toprule Jobs & simple & simple, worst & full (worst) \\\midrule
@CURVE@
\bottomrule\end{tabular}

\vspace{3mm}\textbf{Cost.} Embedding: @ENC@ s per run on a laptop. AI judge: about 6,100 word pieces and about @JUDGE@ s per run.
\end{minipage}

\newpage
\textbf{Across benchmarks.} Simple reader (average and peak of the step embeddings). ``Own line'': alarm line carried over from training,
as in real use. ``Best line'': set on the test benchmark's honest runs, which real use could not do.\\[2pt]
{\small\begin{tabular}{lccc}\toprule Trained on $\to$ tested on & AUC & caught, own line & caught, best line \\\midrule
@CROSS@
\bottomrule\end{tabular}}

\small Earlier versions trained on Terminal Wrench and applied unchanged: MALT AUC @MALT_AUC@ @MALT_CI@ (caught @MALT_C@\%), ImpossibleBench
@IB_AUC@ @IB_CI@ (caught @IB_C@\%). An AI judge using the Terminal Wrench judge prompt caught @J_IB@\% on ImpossibleBench (AUC @J_IB_AUC@).
Code, data loaders and every result file: \url{@REPO@}. Rebuild: \texttt{python3 paper/gen\_figs.py \&\& python3 paper/gen\_paper.py}.

\begin{thebibliography}{9}\footnotesize
\bibitem{tw} I.~Bercovich et al. \emph{Terminal Wrench: a dataset of 331 reward-hackable environments and 3,632 exploit trajectories}. 2026.
\bibitem{malt} METR. \emph{MALT: manually-reviewed agentic labeled transcripts}. 2025.
\bibitem{ib} Z.~Zhong, A.~Raghunathan, N.~Carlini. \emph{ImpossibleBench: measuring LLMs' propensity of exploiting test cases}. 2025.
\bibitem{bge} S.~Xiao et al. \emph{C-Pack: packaged resources to advance general Chinese embedding}. arXiv:2309.07597, 2023. (bge-small model)
\end{thebibliography}
\end{document}
"""
V.update(SIMPLE_R=p0(M["265"]["caught"]), BEST_R=p0(FIN["tpr5"]))
V["JUDGE"] = str(JUDGE_S)
for k in sorted(V, key=len, reverse=True):
    TEX = TEX.replace(f"@{k}@", V[k])
assert "@" not in TEX.replace("\\url{" + REPO + "}", ""), [l for l in TEX.split("\n") if "@" in l and REPO not in l][:3]
open("paper/paper.tex", "w").write(TEX)
