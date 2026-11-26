"""Paranoid Android v1 paper (plain language, all benchmark results in the appendix). Every number is read from
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

TEX = r"""\documentclass[11pt,a4paper]{article}
\usepackage[margin=2.3cm]{geometry}\usepackage[T1]{fontenc}\usepackage{tgheros}\renewcommand{\familydefault}{\sfdefault}
\usepackage{microtype,booktabs,graphicx,pdflscape,xcolor,enumitem}\usepackage[hidelinks]{hyperref}
\setlist{itemsep=3pt,topsep=4pt}\setlength{\parskip}{5pt}\setlength{\parindent}{0pt}\setlength{\tabcolsep}{4pt}
\title{\textbf{Paranoid Android v1}\\[3pt]\Large Catching AI agents that cheat, in a few hundredths of a second}
\author{Questboy23\\\small\url{@REPO@}}\date{November 2026}
\begin{document}\maketitle
\begin{abstract}\noindent
AI agents that write code sometimes get a passing grade without doing the work, for example by copying the answer
into the output file. The usual way to catch this is to have a second AI read every run, which takes about @JUDGE@ seconds
a run. Paranoid Android v1 does it in about @ENC@ seconds on a laptop, roughly @SPEED@ times faster, using a small free
word-embedding model and a few simple scorers. On Terminal Wrench, the benchmark it learned from, it catches somewhere
between @SIMPLE@\% and @BEST@\% of cheating runs on jobs it has never seen, while flagging 5\% of honest runs. The benchmark's
published GPT-5.4 judge catches 44\% at the same setting. It also learns quickly: labels on 25 jobs are enough for @L25@\%.
The catch is that it only knows the kinds of cheating it has seen. On two benchmarks it never trained on, it scored
an AUC of @ONE_LO@ to @ONE_HI@ (0.5 is guessing) and caught almost none of the cheating runs.
\end{abstract}

\section{Why we built it}
When an AI agent is graded by an automatic checker, it can learn to fool the checker instead of solving the task. Labs
already watch for this by asking another AI model to read each run and say whether it looks like cheating. That works, but
it is slow and it costs money for every run, so in practice only a sample of runs gets read. We wanted to know how much of
that job a much cheaper tool could do, so that every run can be checked and the expensive judge can be saved for the
suspicious ones.

\section{How it works}
Figure~\ref{fig:tree} (last page) shows the whole system on one page. In words:

\textbf{1. Start from the saved run.} This is the job description, every command the agent typed, and what the computer
printed back. We delete the agent's own words (its notes and explanations) first. An agent that is cheating will not always
say so, and we did not want the tool to depend on it.

\textbf{2. Turn the text into numbers.} A word-embedding model (bge-small, 33 million parameters, free to download) reads
each step and turns it into a list of 384 numbers. Steps that mean similar things get similar lists. ``Copy the expected
output into the answer file'' lands near other shortcuts, and far from ``run the tests and fix the bug''.

\textbf{3. Seven readers each give a score.} They look at the run in different ways: short letter patterns in the raw text,
averages and peaks of the embedding numbers, whether each step resembles steps from known cheats, which few steps give
the cheat away, and a copy of the embedding model retrained on cheating examples.

\textbf{4. Add the scores and compare to a line.} Each score gets a learned weight and they are summed. If the total is above
a line, the run is flagged. We set the line so that 5 in every 100 honest runs get flagged; that is the price of catching cheats.

\textbf{What ``labelled'' means.} The readers and weights learn from past runs that are already marked as cheating or honest.
In Terminal Wrench, the authors ran agents twice: once asking them to find exploits (3,632 runs, marked cheating) and once
normally (2,352 runs, marked honest). Those marks are the labels.

\section{How we tested it}
We used three public benchmarks (details in Appendix~A): Terminal Wrench \cite{tw} (@NTW@ runs on 331 jobs, @CTW@ cheating),
MALT \cite{malt} (@NMALT@ runs with sabotage, sandbagging and similar behaviour, labelled by people), and ImpossibleBench \cite{ib}
(205 coding runs, 32 of which passed tests that cannot be passed honestly). The rule throughout: a run is only ever scored
by a version of the tool that never saw that job. We split the jobs into 5 groups and score each group with a tool trained
on the other four. ``Caught'' means the share of cheating runs above the line.

\section{What works}
\textbf{It is very fast.} Turning a run into numbers takes about @ENC@ seconds on an ordinary laptop. An AI judge takes about
@JUDGE@ seconds and reads around 6,100 word pieces per run. For a million runs that is about @HOURS@ hours against about @DAYS@
days on one machine. The model is free and runs offline, so checking every run, instead of a sample, becomes realistic.

\textbf{It catches more than the judge on what it knows.} On Terminal Wrench, on jobs it never trained on, it catches between
@SIMPLE@\% and @BEST@\% of cheating runs, depending on how strict we are about the estimate (Section~6). In a later check where
33 jobs were locked away and every reader retrained without them, the full system caught @VAL@\% on 99 other jobs. The
published GPT-5.4 judge catches 44\% at the same false-alarm rate.

\textbf{It works on new examples of familiar cheating.} The question that matters most is whether it handles cheating it
has not seen before. When it is trained on some jobs from a benchmark and tested on different jobs from the same benchmark,
it does well on all three: an AUC of @S_TW@ on Terminal Wrench, @S_MALT@ on MALT and @S_IB@ on ImpossibleBench (1.0 would be
perfect, 0.5 is guessing). So new cases of a kind of misbehaviour it has learned are caught, even on jobs it has never seen.

\textbf{It needs few labels.} With labels for only 25 jobs it catches about @L25@\%. Most of the gain comes early.

\textbf{It does not rely on the agent confessing.} All of the above is with the agent's own words deleted.

\section{What does not work, and other limits}
We want to be plain about where this falls short.
\begin{enumerate}
\item \textbf{The scores are good, not great.} Even in the best case it misses about one cheating run in five, and it flags
5\% of honest runs to get there.
\item \textbf{The @BEST@\% figure is an upper bound.} No run was ever scored by a model that trained on its job, so there is no
direct leak. But we chose the seven readers while looking at these same runs, and that can flatter the result. The fairer
estimates are @SIMPLE@\% (a simple reader with no such choices), @HOLD@\% (an earlier version on 629 runs that were never
used for any choice) and @VAL@\% (the retrained system on 99 jobs).
\item \textbf{It does not carry over to new kinds of misbehaviour.} Trained on one benchmark and tested on another, it scored
an AUC of @ONE_LO@ to @ONE_HI@. With the alarm line it brings along, it caught 0\% of cheating runs on MALT and ImpossibleBench.
Training on two benchmarks and testing on the third did not help (@TWO_LO@ to @TWO_HI@). An AI judge, by contrast, caught
@J_IB@\% on ImpossibleBench without any training. Each new kind of misbehaviour needs its own labelled runs.
\item \textbf{New agents are harder.} On runs from an AI model it never learned from, it caught between @AG_LO@\% and @AG_HI@\%.
\item \textbf{The labels are batches.} Cheating and honest runs come from two separate batches, so anything else that differs
between the batches could help the tool without being cheating. We did not rule this out.
\item \textbf{Small samples outside Terminal Wrench.} ImpossibleBench has only 32 cheating runs, so its numbers are rough. On MALT, 345 of
the 407 misbehaving runs came from agents that were told to misbehave.
\item \textbf{The judge comparison is not run by run.} GPT-5.4's per-run scores are not public, so its 44\% is the benchmark's own published figure.
\item \textbf{Nobody has attacked it yet.} We have not tested it against an agent that knows about the tool and tries to slip past it.
\end{enumerate}

\section{Where it fits}
Paranoid Android v1 is best used as a cheap first filter on a setting where some labelled runs exist: check every run in a
fraction of a second, and send the flagged ones to an AI judge or a person. It should not be trusted on its own for new
kinds of misbehaviour. Version 2, in progress, adds an AI judge and rule-based checks on top of this layer for that reason.

\textbf{Code, data and every number:} \url{@REPO@}. All figures in this paper are produced from the result files by \texttt{paper/gen\_paper.py}.

\begin{thebibliography}{9}\small
\bibitem{tw} I.~Bercovich et al. \emph{Terminal Wrench: a dataset of 331 reward-hackable environments and 3,632 exploit trajectories}. 2026.
\bibitem{malt} METR. \emph{MALT: manually-reviewed agentic labeled transcripts}. 2025.
\bibitem{ib} Z.~Zhong, A.~Raghunathan, N.~Carlini. \emph{ImpossibleBench: measuring LLMs' propensity of exploiting test cases}. 2025.
\bibitem{bge} S.~Xiao et al. \emph{C-Pack: packaged resources to advance general Chinese embedding}. arXiv:2309.07597, 2023.
\end{thebibliography}

\clearpage\appendix
\section{All benchmark results}
\textbf{Terms.} \emph{Caught}: share of cheating runs flagged when the line is set so 5\% of honest runs are flagged.
\emph{AUC}: the chance a random cheating run scores higher than a random honest run (0.5 = guessing, 1 = perfect).
Brackets are 95\% bootstrap intervals. All runs have the agent's own words deleted.

\subsection*{A.1 Benchmarks}
\begin{center}\small\begin{tabular}{lrrp{8cm}}\toprule
Benchmark & runs & cheating & what counts as cheating \\\midrule
Terminal Wrench & @NTW@ & @CTW@ & passing a grader on 331 jobs without doing the job \\
MALT & @NMALT@ & 407 & sabotage, sandbagging, hard-coding, ignoring instructions (scored on all 407 and 1,500 of 6,047 normal runs) \\
ImpossibleBench & 205 & 32 & passing coding tests that contradict the task \\\bottomrule
\end{tabular}\end{center}

\subsection*{A.2 Terminal Wrench, jobs never trained on}
\begin{center}\small\begin{tabular}{lcc}\toprule
Setting & AUC & caught (\%) \\\midrule
Full system, all 5,984 runs (upper bound) & @BEST_AUC@ @BEST_AUCCI@ & @BEST@ @BESTCI@ \\
Full system retrained without 33 locked jobs, on 99 other jobs & @VAL_AUC@ & @VAL@ @VALCI@ \\
Simple reader & -- & @SIMPLE@ \\
Earlier version, 629 runs never used for any choice & @HOLD_AUC@ @HOLD_AUCCI@ & @HOLD@ @HOLDCI@ \\
No labels at all (written descriptions of cheating only) & @ZERO_AUC@ & @ZERO@ \\
@AGENTS@
\midrule
GPT-5.4 judge, published by the benchmark (all runs) & 0.917 & 44 \\\bottomrule
\end{tabular}\end{center}

\subsection*{A.3 How many labelled jobs it needs (Terminal Wrench)}
Caught (\%) with labels from only $k$ jobs (about 18 runs each), tested on all other jobs. Median of 10 random picks; worst pick shown.
\begin{center}\small\begin{tabular}{lccc}\toprule
Labelled jobs & simple reader & simple, worst pick & full system (worst pick) \\\midrule
@CURVE@
\bottomrule\end{tabular}\end{center}

\subsection*{A.4 Across benchmarks}
Simple reader (average and peak of the step embeddings). ``Own line'' carries the alarm line over from training, as a real
deployment would. ``Best line'' sets it on the test benchmark's honest runs, which a deployment could not do.
\begin{center}\small\begin{tabular}{lccc}\toprule
Trained on $\to$ tested on & AUC & caught, own line & caught, best line \\\midrule
@CROSS@
\bottomrule\end{tabular}\end{center}
Earlier versions trained on Terminal Wrench and applied unchanged: MALT AUC @MALT_AUC@ @MALT_CI@ (caught @MALT_C@\%), ImpossibleBench
@IB_AUC@ @IB_CI@ (caught @IB_C@\%). An AI judge using the Terminal Wrench judge prompt caught @J_IB@\% on ImpossibleBench (AUC @J_IB_AUC@).

\subsection*{A.5 Cost}
Embedding: @ENC@ seconds per run (bge-small, on a laptop). AI judge: about 6,100 word pieces and about @JUDGE@ seconds per run.

\newgeometry{margin=0.8cm}\begin{landscape}\thispagestyle{empty}\begin{figure}[p]\centering
\includegraphics[width=\linewidth,height=0.95\textheight,keepaspectratio,trim=0 3cm 0 0,clip]{tree.pdf}
\caption{The whole system on one page: a saved run flows down to the alarm; the right side sums up the results.}\label{fig:tree}
\end{figure}\end{landscape}\restoregeometry
\end{document}
"""
V["JUDGE"] = str(JUDGE_S)
for k in sorted(V, key=len, reverse=True):
    TEX = TEX.replace(f"@{k}@", V[k])
assert "@" not in TEX.replace("\\url{" + REPO + "}", ""), [l for l in TEX.split("\n") if "@" in l and REPO not in l][:3]
open("paper/paper.tex", "w").write(TEX)
