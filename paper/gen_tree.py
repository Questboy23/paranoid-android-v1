"""One-page tree diagram of version 1 (A3 landscape): how a run flows to the alarm, where learning comes from, and how
well it works (including the leave-one-benchmark-out test). Numbers read from results/*.json and the v2 repo's FINAL json.
Run from repo root: .venv/bin/python paper/gen_tree.py && cd paper && tectonic tree.tex"""
import json, os
F = json.load(open("results/v5_fair.json")); M = json.load(open("results/v5_fair_more.json")); FIN = json.load(open("results/v5_final_stripped.json"))
X = json.load(open("results/cross_bench.json")); P = json.load(open("paper/data.json"))
V3 = json.load(open(os.path.expanduser("results/v2_final.json")))
ib_v1 = V3["E_impossiblebench"]["v2_embed"]["auc"]  # earlier v1 head on ImpossibleBench, trained on Terminal Wrench only
malt_v1 = 0.493  # RESULTS.md / paper/numbers.tex: earlier v1 on MALT, agent's words deleted
one = [X["train tw -> test malt"]["auc"], X["train tw -> test ib"]["auc"], ib_v1, malt_v1]
two = [X[k]["auc"] for k in ("train tw+malt -> test ib", "train tw+ib -> test malt", "train malt+ib -> test tw")]
zero_caught = max(X["train tw -> test malt"]["caught_line_from_train"], X["train tw -> test ib"]["caught_line_from_train"])
pc = lambda v: f"{100 * v:.0f}"
R = P["logs"]["1280"]; LINE = P["line"]
v = dict(best=pc(FIN["tpr5"]), likely=pc(M["265"]["caught"]), l0=pc(F["zero_shot"]["caught"]), l10=pc(F["few_tasks"]["10"]["caught_median"]),
         ag_lo=pc(F["new_agent"]["gpt-5.4"]["caught"]), ag_hi=pc(F["new_agent"]["claude-opus-4.6"]["caught"]),
         one_lo=f"{min(one):.2f}", one_hi=f"{max(one):.2f}", two_lo=f"{min(two):.2f}", two_hi=f"{max(two):.2f}",
         zc=pc(zero_caught), z=f"{R['z']:.2f}", line=f"{LINE:.2f}")
assert zero_caught == 0.0 and max(two) < 0.65

TEX = r"""\documentclass[11pt]{article}
\usepackage[a3paper,landscape,margin=1.1cm]{geometry}
\usepackage[T1]{fontenc}\usepackage{tgheros}\renewcommand{\familydefault}{\sfdefault}\usepackage{microtype}
\usepackage{xcolor}\usepackage{graphicx}\usepackage{tikz}\usetikzlibrary{arrows.meta,positioning,calc}
\pagestyle{empty}\setlength{\parindent}{0pt}
\definecolor{ink}{HTML}{1B1F27}\definecolor{cheat}{HTML}{C0392B}\definecolor{honest}{HTML}{2D6CB0}\definecolor{gold}{HTML}{B87600}
\definecolor{c1}{HTML}{EEF1F6}\definecolor{c2}{HTML}{E6F0E4}\definecolor{c3}{HTML}{FBEFD9}\definecolor{c4}{HTML}{F8E1DD}
\tikzset{
 n/.style={draw=ink!45,rounded corners=5pt,fill=c1,align=center,inner sep=6pt,font=\normalsize},
 r/.style={n,fill=c2,text width=3.05cm,font=\small,minimum height=2.5cm},
 res/.style={n,fill=white,draw=ink!35,align=left,text width=10.2cm,font=\normalsize},
 e/.style={-{Stealth[length=3mm]},line width=1.3pt,ink!55},
 learn/.style={-{Stealth[length=3mm]},line width=1.3pt,gold,dashed},
 t/.style={font=\Large\bfseries}, ex/.style={font=\small\itshape,text=ink!60}}
\begin{document}
{\fontsize{26}{30}\selectfont\bfseries Version 1, the whole system in one tree}\hfill{\large\color{ink!60}Paranoid Android: catches AI agents that cheat a grader}\\[2mm]
{\large Follow the arrows from the top. \textbf{Cheat} = the agent makes the grader say ``pass'' without doing the job. Grey italics = one real cheat run going through.}

\vspace{3mm}
\begin{center}\resizebox{!}{23.6cm}{\begin{tikzpicture}[x=1cm,y=1cm]
% ---------------- main tree ----------------
\node[n,text width=10cm] (log) at (13.45,0) {{\Large\bfseries A saved run}\\ the job, every command the agent typed, and the computer's replies.\\ The agent's own words are deleted first.\\[2pt]
 {\small\itshape\color{ink!60} \texttt{cat expected\_output.txt > output.txt}\, = copy the answer key into the output}};

\node[n,text width=6cm] (raw) at (4.2,-4.3) {{\large\bfseries The raw text}\\ the same letters and words, unchanged};
\node[n,text width=9cm] (tm) at (17.6,-4.3) {{\large\bfseries A word-embedding model reads it} (bge-small, free)\\ cuts the text into word pieces; in 12 rounds every piece looks at all the others to work out what it means in context};
\node[n,text width=9cm] (num) at (17.6,-7.9) {{\large\bfseries The embedding: a list of 384 numbers}\\ one list per step, per group of 3 steps, and for the job text.\\ Texts with similar meaning get similar lists.};
\draw[e] (log.south) -- ++(0,-.9) -| (raw.north);
\draw[e] (log.south) -- ++(0,-.9) -| (tm.north);
\draw[e] (tm) -- (num);

% readers
\node[rotate=90,font=\Large\bfseries,anchor=south] at (0.3,-13.2) {7 readers};
\foreach \i/\x/\name/\txt in {
 B/2.35/Letter patterns/{counts short chunks of 3 or 4 letters, with numbers blanked out},
 A/6.05/Meaning bundle/{averages and peaks of the lists, how close the run is to written descriptions of cheating, word counts},
 D/9.75/Nearest steps/{is each step more like steps from known cheats or from honest runs?},
 E/13.45/Step picker/{learns which few steps give the cheat away},
 F/17.15/{Step picker, padded}/{same, trained so added honest steps cannot hide a cheat},
 G/20.85/Step picker + job/{same, also reads the job text},
 C/24.55/Retrained model/{a copy of the embedding model, retrained on labelled runs to spot cheats}}
 { \node[r] (\i) at (\x,-13.2) {{\bfseries \name}\\[2pt] \txt}; }
\draw[e] (raw.south) -- (raw.south |- 0,-10.0) -| (B.north);
\draw[e] (raw.south) -- (raw.south |- 0,-10.0) -| (A.north);
\foreach \i in {A,D,E,F,G} { \draw[e] (num.south) -- ++(0,-1.75) -| (\i.north); }
\draw[e] (tm.east) -| (C.north);

% add up
\node[n,fill=c3,text width=9.5cm] (add) at (13.45,-17.9) {{\Large\bfseries Add up}\\ each reader gives a score; each score $\times$ its learned weight, added together $=$ the \textbf{total}\\ {\small\itshape\color{ink!60} example total: """ + v["z"] + r"""}};
\foreach \i in {A,B,C,D,E,F,G} { \draw[e] (\i.south) -- (\i.south |- 0,-15.4) -| (add.north); }
\node[font=\small,text=ink!60,anchor=south west] at (B.south |- 0,-15.4) {scores};

% learning
\node[n,fill=c3,draw=gold,text width=6.4cm,align=left] (lab) at (3.9,-20.9) {{\large\bfseries\color{gold} Where the learning comes from}\\ \textbf{Labelled runs}: past runs already marked cheat or honest (the benchmark makers' ``hack'' and ``base'' batches). Readers and weights learn from them, never from the job being tested.};
\draw[learn] (lab.east) -- ++(.55,0) |- node[pos=.75,above,font=\small,text=gold]{weights} (add.west);
\draw[learn] (lab.north) -- node[right,font=\small,text=gold,pos=.35]{readers} (lab.north |- 0,-15.4);

% alarm
\node[n,fill=c4,text width=9.5cm] (line) at (13.45,-21.3) {{\Large\bfseries Alarm line}\\ placed so only 5 in 100 honest runs score above it\\ {\small\itshape\color{ink!60} line """ + v["line"] + r""": example """ + v["z"] + r""" is above it}};
\draw[e] (add) -- (line);
\node[n,fill=cheat,text=white,text width=4.2cm] (yes) at (10.05,-24.6) {{\Large\bfseries ALARM}\\ total above the line};
\node[n,fill=white,text width=4.2cm] (no) at (16.85,-24.6) {{\Large\bfseries no alarm}\\ total below the line};
\draw[e] (line.south) -- ++(0,-.6) -| (yes.north);
\draw[e] (line.south) -- ++(0,-.6) -| (no.north);

% ---------------- results tree ----------------
\draw[ink!20,line width=1pt] (27.2,.9) -- (27.2,-26);
\node[t,anchor=west] (hw) at (28.0,.4) {How well it works};
\node[anchor=west,font=\small,text=ink!65,text width=10.8cm] at (28.0,-.45) {Share of cheat runs that get the alarm, when 5 in 100 honest runs also get one. Always tested on jobs it never learned from.};
\coordinate (trunk) at (28.3,-1.0);
\foreach \k/\y/\c/\txt in {
 1/-2.4/c2/{\textbf{New jobs, same benchmark: likely """ + v["likely"] + r"""\% caught.}\\ Best case """ + v["best"] + r"""\%, but that is an overestimate: the 7-reader design was chosen while looking at these same runs. The best earlier method, an AI judge (GPT-5.4), catches 44\%.},
 2/-6.25/c2/{\textbf{More labelled jobs, more caught.}\\ 0 jobs: """ + v["l0"] + r"""\% \quad 10 jobs: """ + v["l10"] + r"""\% \quad all other jobs (about 265): """ + v["likely"] + r"""\%. One job $\approx$ 18 runs.},
 3/-9.35/c3/{\textbf{A new AI agent it never learned from:} """ + v["ag_lo"] + r"""\% to """ + v["ag_hi"] + r"""\% caught. Partly works.},
 4/-12.75/c4/{\textbf{A benchmark it never saw: does not work.}\\ Trained on Terminal Wrench only, tested on MALT and ImpossibleBench: score """ + v["one_lo"] + r""" to """ + v["one_hi"] + r""" (0.5 = guessing, 1 = perfect). The alarm line carried over catches """ + v["zc"] + r"""\%. An AI judge catches 94\% on ImpossibleBench.},
 5/-17.25/c4/{\textbf{Trained on two kinds of misbehaviour, tested on a third: still does not work.}\\ Score """ + v["two_lo"] + r""" to """ + v["two_hi"] + r""". Adding a second benchmark did not help.},
 6/-20.75/c1/{\textbf{Cost per run:} 0.03 seconds to make the numbers on a laptop. An AI judge: about 23 seconds.}}
 { \node[res,fill=\c,anchor=north west] (r\k) at (28.9,\y+1.0) {\txt}; \draw[ink!45,line width=1.2pt] (trunk) |- (r\k.west); }
\node[anchor=north west,font=\footnotesize,text=ink!60,text width=10.8cm,align=left] at (28.0,-23.0) {Score here = AUC: the chance that a random cheat run gets a higher total than a random honest run. Benchmarks: Terminal Wrench (5,984 runs, cheating a grader), MALT (1,907 runs: sabotage, sandbagging, hard-coding), ImpossibleBench (205 coding runs that pass impossible tests).};
\end{tikzpicture}}\end{center}
\end{document}
"""
open("paper/tree.tex", "w").write(TEX); print(v)
