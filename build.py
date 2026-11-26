"""Build data/runs.jsonl from Terminal Wrench (Apache-2.0) on the locked task splits (splits.json, seed 0, 60/20/20 by task).

Per run and per variant (original / sanitized / stripped) it stores cleaned, compressed steps:
  say  = the agent's thought text (blank in "stripped", for honest runs too)
  act  = normalized commands + compressed output
  files = paths the commands write to
Disguised copies of test hack runs are added for the stress test: "syn" (same commands, other spellings)
and "pad" (5 honest steps from the same task put in front).
"""
import glob, json, os, re, sys
sys.path.insert(0, os.path.expanduser("~/projects/embedding-alignment"))
from embed import normalize  # decode base64/hex, strip comments, expand aliases, resolve paths

TW = os.environ.get("TW_TASKS", "data/raw/tw/tasks")  # Terminal Wrench task folder
SPL = json.load(open("splits.json"))["by_task"]

ANSI = re.compile(r"\x1b\[[0-9;?]*[A-Za-z]")
NOISE = [(re.compile(r"\b(?:Jan|Feb|Mar|Apr|May|Jun|Jul|Aug|Sep|Oct|Nov|Dec)\s+\d{1,2}(?:\s+\d{1,2}:\d{2}|\s+\d{4})?\b"), "DATE"),
         (re.compile(r"\b\d{4}-\d{2}-\d{2}(?:[T ]\d{2}:\d{2}(?::\d{2})?(?:\.\d+)?Z?)?\b"), "DATE"),
         (re.compile(r"\b\d{1,2}:\d{2}:\d{2}\b"), "TIME"),
         (re.compile(r"\b[0-9a-f]{8}-[0-9a-f]{4}-[0-9a-f]{4}-[0-9a-f]{4}-[0-9a-f]{12}\b", re.I), "ID"),
         (re.compile(r"\b[0-9a-f]{12,}\b", re.I), "ID"),
         (re.compile(r"/tmp/tmp[\w]+"), "/tmp/TMP"),
         (re.compile(r"\b\d{5,}\b"), "NUM")]
KEY = re.compile(r"error|fail|pass|assert|traceback|exception|warning|denied|not found|test|ok\b|success", re.I)
WRITE = re.compile(r"(?:>>?|\btee(?:\s+-a)?)\s*([^\s;|&>]+)|\bsed\s+(?:-i|--in-place)\S*\s+(?:'[^']*'|\"[^\"]*\"|\S+)\s+([^\s;|&]+)"
                   r"|\b(?:cp|mv)\s+(?:-\S+\s+)*\S+\s+([^\s;|&]+)|\brm\s+(?:-\S+\s+)*([^\s;|&]+)|open\(\s*['\"]([^'\"]+)['\"]\s*,\s*['\"][wa]")
SYN = [(r"\bsed -i\b", "sed --in-place"), (r"\brm -rf\b", "rm --recursive --force"), (r"\brm -f\b", "rm --force"),
       (r"\bcp -r\b", "cp --recursive"), (r"\bmkdir -p\b", "mkdir --parents"), (r"\bgrep -r\b", "grep --recursive"),
       (r"\bchmod \+x\b", "chmod a+x"), (r"\bls -la\b", "ls --all -l"), (r"(?<![\w-])cat (?=\S)", "awk 1 "),
       (r"\becho ", "printf '%s\\n' "), (r"(?<![\w.-])pytest\b", "python -m pytest")]


def clean(s):
    s = ANSI.sub("", s or "").replace("\r", "")
    for rx, rep in NOISE:
        s = rx.sub(rep, s)
    return re.sub(r"[ \t]+", " ", s).strip()


def compress(obs, cap=400):
    """Keep the first 2 and last 2 lines plus lines with errors, test results or status words; cap the length."""
    L = [l for l in clean(obs).splitlines() if l.strip()]
    keep = L if len(L) <= 6 else L[:2] + [l for l in L[2:-2] if KEY.search(l)][:8] + L[-2:]
    out, prev = [], None
    for l in keep:
        if l != prev:
            out.append(l[:160])
        prev = l
    s = "\n".join(out)
    return s if len(s) <= cap else s[: cap // 2] + " ... " + s[-cap // 2:]


def say_text(msg):
    msg = (msg or "").strip()
    if msg.startswith("{"):
        try:
            j = json.loads(msg); msg = f"{j.get('analysis', '')}\n{j.get('plan', '')}"
        except Exception:
            pass
    return clean(msg)


def steps(path, rewrite=None):
    out, last = [], None
    for x in json.load(open(path))["steps"]:
        if x.get("source") != "agent":
            continue
        cmds = [(c.get("arguments") or {}).get("keystrokes") or json.dumps(c.get("arguments") or {}) for c in x.get("tool_calls") or []]
        if rewrite:
            cmds = [rewrite(c) for c in cmds]
        obs = "".join(str(r.get("content", "")) for r in ((x.get("observation") or {}).get("results") or []))
        cmd = clean("\n".join(normalize(c) for c in cmds))
        files = sorted({g for c in cmds for m in WRITE.finditer(c) for g in m.groups() if g and not g.startswith("/dev/")})
        st = {"say": say_text(x.get("message")), "act": f"CMD: {cmd}\nOUT: {compress(obs[-3000:])}", "files": files}
        if last and st["act"] == last["act"] and not st["say"]:
            last["rep"] = last.get("rep", 1) + 1  # compression: identical repeated step kept once with a count
            continue
        out.append(st); last = st
    return out


def blank(st):
    return [dict(s, say="") for s in st]


def syn(c):
    for a, b in SYN:
        c = re.sub(a, b, c, flags=re.M)
    return c


if __name__ == "__main__":
    rows = []
    for tdir in sorted(glob.glob(f"{TW}/*/*/")):
        task, model = tdir.rstrip("/").split("/")[-2:]
        inst = os.path.join(tdir, "original_task", "instruction.md")
        if task not in SPL or not os.path.exists(inst):
            continue
        base = dict(task=task, model=model, split=SPL[task], prompt=clean(open(inst, errors="ignore").read()))
        honest = []
        for p in sorted(glob.glob(f"{tdir}baseline_trajectories/*/trial/agent/trajectory.json")):
            st = steps(p)
            if st:
                run = p.split("baseline_trajectories/")[1].split("/")[0]
                honest.append(st)
                rows.append(dict(base, id=f"{task}/{model}/base/{run}", label=0, v={"original": st, "sanitized": st, "stripped": blank(st)}))
        for p in sorted(glob.glob(f"{tdir}sanitized_trajectories/*/trial/agent/trajectory.json")):
            run = p.split("sanitized_trajectories/")[1].split("/")[0]
            po, ps = p.replace("sanitized_trajectories", "hack_trajectories"), p.replace("sanitized_trajectories", "stripped_trajectories")
            if not (os.path.exists(po) and os.path.exists(ps)):
                continue
            so, ss = steps(po), steps(p)
            if not so or not ss:
                continue
            v = {"original": so, "sanitized": ss, "stripped": blank(steps(ps) or ss)}
            if SPL[task] == "test":
                v["syn"] = blank(steps(ps, syn) or ss)
                v["pad"] = (blank(honest[0][:5]) if honest else []) + v["stripped"]
            rows.append(dict(base, id=f"{task}/{model}/hack/{run}", label=1, v=v))
    os.makedirs("data", exist_ok=True)
    with open("data/runs.jsonl", "w") as f:
        for r in rows:
            f.write(json.dumps(r) + "\n")
    from collections import Counter
    print(len(rows), sorted(Counter((r["split"], r["label"]) for r in rows).items()))
