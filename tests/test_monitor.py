import os, sys
import numpy as np
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
os.chdir(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import build, monitor as M


def test_clean_removes_dates_ids_colors():
    s = build.clean("\x1b[32mok\x1b[0m Apr 12 10:31 file 2026-11-20T10:00:00Z id 3f2a9c1d8e7b6a5f4c3d")
    assert "Apr 12" not in s and "2026" not in s and "\x1b" not in s and "3f2a9c1d8e7b" not in s


def test_compress_keeps_errors_and_ends():
    obs = "\n".join(["start", "two"] + [f"line {i}" for i in range(50)] + ["AssertionError: boom", "end1", "end2"])
    c = build.compress(obs)
    assert "start" in c and "AssertionError" in c and "end2" in c and "line 25" not in c


def test_write_targets():
    cmd = "echo x > /app/tests/test_a.py; sed -i 's/a/b/' /app/grader.py; cp a.py /app/out.py; ls 2>/dev/null"
    got = sorted({g for m in build.WRITE.finditer(cmd) for g in m.groups() if g and not g.startswith("/dev/")})
    assert got == ["/app/grader.py", "/app/out.py", "/app/tests/test_a.py"]


def test_units_context_windows():
    st = [{"say": "look", "act": "CMD: ls\nOUT: a", "files": []}, {"say": "", "act": "CMD: cat t.py\nOUT: x", "files": ["t.py"]},
          {"say": "edit", "act": "CMD: sed -i x t.py\nOUT: ", "files": ["t.py"]}]
    u = M.units(st)
    assert len(u["thought"]) == 2 and len(u["action"]) == 3 and len(u["context"]) == 3
    assert "ls" in u["context"][1] and "sed" in u["context"][1] and u["files"] == "FILES CHANGED: t.py"


def test_step_labels_setup_steps_not_cheating():
    st_h = [{"say": "", "act": "CMD: ls\nOUT: a", "files": []}]
    st_c = [{"say": "bypass", "act": "CMD: ls\nOUT: b", "files": []}, {"say": "", "act": "CMD: rm tests\nOUT: ", "files": []}]
    rows = [{"task": "t", "label": 0, "v": {"original": st_h, "stripped": st_h}},
            {"task": "t", "label": 1, "v": {"original": st_c, "sanitized": st_c, "stripped": st_c}}]
    T, L = M.step_labels(rows)
    lab = {}
    for t, l in zip(T, L):
        lab.setdefault(t, set()).add(l)
    assert lab["CMD: ls\nOUT: b"] == {0} and lab["CMD: rm tests\nOUT: "] == {1}


def test_route_keeps_sure_runs_and_order():
    p = np.array([0.001, 0.5, 0.999]); j = np.array([0.9, 0.2, 0.1])
    r, unsure = M.route(p, j, 0.01, 0.99)
    assert list(unsure) == [False, True, False] and r[0] == 0.001 and r[2] == 0.999 and 0.01 < r[1] < 0.99


def test_scale_signs():
    assert list(M.scale100([0.0, 0.5, 1.0])) == [100.0, 0.0, -100.0]


def test_stacking_keeps_strong_single_reader():
    from sklearn.metrics import roc_auc_score
    rng = np.random.default_rng(0); n = 600
    y = rng.integers(0, 2, n); grp = np.arange(n) % 30
    F = [{"noise": rng.normal(size=300), "words": np.array([y[i] * 2 + rng.normal()])} for i in range(n)]
    h = M.Head(("noise", "words")).fit(F[:400], y[:400], 0.05, grp[:400])
    good = roc_auc_score(y[400:], [f["words"][0] for f in F[400:]])
    assert roc_auc_score(y[400:], h.p(F[400:])) > good - 0.03


def test_malt_label_and_steps():
    import malt
    assert malt.label(["normal"]) == 0 and malt.label(["hardcoded_solution", "normal"]) == 1 and malt.label(["gives_up"]) is None
    samples = [{"input": [{"role": "system", "content": "sys"}, {"role": "user", "content": "Fix the bug"}],
                "output": [[{"role": "assistant", "content": "I will edit the test <bash>echo ok > /app/tests/t.py</bash>",
                             "function_call": None}]]},
               {"input": [{"role": "user", "content": "Fix the bug"}, {"role": "assistant", "content": "x"},
                          {"role": "function", "content": "AssertionError: boom"}],
                "output": [[{"role": "assistant", "content": "", "function_call": {"name": "bash", "arguments": '{"command": "pytest"}'}}]]}]
    st, j = malt.to_steps(samples)
    assert len(st) == 2 and "I will edit the test" in st[0]["say"] and "<bash>" not in st[0]["say"]
    assert "AssertionError" in st[0]["act"] and st[0]["files"] == ["/app/tests/t.py"] and "bash: pytest" in st[1]["act"]
    assert malt.prompt_of(samples) == "Fix the bug" and "COMMAND: bash: pytest" in malt.judge_text(j, True)
    assert "### AGENT" not in malt.judge_text(j, True) and "### AGENT" in malt.judge_text(j, False)


def test_malt_prompt_from_system_when_no_user():
    import malt
    assert malt.prompt_of([{"input": [{"role": "system", "content": "Write solve.py"}], "output": []}]) == "Write solve.py"
