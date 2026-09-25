"""scripts/: task runner, scrub audit, official fetcher, examples."""
import glob
import json
import os
import subprocess
import sys
import zipfile

import pytest

ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(
    __file__))))
SCRIPTS = os.path.join(ROOT, "scripts")
sys.path.insert(0, SCRIPTS)

import dev  # noqa: E402
import fetch_official  # noqa: E402
import scrub_audit  # noqa: E402


def test_dev_task_list_and_unknown(capsys):
    assert dev.main(["--list"]) == 0
    out = capsys.readouterr().out
    for t in ("build", "test", "certify", "bench", "clean", "distclean"):
        assert t in out
    assert dev.main(["no-such-task"]) == 2


def test_makefile_targets_match_dev_tasks():
    mk = open(os.path.join(ROOT, "Makefile"), encoding="utf-8").read()
    block = mk.split("TARGETS =", 1)[1].split("\n\n", 1)[0]
    targets = set(block.replace("\\", " ").split())
    assert targets == set(dev.TASKS), targets ^ set(dev.TASKS)


def test_docker_tasks_build_expected_commands(monkeypatch):
    calls = []

    class R:
        returncode = 0

    monkeypatch.setattr(dev.subprocess, "run",
                        lambda cmd, **kw: calls.append(cmd) or R())
    dev.docker_linux()
    dev.docker_test()
    assert calls[0][:3] == ["docker", "run", "--rm"]
    assert "x86_64-unknown-linux-musl" in calls[0][-1]
    assert any(a.startswith("KAGG_BIN=/w/target/linux/") for a in calls[1])
    assert "kaggle-environments==1.32.7" in calls[1][-1]


def test_scrub_audit_detects_and_passes(tmp_path):
    bad = tmp_path / "bad.txt"
    # assembled at run time so this test file itself stays clean
    path = "C:" + "\\" + "Users" + "\\someone\\x"
    tok = "to" + "ken = '" + "ABCDEFGHIJKLMNOPQRS'"
    mail = "person" + "@" + "realdomain.org"
    bad.write_text(f"{path}\nkey: {tok}\nmail me: {mail}\n"
                   "secret-word here\n")
    found = scrub_audit.audit([str(bad)], terms=["secret-word"])
    kinds = {f[2] for f in found}
    assert "windows absolute path" in kinds
    assert "e-mail address" in kinds
    assert "generic api key" in kinds
    assert "forbidden term 'secret-word'" in kinds
    ok = tmp_path / "ok.txt"
    ok.write_text("noreply@example.invalid\nnothing to see\n")
    assert scrub_audit.audit([str(ok)]) == []


def test_repository_is_clean():
    assert scrub_audit.main([]) == 0


def test_fetch_official_verifies_hashes(tmp_path):
    fake = tmp_path / "fake.whl"
    with zipfile.ZipFile(fake, "w") as z:
        z.writestr(fetch_official.ENGINE, "print('other engine')\n")
        z.writestr(fetch_official.SPEC, "{}")
    with pytest.raises(ValueError):
        fetch_official.verify_wheel(str(fake))
    real = glob.glob(os.path.join(ROOT, ".pinwork",
                                  "kaggle_environments-1.32.7-*.whl"))
    if not real:
        pytest.skip("pinned wheel not downloaded (run make official)")
    fetch_official.verify_wheel(real[0])
    assert fetch_official.main(["--wheel", real[0], "--dest",
                                str(tmp_path)]) == 0
    assert os.path.exists(tmp_path / "official" / fetch_official.ENGINE)


@pytest.mark.parametrize("script,expect", [
    ("play_match.py", "rust"),
    ("gym_loop.py", "steps 719"),
    ("tape_tools.py", "batch banks"),
    ("tournament_example.py", "games_per_sec"),
    ("selfplay_example.py", "records_written"),
])
def test_examples_run(script, expect, kagg, tmp_path):
    env = dict(os.environ, OUT=str(tmp_path), WORKERS="2")
    out = subprocess.run([sys.executable,
                          os.path.join(SCRIPTS, "examples", script)],
                         capture_output=True, text=True, env=env,
                         cwd=str(tmp_path), timeout=600)
    assert out.returncode == 0, out.stderr[-2000:]
    assert expect in out.stdout
    if script == "selfplay_example.py":
        custom = tmp_path / "example" / "custom.jsonl"
        row = json.loads(custom.read_text().splitlines()[0])
        assert "shed_market_value" in row["features"]
        assert "money_me" in row["features"]
