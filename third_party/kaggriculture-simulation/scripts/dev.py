#!/usr/bin/env python3
"""Cross-platform task runner (the Makefile delegates here).

    python scripts/dev.py <task> [<task> ...]
    python scripts/dev.py --list

Environment:
  CARGO_JOBS      parallel cargo jobs (default 4)
  TEST_THREADS    Rust test threads (default 2)
  EPISODES        differential / certify episodes (default 20)
  WORKERS         benchmark workers (default 2)
  PYTHON          interpreter for Python tasks (default: this one)
"""
from __future__ import annotations

import os
import shutil
import subprocess
import sys

ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
PY = os.environ.get("PYTHON", sys.executable)
JOBS = os.environ.get("CARGO_JOBS", "4")
THREADS = os.environ.get("TEST_THREADS", "2")
EPISODES = os.environ.get("EPISODES", "20")
WORKERS = os.environ.get("WORKERS", "2")


def sh(*cmd, env=None):
    print("+", " ".join(cmd), flush=True)
    full = dict(os.environ)
    full["PYTHONPATH"] = os.pathsep.join(
        [os.path.join(ROOT, "src-python")]
        + [p for p in [full.get("PYTHONPATH")] if p])
    full.update(env or {})
    r = subprocess.run(cmd, cwd=ROOT, env=full)
    if r.returncode:
        raise SystemExit(r.returncode)


MANIFEST = ("--manifest-path", os.path.join("src-rust", "Cargo.toml"))


def cargo(sub, *args):
    """Run a cargo subcommand on the src-rust workspace."""
    sh("cargo", sub, *MANIFEST, *args)


TASKS = {}


def task(fn):
    TASKS[fn.__name__.replace("_", "-")] = fn
    return fn


@task
def build():
    """Release build of the kagg binary (src-rust/target/release/kagg)."""
    cargo("build", "--release", "-j", JOBS)


@task
def install_python():
    """Editable install of the kaggsim package."""
    sh(PY, "-m", "pip", "install", "-e", ".")


@task
def fmt():
    """Format Rust sources."""
    cargo("fmt", "--all")


@task
def lint():
    """cargo fmt --check, clippy -D warnings, pyflakes (if installed)."""
    cargo("fmt", "--all", "--check")
    cargo("clippy", "-j", JOBS, "--all-targets", "--", "-D", "warnings")
    import importlib.util
    if importlib.util.find_spec("pyflakes") is None:
        print("pyflakes not installed; skipping Python lint")
    else:
        sh(PY, "-m", "pyflakes", "src-python", "tests", "scripts")


@task
def test_rust():
    """cargo test (unit + CLI integration)."""
    cargo("test", "-j", JOBS, "--", f"--test-threads={THREADS}")


@task
def test_python():
    """pytest: unit, regressions, integration, differential."""
    sh(PY, "-m", "pytest", "-q", env={"KAGGSIM_DIFF_EPISODES": EPISODES})


@task
def test():
    """build + test-rust + test-python."""
    build()
    test_rust()
    test_python()


@task
def certify():
    """Differential certification vs the official engine (EPISODES)."""
    sh(PY, "-m", "kaggsim.fidelity", "certify", "--episodes", EPISODES)


@task
def bench():
    """Official vs Rust benchmark (EPISODES games, WORKERS workers)."""
    sh(PY, "-m", "kaggsim.benchmark", "--games", EPISODES, "--workers",
       WORKERS)


RUST_IMAGE = os.environ.get("RUST_IMAGE", "rust:latest")
PY_IMAGE = os.environ.get("PY_IMAGE", "python:3.11-slim")
LINUX_TARGET = "x86_64-unknown-linux-musl"
LINUX_BIN = f"target/linux/{LINUX_TARGET}/release/kagg"


def _docker(image, script, env=()):
    """Run ``script`` (bash) in ``image`` with the repo mounted at /w."""
    mount = ROOT.replace("\\", "/")
    cmd = ["docker", "run", "--rm", "-v", f"{mount}:/w", "-w", "/w"]
    for e in env:
        cmd += ["-e", e]
    cmd += [image, "bash", "-c", script]
    print("+", " ".join(cmd[:-1]), repr(script), flush=True)
    r = subprocess.run(cmd, cwd=ROOT,
                       env=dict(os.environ, MSYS_NO_PATHCONV="1"))
    if r.returncode:
        raise SystemExit(r.returncode)


@task
def docker_linux():
    """Static Linux binary in Docker -> target/linux/.../release/kagg."""
    _docker(RUST_IMAGE,
            f"rustup target add {LINUX_TARGET} >/dev/null && "
            f"cargo build --manifest-path src-rust/Cargo.toml --release "
            f"--target {LINUX_TARGET} -j {JOBS} && "
            f"{LINUX_BIN} version",
            env=["CARGO_TARGET_DIR=/w/target/linux"])


@task
def docker_test():
    """Full pytest suite on Linux in Docker (needs docker-linux first)."""
    _docker(PY_IMAGE,
            "pip install -q --root-user-action=ignore --no-deps "
            "kaggle-environments==1.32.7 && "
            "pip install -q --root-user-action=ignore jsonschema requests "
            "pytest && python -m pytest -q -p no:cacheprovider "
            "-k 'not repository_is_clean'",
            env=[f"KAGG_BIN=/w/{LINUX_BIN}", "PYTHONPATH=/w/src-python",
                 "PYTHONDONTWRITEBYTECODE=1", "KAGGSIM_REQUIRE_OFFICIAL=1",
                 f"KAGGSIM_DIFF_EPISODES={EPISODES}"])


@task
def official():
    """Download + verify the pinned official engine into .pinwork/."""
    sh(PY, "scripts/fetch_official.py")


@task
def scrub():
    """Scrub / secret audit of tracked files."""
    sh(PY, "scripts/scrub_audit.py")


def _rm(path):
    p = os.path.join(ROOT, path)
    if os.path.isdir(p):
        shutil.rmtree(p, ignore_errors=True)
        print("removed", path)
    elif os.path.exists(p):
        os.remove(p)
        print("removed", path)


@task
def clean():
    """Remove build outputs and caches (keeps .pinwork)."""
    _rm(os.path.join("src-rust", "target"))
    _rm("target")
    for dirpath, dirnames, _ in os.walk(ROOT):
        rel = os.path.relpath(dirpath, ROOT).replace(os.sep, "/")
        if rel.startswith((".git", "target", "src-rust/target", ".pinwork")):
            continue
        for d in list(dirnames):
            if d in ("__pycache__", ".pytest_cache") or d.endswith(
                    ".egg-info"):
                _rm(os.path.relpath(os.path.join(dirpath, d), ROOT))
    for d in ("build", "dist", "tournaments", "selfplay", "gauntlets"):
        _rm(d)


@task
def distclean():
    """clean + downloaded engine (.pinwork) and scratch (.scratch)."""
    clean()
    _rm(".pinwork")
    _rm(".scratch")


def main(argv):
    if not argv or argv[0] in ("-h", "--help", "--list", "help"):
        print(__doc__)
        for name, fn in TASKS.items():
            print(f"  {name:<15} {fn.__doc__}")
        return 0
    for name in argv:
        if name not in TASKS:
            print(f"unknown task {name!r}; see --list")
            return 2
    for name in argv:
        TASKS[name]()
    return 0


if __name__ == "__main__":
    raise SystemExit(main(sys.argv[1:]))
