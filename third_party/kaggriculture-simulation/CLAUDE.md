# CLAUDE.md

Everything in [AGENTS.md](AGENTS.md) applies: layout, exact build and test
commands, the invariants that must never break, verification steps,
conventions, what never to commit, and how to port a new engine version.
Read it first.

Claude-specific notes:

* Prefer the Read / Edit / Write tools for source files. Shell heredocs and
  inline Python string literals have mangled backslash escapes (`\n`, `\t`,
  `\\`) in this repository before; write files with the Write tool.
* Keep work inside the repository. Scratch output goes to `.scratch/`
  (gitignored); never leave run outputs in tracked directories.
* The Cargo workspace is `src-rust/Cargo.toml`: pass
  `--manifest-path src-rust/Cargo.toml` (or use `make`). Build with a
  modest job count (`-j 4`) and run tests with `--test-threads=2`; the
  differential suite is CPU-heavy.
* Do not claim parity from reasoning. Run the differential suite or
  `python -m kaggsim.fidelity certify` and report the exact counts.
* Before finishing, run `make lint` (fmt check, clippy -D warnings,
  pyflakes), the tests you touched, and `python scripts/scrub_audit.py`.
