# kaggriculture-simulation -- build, test and housekeeping.
#
# Every target delegates to scripts/dev.py, which also works without make:
#     python scripts/dev.py build test
#
# Knobs (environment or `make VAR=value`):
#     CARGO_JOBS=4 TEST_THREADS=2 EPISODES=20 WORKERS=2 PYTHON=python
#     RUST_IMAGE=rust:latest PY_IMAGE=python:3.11-slim   (docker targets)
#
#   make build           release build -> src-rust/target/release/kagg
#   make install-python  pip install -e . (the kaggsim package)
#   make official        download + verify kaggle-environments==1.32.7
#   make test            build, cargo test, pytest
#   make test-rust       cargo test only
#   make test-python     pytest only (needs a built kagg)
#   make lint            cargo fmt --check, clippy -D warnings, pyflakes
#   make fmt             cargo fmt
#   make certify         Rust vs official differential (EPISODES)
#   make bench           official vs Rust benchmark (EPISODES, WORKERS)
#   make docker-linux    static Linux binary built in Docker
#   make docker-test     full test suite on Linux in Docker
#   make scrub           scrub / secret audit of tracked files
#   make clean           remove build outputs, caches, run outputs
#   make distclean       clean + .pinwork/ and .scratch/

PYTHON ?= python
DEV = $(PYTHON) scripts/dev.py
export CARGO_JOBS TEST_THREADS EPISODES WORKERS RUST_IMAGE PY_IMAGE

TARGETS = build install-python official test test-rust test-python lint fmt \
          certify bench docker-linux docker-test scrub clean distclean

.PHONY: all help $(TARGETS)

all: build

help:
	@$(DEV) --list

$(TARGETS):
	@$(DEV) $@
