"""Shared fixtures.

* The package is imported from ``python/`` (no install needed).
* ``kagg`` is found by ``kaggsim.binary.find_kagg`` (``$KAGG_BIN``,
  ``$CARGO_TARGET_DIR/release``, ``src-rust/target/release``); build it
  first with ``make build``.
* Tests marked ``official`` need the pinned official engine. They are
  skipped when it is not importable, unless ``KAGGSIM_REQUIRE_OFFICIAL=1``
  (CI), in which case a missing or wrong engine FAILS the run. Point at an
  unpacked copy with ``KAGGSIM_OFFICIAL_PATH=<dir containing
  kaggle_environments>``.
"""
import os
import sys

import pytest

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.join(ROOT, "src-python"))

from kaggsim import official  # noqa: E402
from kaggsim.binary import find_kagg  # noqa: E402


@pytest.fixture(scope="session")
def kagg():
    try:
        return find_kagg()
    except FileNotFoundError as exc:
        pytest.fail(str(exc))


@pytest.fixture(scope="session")
def official_mod():
    try:
        return official.engine_module()
    except (ImportError, official.EngineMismatch) as exc:
        if os.environ.get("KAGGSIM_REQUIRE_OFFICIAL") == "1":
            pytest.fail(f"pinned official engine required: {exc}")
        pytest.skip(f"pinned official engine not available: {exc}")


@pytest.fixture
def serve(kagg):
    from kaggsim.serve import Serve
    srv = Serve(kagg)
    yield srv
    srv.close()
