"""The imported official engine must be the pinned 1.32.7 file."""
import os

import pytest

from kaggsim import official

pytestmark = pytest.mark.official


def test_engine_file_hash_is_pinned(official_mod):
    got = official.sha256_file(official_mod.__file__)
    assert got == official.PINNED_ENGINE_SHA256
    assert official.PINNED_VERSION == "1.32.7"
    spec = os.path.join(os.path.dirname(official_mod.__file__),
                        "kaggriculture.json")
    assert official.sha256_file(spec) == official.PINNED_SPEC_SHA256


def test_mismatch_is_detected(official_mod, tmp_path):
    fake = tmp_path / "kaggriculture.py"
    fake.write_text("# some other engine\n")

    class Mod:
        __file__ = str(fake)

    with pytest.raises(official.EngineMismatch):
        official.verify(Mod)
