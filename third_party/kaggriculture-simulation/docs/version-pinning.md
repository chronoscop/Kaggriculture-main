# Version pinning

The Rust engine reproduces `kaggle-environments==1.32.7`.

| Release | `envs/kaggriculture/kaggriculture.py` sha256 |
|---|---|
| **1.32.7 (pinned)** | `bc8a54879ef02c7ea64b8b333d6a976f0ea65c4949149d01f463f23bccee653e` |
| 1.32.6 | `fb9215c5e21a25243e2d13e75b3d70a79cf7d78fff150a90f1bb5eacf9ba2bcf` |

The 1.32.7 environment spec `kaggriculture.json` has sha256
`a82c89c1a2315b93f39775d8e025471a01b738647c9772658368ee6b1b6f4867`.

The main behavioural change in 1.32.7 is the **hinge** price curve on the
scarcity side of CARROT (target 1.00), TOMATO and EGG: linear in `u = x/T`
up to the capacity `T`, then `u + 8 * max(0, u - 1)^2`. Older releases used
log/linear curves there, so a stale engine quotes different prices without
raising any error.

## Guard

`kaggsim.official.engine_module()` imports `kaggle_environments`, hashes the
imported `kaggriculture.py`, its `kaggriculture.json` spec and the runner
files that decide how agents are called (`core.py`, `agent.py`,
`utils.py`), and raises `EngineMismatch` unless all equal the pins. It also
refuses an engine whose RNG has been patched by `kaggsim.forced`. Every fidelity tool goes through it, and
`tests/regressions/test_engine_pin.py` asserts it.

Watch out for a stale copy in a user site-packages directory, which can
shadow the one you think you installed. Either install the pinned release in
a clean virtual environment, or set `KAGGSIM_OFFICIAL_PATH` to a directory
that contains the right `kaggle_environments` package (it is put first on
`sys.path` before the import).

## Installing the pinned engine cheaply

The full dependency list of `kaggle-environments` is heavy (jax, open_spiel,
transformers, ...). Importing it and running the kaggriculture environment
needs only two of those dependencies:

```
pip install --no-deps kaggle-environments==1.32.7
pip install jsonschema requests
```

Other environments in the package then fail to register. That is harmless,
and `kaggsim.official` suppresses the message.

## When the competition moves to a new release

1. `pip download --no-deps kaggle-environments==X -d tmp/` and hash
   `kaggle_environments/envs/kaggriculture/kaggriculture.py` inside the
   wheel.
2. If it changed, port the diff to `src-rust/kagg-engine`, then update
   `ENGINE_VERSION` (`src-rust/kagg-engine/src/market.rs`), the `PINNED_*`
   constants (`src-python/kaggsim/official.py`), this page and the CI pin,
   and run the differential suite at nightly size.

`scripts/fetch_official.py` (`make official`) downloads the pinned wheel,
verifies both hashes and extracts it under `.pinwork/official`, printing
the `KAGGSIM_OFFICIAL_PATH` to use.
