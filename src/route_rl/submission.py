"""Export a trained checkpoint to a small, self-contained Linux submission."""
from __future__ import annotations
import argparse
from array import array
import gzip
import hashlib
import io
import json
import os
from pathlib import Path
import platform
import shutil
import subprocess
import sys
import tarfile
import tempfile

from .features import FEATURE_SIZE, STATE_SIZE, SCHEMA
from .deployment import NativePolicy

MAX_ARCHIVE = 100 * 1024 ** 2
MAX_DISK = 8 * 1024 ** 3
LAYOUT = (
    ("encoder.0.weight", (128, FEATURE_SIZE)), ("encoder.0.bias", (128,)),
    ("context.0.weight", (128, STATE_SIZE)), ("context.0.bias", (128,)),
    ("context.2.weight", (128, 128)), ("context.2.bias", (128,)),
    ("actor.0.weight", (128, 256)), ("actor.0.bias", (128,)),
    ("actor.2.weight", (1, 128)), ("actor.2.bias", (1,)),
)
ENTRYPOINT = '''"""Kaggriculture dynamic-route policy: agent(observation, configuration)."""
import sys
from pathlib import Path
_root = Path(globals().get("__file__", "/kaggle_simulations/agent/main.py")).resolve().parent
if str(_root) not in sys.path:
    sys.path.insert(0, str(_root))
from _route_submission.deployment import SubmissionAgent
_submission = SubmissionAgent()

def agent(observation, configuration=None):
    return _submission.act(observation, configuration)
'''

def export(checkpoint_path, output, planning_seconds=0.65, compiler="cc"):
    import torch
    torch.set_num_threads(1)
    if sys.platform != "linux" or platform.machine() != "x86_64" or sys.byteorder != "little":
        raise RuntimeError("build submission on Linux x86_64 little-endian")
    if not 0 <= planning_seconds <= 0.8:
        raise ValueError("planning_seconds must be 0..0.8 (0 disables the budget for parity tests)")
    cc = shutil.which(compiler)
    if cc is None:
        raise RuntimeError("a C compiler is needed at export time, not during submission inference")
    checkpoint_path, output = Path(checkpoint_path), Path(output)
    checkpoint = torch.load(checkpoint_path, map_location="cpu", weights_only=False)
    if checkpoint.get("schema") != SCHEMA:
        raise ValueError("export requires a dynamic-routes-v2 checkpoint")
    if checkpoint.get("engine") != "kaggle-environments==1.32.7":
        raise ValueError("checkpoint engine does not match")
    if checkpoint.get("takeover") != 0:
        raise ValueError("submission export requires takeover=0; warm-up baseline is not bundled")
    horizon, interval = checkpoint["horizon"], checkpoint["replan_interval"]
    if not 1 <= horizon <= 24 or interval < 1:
        raise ValueError("invalid checkpoint planning parameters")
    weights = array("f")
    network = checkpoint["network"]
    for key, shape in LAYOUT:
        value = network[key].detach().cpu().float()
        if tuple(value.shape) != shape or not torch.isfinite(value).all():
            raise ValueError(f"invalid policy parameter: {key}")
        weights.extend(value.flatten().tolist())
    raw = weights.tobytes()
    metadata = dict(schema=SCHEMA, engine=checkpoint["engine"], horizon=horizon,
        replan_interval=interval, planning_seconds=planning_seconds,
        weights_sha256=hashlib.sha256(raw).hexdigest(),
        checkpoint_sha256=hashlib.sha256(checkpoint_path.read_bytes()).hexdigest(),
        runtime="stdlib + bundled single-thread C, Linux x86_64", dtype="float32",
        parameters=len(weights), iteration=checkpoint.get("iteration"),
        mean_margin=checkpoint.get("mean_margin"))
    source = Path(__file__).resolve().parent
    output.parent.mkdir(parents=True, exist_ok=True)
    with tempfile.TemporaryDirectory(prefix="route-export-") as temporary:
        root = Path(temporary)
        package = root / "_route_submission"
        package.mkdir()
        (root / "main.py").write_text(ENTRYPOINT)
        (root / "NOTICE.txt").write_text(
            "Dynamic-route RL inference artifact.\n"
            "Contains the learned policy and this repository's route executor.\n"
            "Does not contain the reference submission, training optimizer,\n"
            "baseline opponent, simulator, training trajectories, or training seeds.\n"
            "Planning may stop early at the configured per-action time budget.\n")
        (package / "__init__.py").write_text("")
        for name in ("routes.py", "controller.py", "features.py", "deployment.py"):
            shutil.copyfile(source / name, package / name)
        (package / "weights.bin").write_bytes(raw)
        (package / "metadata.json").write_text(json.dumps(metadata, indent=2) + "\n")
        subprocess.run([cc, "-O3", "-std=c99", "-shared", "-fPIC",
            "-march=x86-64", "-mtune=generic", "-ffp-contract=off",
            str(source / "inference.c"), "-o", str(package / "inference.so"), "-lm"],
            check=True, capture_output=True, text=True)
        # Verify exported policy logits against PyTorch before publishing bytes.
        native = NativePolicy(package)
        from .training import Network
        model = Network(torch)
        model.module.load_state_dict(network)
        model.module.eval()
        generator = torch.Generator().manual_seed(67281)
        max_error = 0.0
        for count in (1, 7, 257):
            features = torch.rand((1, count, FEATURE_SIZE), generator=generator) * 2 - 1
            state = torch.rand((1, STATE_SIZE), generator=generator) * 2 - 1
            with torch.no_grad():
                expected, _ = model(features, torch.ones((1, count), dtype=torch.bool), state)
            actual = torch.tensor(native.scores(features[0].tolist(), state[0].tolist()))
            error = float((expected[0] - actual).abs().max())
            max_error = max(max_error, error)
            if not torch.allclose(expected[0], actual, atol=2e-5, rtol=2e-5):
                raise ValueError(f"native/PyTorch logit mismatch: {error}")
            if int(actual.argmax()) != int(expected[0].argmax()):
                raise ValueError("native/PyTorch greedy choice mismatch")
        files = sorted(p for p in root.rglob("*") if p.is_file())
        unpacked = sum(p.stat().st_size for p in files)
        if unpacked > MAX_DISK:
            raise ValueError("unpacked artifact exceeds 8 GiB")
        archive = root / "submission.tar.gz"
        with archive.open("wb") as fh:
            with gzip.GzipFile(filename="", mode="wb", fileobj=fh, mtime=0) as gz:
                with tarfile.open(fileobj=gz, mode="w") as tar:
                    for path in files:
                        data = path.read_bytes()
                        info = tarfile.TarInfo(path.relative_to(root).as_posix())
                        info.size, info.mode, info.mtime = len(data), 0o644, 0
                        tar.addfile(info, io.BytesIO(data))
        compressed = archive.stat().st_size
        if compressed > MAX_ARCHIVE:
            raise ValueError("submission exceeds 100 MiB")
        # Replace only once all validations succeed.
        with tempfile.NamedTemporaryFile(dir=output.parent, delete=False) as fh:
            staged = Path(fh.name)
            fh.write(archive.read_bytes())
        try:
            os.replace(staged, output)
        finally:
            staged.unlink(missing_ok=True)
    return dict(archive=str(output), compressed_bytes=compressed, unpacked_bytes=unpacked,
                policy_parameters=len(weights), max_logit_error=max_error,
                planning_seconds=planning_seconds, sha256=hashlib.sha256(output.read_bytes()).hexdigest())

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--planning-seconds", type=float, default=0.65)
    parser.add_argument("--compiler", default="cc")
    args = parser.parse_args()
    print(json.dumps(export(args.checkpoint, args.out, args.planning_seconds, args.compiler),
                     ensure_ascii=False, indent=2))

if __name__ == "__main__":
    main()
