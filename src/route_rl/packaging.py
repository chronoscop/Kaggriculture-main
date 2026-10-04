"""Build reviewable source releases and candidate policy bundles from this project.

No reference checkout, training run, credentials or compiled artifact is included.
Archives have deterministic metadata and a checked SHA256 file inventory.
"""
from __future__ import annotations

import argparse
import gzip
import hashlib
import io
import json
import os
from pathlib import Path, PurePosixPath
import pickle
import tarfile

SOURCE_CONTRACT = "route-rl-source-release-v1"
SUBMISSION_CONTRACT = "route-rl-policy-bundle-v1"
PPO_ACTION_CONTRACT = "official-prefix-full-action-ppo-v1"
PPO_OBJECTIVE_CONTRACT = "terminal-match-score-1-0.5-0-v1"
MANIFEST_NAME = "bundle-manifest.json"
SOURCE_DIRECTORIES = ("src", "tools", "docs", "tests", "native", "agents/farm2945_resilient_response",
                      "third_party/kaggriculture-simulation")
SOURCE_SUFFIXES = frozenset((".py", ".json", ".md", ".svg", ".rs", ".toml", ".lock", ".cpp", ".inc", ".c", ".h", ".sh", ".txt"))
SOURCE_NAMES = frozenset(("LICENSE", "NOTICE", "Makefile", ".gitignore"))
EXCLUDED_DIRECTORIES = frozenset((".git", ".aws", ".codex", ".agents", "__pycache__", "target", "build", "dist",
                                  "runs", "data", "replay", "models", "venv", ".venv", "node_modules",
                                  "kaggriculture-solution", ".pytest_cache", ".mypy_cache", "secrets"))

ENTRYPOINT = '''"""Kaggle entry point for an explicitly packaged project candidate."""
import os
from pathlib import Path
import sys

os.environ.setdefault("JAX_PLATFORMS", "cpu")
os.environ.setdefault("CUDA_VISIBLE_DEVICES", "")
os.environ.setdefault("XLA_PYTHON_CLIENT_PREALLOCATE", "false")
_policy = None


def bundle_root(configuration=None):
    entry = globals().get("__file__")
    if configuration is not None:
        raw = (configuration.get("__raw_path__") if hasattr(configuration, "get")
               else getattr(configuration, "__raw_path__", None))
        if raw and Path(raw).is_file():
            entry = raw
    candidates = [Path(entry).resolve().parent] if entry else []
    candidates.append(Path("/kaggle_simulations/agent"))
    for root in candidates:
        if (root / "policy.pkl").is_file() and (root / "bundle-manifest.json").is_file():
            return root
    raise FileNotFoundError("policy.pkl must be beside the packaged main.py")


def agent(observation, configuration=None):
    global _policy
    if _policy is None:
        root = bundle_root(configuration)
        sys.path.insert(0, str(root))
        from __POLICY_MODULE__ import load_policy
        _policy = load_policy(root / "policy.pkl", warm=True)
    return _policy(observation)
'''


def _json(value: object) -> bytes:
    return (json.dumps(value, indent=2, sort_keys=True) + "\n").encode()


def _sha256(value: bytes) -> str:
    return hashlib.sha256(value).hexdigest()


def _safe_name(name: str) -> bool:
    path = PurePosixPath(name)
    return bool(name) and not path.is_absolute() and ".." not in path.parts and "\\" not in name


def _allowed(path: Path) -> bool:
    if any(part in EXCLUDED_DIRECTORIES or part.startswith(".venv") or part.endswith(".egg-info")
           for part in path.parts):
        return False
    name = path.name.lower()
    if name.startswith(".env") or name in {"kaggle.json", "credentials.json", "credentials", "token.json"}:
        return False
    if any(word in name for word in ("secret", "private_key", "credential")):
        return False
    return path.suffix in SOURCE_SUFFIXES or path.name in SOURCE_NAMES or path.name.startswith("LICENSE")


def _collect(root: Path, directories: tuple[str, ...]) -> dict[str, bytes]:
    files = {}
    for directory in directories:
        base = root / directory
        if not base.exists():
            continue
        if base.is_symlink():
            raise ValueError(f"release directory cannot be a symlink: {directory}")
        for current, names, filenames in os.walk(base, followlinks=False):
            current = Path(current)
            names[:] = sorted(name for name in names if name not in EXCLUDED_DIRECTORIES
                              and not name.startswith(".") and not name.endswith(".egg-info"))
            for name in names:
                if (current / name).is_symlink():
                    raise ValueError(f"release directory cannot contain symlinks: {current / name}")
            for name in sorted(filenames):
                path = current / name
                relative = path.relative_to(root)
                if not _allowed(relative):
                    continue
                if path.is_symlink():
                    raise ValueError(f"release file cannot be a symlink: {relative}")
                files[relative.as_posix()] = path.read_bytes()
    return files


def _write_archive(output: Path, files: dict[str, bytes], metadata: dict) -> dict:
    output = Path(output)
    if output.suffixes[-2:] != [".tar", ".gz"]:
        raise ValueError("output must end in .tar.gz")
    if MANIFEST_NAME in files:
        raise ValueError("manifest name is reserved")
    if any(not _safe_name(name) for name in files):
        raise ValueError("archive contains an unsafe file path")
    manifest = {**metadata, "files": {name: _sha256(data) for name, data in sorted(files.items())}}
    all_files = {**files, MANIFEST_NAME: _json(manifest)}
    output.parent.mkdir(parents=True, exist_ok=True)
    # Exclusive creation preserves existing releases and policy artifacts.
    try:
        with output.open("xb") as destination:
            with gzip.GzipFile(fileobj=destination, mode="wb", filename="", mtime=0) as compressed:
                with tarfile.open(fileobj=compressed, mode="w", format=tarfile.PAX_FORMAT) as archive:
                    for name, data in sorted(all_files.items()):
                        member = tarfile.TarInfo(name)
                        member.size = len(data)
                        member.mode = 0o755 if name.endswith(".sh") else 0o644
                        member.mtime = member.uid = member.gid = 0
                        member.uname = member.gname = ""
                        archive.addfile(member, io.BytesIO(data))
    except FileExistsError:
        raise FileExistsError(f"refusing to overwrite existing archive: {output}") from None
    except BaseException:
        output.unlink(missing_ok=True)
        raise
    verify_archive(output)
    return {"archive": str(output), "bytes": output.stat().st_size,
            "archive_sha256": _sha256(output.read_bytes()), "file_count": len(files), **manifest}


def source_release(root: Path, output: Path) -> dict:
    """Snapshot the current owned pipelines and their preserved source dependencies."""
    root = Path(root).resolve()
    for name in ("pyproject.toml", "README.md", "AGENTS.md", "src/route_rl/__init__.py"):
        if not (root / name).is_file():
            raise FileNotFoundError(f"required source release file missing: {name}")
    files = _collect(root, SOURCE_DIRECTORIES)
    for path in sorted(root.iterdir()):
        if path.is_file() and (path.name in {"README.md", "AGENTS.md", "pyproject.toml", "setup.py", ".gitignore"}
                               or path.name.startswith(("LICENSE", "NOTICE", "THIRD_PARTY"))):
            if path.is_symlink():
                raise ValueError(f"release file cannot be a symlink: {path.name}")
            files[path.name] = path.read_bytes()
    return _write_archive(output, files, {
        "contract": SOURCE_CONTRACT, "kind": "source", "reference_checkout_required": False,
        "contains_training_data": False, "contains_policy_weights": False,
        "deployment": "source_only", "entrypoints": ["route_rl.action_bc", "route_rl.action_ppo", "route_rl.action_teacher"],
    })


def _policy_module(payload: dict) -> str:
    marker = payload.get("action_selection_contract")
    if marker is None:
        return "route_rl.full_action.inference"
    if marker in (PPO_ACTION_CONTRACT, 'official-prefix-economic-full-action-ppo-v2'):
        if payload.get("learning_objective_contract") != PPO_OBJECTIVE_CONTRACT:
            raise ValueError("PPO policy requires the current terminal match-score learning objective contract")
        return "route_rl.ppo.inference"
    raise ValueError(f"unsupported policy action-selection contract: {marker!r}")


def package_submission(policy: Path, output: Path, *, root: Path | None = None) -> dict:
    """Package an existing BC/PPO candidate without modifying or promoting it.

    The checkpoint is a trusted local pickle. Runtime dependencies are supplied
    by the target environment, recorded in requirements.txt and the manifest.
    """
    from route_rl.full_action.checkpoints import load_training_source
    from route_rl.full_action.model import JaxModelConfig

    root = Path(root).resolve() if root else Path(__file__).resolve().parents[2]
    policy = Path(policy)
    payload = load_training_source(policy)
    JaxModelConfig(**payload["model_config"]).validate()
    module = _policy_module(payload)
    module_file = root / "src" / Path(*module.split(".")).with_suffix(".py")
    if not module_file.is_file():
        raise FileNotFoundError(f"required policy execution module missing: {module}")
    owned = _collect(root, ("src/route_rl",))
    files = {name.removeprefix("src/"): data for name, data in owned.items()}
    fields = ("contract", "params", "model_config", "policy_sha256", "action_selection_contract",
              "learning_objective_contract", "objective_contract", "continuation_policy_version",
              "rl_completed_updates", "training_backend", "compute_dtype", "execution_contract", "ppo_contract",
              "season_controller", "controller_identity", "candidate_origin", "candidate_source",
              "base_policy_file_sha256", "base_policy_sha256", "base_action_selection_contract")
    exported = {name: payload[name] for name in fields if name in payload}
    files["policy.pkl"] = pickle.dumps(exported, protocol=pickle.HIGHEST_PROTOCOL)
    files["main.py"] = ENTRYPOINT.replace("__POLICY_MODULE__", module).encode()
    if payload.get('season_controller') is not None:
        from .ppo.controllers import contract_identity
        if contract_identity(payload['season_controller']) != payload.get('controller_identity'):
            raise ValueError('season controller source/binary/config changed; create a new candidate before packaging')
        asset = root / 'src/route_rl/season_search/terminal_search.so'
        if not asset.is_file() or asset.is_symlink():
            raise FileNotFoundError('owned precompiled season-search library is required in the candidate bundle')
        if _sha256(asset.read_bytes()) != payload['controller_identity']['binary_sha256']:
            raise ValueError('candidate bundle season-search binary mismatch')
        files['route_rl/season_search/terminal_search.so'] = asset.read_bytes()
    requirements = ["numpy==2.5.3", "jax==0.11.1"]
    if module.startswith("route_rl.ppo."):
        requirements.append("kaggle-environments==1.32.7")
    files["requirements.txt"] = ("\n".join(requirements) + "\n").encode()
    for name in ("docs/action_bc_sources.md", "docs/action_bc_sources.json", "docs/action_ppo_sources.md",
                 "docs/action_ppo_sources.json"):
        path = root / name
        if path.is_file():
            files[name] = path.read_bytes()
    for path in sorted(root.iterdir()):
        if path.is_file() and path.name.startswith(("LICENSE", "NOTICE", "THIRD_PARTY")):
            if path.is_symlink():
                raise ValueError(f"bundle attribution file cannot be a symlink: {path.name}")
            files[path.name] = path.read_bytes()
    return _write_archive(output, files, {
        "contract": SUBMISSION_CONTRACT, "kind": "submission", "deployment": "candidate_only",
        "reference_checkout_required": False, "policy_sha256": payload["policy_sha256"],
        "source_policy_file_sha256": _sha256(policy.read_bytes()), "policy_execution_module": module,
        "action_selection_contract": payload.get("action_selection_contract", "legacy-bc-greedy-v1"),
        "runtime_dependencies": requirements, "dependencies_vendored": False,
        "target_environment_validation": "not_performed_by_packaging",
        "controller_identity": payload.get('controller_identity'),
    })


def verify_archive(path: Path) -> dict:
    """Check exact inventory, regular files, safe paths and all content hashes."""
    with tarfile.open(path, "r:gz") as archive:
        members = archive.getmembers()
        names = [member.name for member in members]
        if len(names) != len(set(names)):
            raise ValueError("archive contains duplicate paths")
        if any(not member.isfile() or not _safe_name(member.name) for member in members):
            raise ValueError("archive contains unsafe paths or non-regular files")
        if MANIFEST_NAME not in names:
            raise ValueError("archive manifest missing")
        manifest = json.load(archive.extractfile(MANIFEST_NAME))
        expected = manifest.get("files", {})
        if set(names) != {*expected, MANIFEST_NAME}:
            raise ValueError("archive contents do not match the manifest")
        if manifest.get("contract") not in {SOURCE_CONTRACT, SUBMISSION_CONTRACT}:
            raise ValueError("unsupported archive contract")
        for name, checksum in expected.items():
            if _sha256(archive.extractfile(name).read()) != checksum:
                raise ValueError(f"archive checksum mismatch: {name}")
    return manifest


def main(argv: list[str] | None = None) -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    subparsers = parser.add_subparsers(dest="command", required=True)
    source = subparsers.add_parser("source", help="package this project's source without runs, data or models")
    source.add_argument("--out", type=Path, required=True)
    submission = subparsers.add_parser("submission", help="package an existing candidate policy for review")
    submission.add_argument("--policy", type=Path, required=True)
    submission.add_argument("--out", type=Path, required=True)
    verify = subparsers.add_parser("verify", help="verify an archive's exact inventory and checksums")
    verify.add_argument("archive", type=Path)
    args = parser.parse_args(argv)
    if args.command == "source":
        result = source_release(Path(__file__).resolve().parents[2], args.out)
    elif args.command == "submission":
        result = package_submission(args.policy, args.out)
    else:
        manifest = verify_archive(args.archive)
        result = {"archive": str(args.archive), "contract": manifest["contract"], "file_count": len(manifest["files"]),
                  "verified": True}
    # Detailed inventory belongs inside the artifact, keeping CLI output concise.
    print(json.dumps({name: value for name, value in result.items() if name != "files"}, sort_keys=True))


if __name__ == "__main__":
    main()
