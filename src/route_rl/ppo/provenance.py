"""Versioned source, run, and seed identities outside the unchanged BC branch."""
from __future__ import annotations

import hashlib
import json
import os
import re
from pathlib import Path

PIPELINE_CONTRACT = 'route-rl-full-action-ppo-v2'
OBJECTIVE_CONTRACT = 'terminal-match-score-1-0.5-0-v1'
PACKAGE_ROOT = Path(__file__).resolve().parent


def file_hash(path: Path) -> str:
    digest = hashlib.sha256()
    with Path(path).open('rb') as stream:
        while block := stream.read(1024 * 1024):
            digest.update(block)
    return digest.hexdigest()


def write_json(path: Path, value: dict) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f'.{path.name}.{os.getpid()}.tmp')
    with temporary.open('w') as stream:
        json.dump(value, stream, indent=2, sort_keys=True)
        stream.write('\n')
        stream.flush()
        os.fsync(stream.fileno())
    temporary.replace(path)


def source_identity() -> dict:
    from ..action_bc import source_identity as bc_source

    paths = list(PACKAGE_ROOT.rglob('*.py')) + list((PACKAGE_ROOT / 'configs').glob('*.json'))
    paths.append(PACKAGE_ROOT.parent / 'action_ppo.py')
    teacher = PACKAGE_ROOT.parent / 'action_teacher.py'
    if teacher.exists():
        paths.append(teacher)
    search = PACKAGE_ROOT.parent / 'season_search'
    if search.exists():
        paths += [p for p in search.rglob('*') if p.is_file() and p.suffix in ('.py', '.cpp', '.inc', '.json')]
    hashes = {str(p.relative_to(PACKAGE_ROOT.parent)): file_hash(p) for p in sorted(paths)}
    hashes['bc_dependencies'] = bc_source()['files_sha256']
    return {'implementation': 'route_rl.ppo', 'reference_checkout_required': False,
            'files_sha256': hashlib.sha256(json.dumps(hashes, sort_keys=True).encode()).hexdigest()}


def bc_provenance(run: Path) -> dict:
    from ..action_bc import CONTRACT, source_identity as bc_source

    identity = json.loads((run / 'integration.json').read_text())
    if identity['contract'] != CONTRACT or identity['source'] != bc_source():
        raise ValueError('BC source/contract changed; restore matching BC code before using its lineage')
    inventory = identity['preparation']['input']
    return {'integration_sha256': file_hash(run / 'integration.json'),
            'source': identity['source'], 'initial_sha256': identity['initial_sha256'],
            'known_game_seeds': sorted(set(inventory['known_game_seeds'])),
            'games_without_seed': inventory['games_without_seed'],
            'cache': identity['cache']}


def verify_bc_policy(run: Path, initial: Path) -> dict:
    """Bind a selected policy or byte-identical copy to the named BC run.

    Existing BC inference exports have no embedded run identifier. Their own
    final/epoch files are the trusted membership evidence; arbitrary policies
    cannot borrow another run's demonstration seed exclusions. Receipt hashes
    validate the final artifact when that artifact is selected. Only stable
    identities are returned, so later BC epochs cannot invalidate an older,
    frozen policy's PPO continuation.
    """
    run, initial = Path(run), Path(initial)
    provenance = bc_provenance(run)
    selected_hash = file_hash(initial)
    matches = []
    for candidate in sorted(run.iterdir()):
        supported = (candidate.name == 'final_student_jax.pkl'
                     or re.fullmatch(r'epoch-[0-9]+-policy\.pkl', candidate.name) is not None)
        if supported and candidate.is_file() and not candidate.is_symlink():
            if file_hash(candidate) == selected_hash:
                matches.append(candidate.name)
    if not matches:
        raise ValueError('BC policy does not match a final/epoch artifact from the specified BC run')

    receipt_path = run / 'receipt.json'
    if receipt_path.is_file():
        receipt = json.loads(receipt_path.read_text())
        expected = {'initial_sha256': provenance['initial_sha256'],
                    'cache_sha256': provenance['cache'].get('index_sha256')}
        if any(value is None or receipt.get(key) != value for key, value in expected.items()):
            raise ValueError('BC final receipt initial/cache lineage mismatch')
        if 'final_student_jax.pkl' in matches:
            from ..full_action.checkpoints import load_training_source

            payload = load_training_source(initial)
            if receipt.get('policy_sha256') != payload['policy_sha256']:
                raise ValueError('BC final receipt policy checksum mismatch')
    return {'membership_contract': 'owned-bc-run-policy-membership-v1',
            'policy_file_sha256': selected_hash,
            'bc_integration_sha256': provenance['integration_sha256']}


def establish_run(output: Path, identity: dict, settings: dict, target_updates: int) -> None:
    if target_updates < 1:
        raise ValueError('updates is a positive cumulative target')
    receipt = output / 'integration.json'
    if output.exists() and any(output.iterdir()):
        if not receipt.exists() or json.loads(receipt.read_text()) != identity:
            raise ValueError('PPO run source/input/settings mismatch; use a new output directory')
        old = output / 'run_config.json'
        if old.exists() and target_updates < json.loads(old.read_text())['target_updates']:
            raise ValueError('updates cannot decrease on resume')
    write_json(receipt, identity)
    write_json(output / 'run_config.json', {**settings, 'target_updates': target_updates})



def require_training_state(output: Path, state_name: str, phase: str) -> None:
    """An existing run cannot silently reset after losing its optimizer state."""
    if (output / state_name).exists():
        return
    evidence = [(output / 'receipt.json'), (output / 'metrics.jsonl')]
    evidence += list(output.glob('policy-*.pkl'))
    evidence += [output / ('policy_with_critic.pkl' if phase == 'critic' else 'policy_latest_jax.pkl')]
    if any(path.is_file() and path.stat().st_size for path in evidence):
        raise ValueError(f'existing {phase} artifacts require {state_name}; restore the state or use a new run')


def check_seed_ranges(ranges: dict[str, tuple[int, int]]) -> None:
    items = list(ranges.items())
    for name, (start, stop) in items:
        if not 0 < start < stop <= 2**31:
            raise ValueError(f'{name} requires a nonzero, bounded seed range [start, stop)')
    for index, (name, (start, stop)) in enumerate(items):
        for other, (a, b) in items[index + 1:]:
            if max(start, a) < min(stop, b):
                raise ValueError(f'seed ranges overlap: {name} and {other}')


def next_seeds(cursor: int, count: int, stop: int, excluded: set[int]) -> tuple[list[int], int]:
    if count < 1:
        raise ValueError('games must be positive')
    seeds = []
    while len(seeds) < count and cursor < stop:
        if cursor != 0 and cursor not in excluded:
            seeds.append(cursor)
        cursor += 1
    if len(seeds) != count:
        raise ValueError('training seed range exhausted; start a new run with disjoint ranges')
    return seeds, cursor


def seed_exclusions(identity: dict, metadata: dict) -> set[int]:
    result = set(identity['bc']['known_game_seeds'])
    result.update(metadata.get('training_seeds', []))
    result.update(identity.get('critic_training_seeds', []))
    result.update(identity.get('critic_validation_seeds', []))
    return result
