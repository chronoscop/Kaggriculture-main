"""Project-owned frozen-actor critic fitting, self-play PPO, and paired verification."""
from __future__ import annotations

import argparse
from importlib import metadata
import json
from pathlib import Path

CONFIG_ROOT = Path(__file__).resolve().parent / 'ppo/configs'


def settings(args) -> dict:
    kind = 'critic' if args.command == 'warmup' else 'ppo'
    config = json.loads((CONFIG_ROOT / f'{kind}.json').read_text())
    if args.config:
        overrides = json.loads(args.config.read_text())
        if not isinstance(overrides, dict) or set(overrides) - set(config):
            raise ValueError('config contains unknown settings; use the versioned project preset keys')
        config.update(overrides)
    for name in ('games', 'minibatch_size', 'epochs', 'compute_dtype', 'environment_seed_start',
                 'collection_backend', 'action_selection_contract'):
        value = getattr(args, name, None)
        if value is not None:
            config[name] = value
    if args.command == 'warmup' and args.validation_games is not None:
        config['validation_games'] = args.validation_games
    if getattr(args, 'season_search', False) or getattr(args, 'controller_config', None):
        config['season_controller'] = controller_settings(args)
    return config


def controller_settings(args) -> dict | None:
    if args.controller_config:
        value = json.loads(args.controller_config.read_text())
        if not isinstance(value, dict):
            raise ValueError('controller config must be a JSON object')
        return value
    if args.season_search:
        from .ppo.controllers import DEFAULTS
        return dict(DEFAULTS)
    return None


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    commands = parser.add_subparsers(dest='command', required=True)
    commands.add_parser('doctor', help='show dependencies, contracts and devices without training')
    for kind in ('warmup', 'train'):
        command = commands.add_parser(kind, help='fit only the critic' if kind == 'warmup' else 'train a self-play PPO candidate')
        command.add_argument('--initial', type=Path, required=True)
        command.add_argument('--out', type=Path, required=True)
        command.add_argument('--updates', type=int, default=100, help='cumulative completed rollout-update target')
        command.add_argument('--config', type=Path, help='JSON overrides using project preset keys')
        command.add_argument('--games', type=int, help='complete games per rollout, both seats')
        command.add_argument('--minibatch-size', type=int, help='even paired-seat minibatch')
        command.add_argument('--epochs', type=int, help='epochs per rollout')
        command.add_argument('--compute-dtype', choices=('float32', 'bfloat16'))
        command.add_argument('--environment-seed-start', type=int)
        command.add_argument('--collection-backend', choices=('rust-batch', 'official-python'))
        command.add_argument('--execution', dest='action_selection_contract',
                             choices=('official-prefix-full-action-ppo-v1', 'official-prefix-economic-full-action-ppo-v2'))
        command.add_argument('--season-search', action='store_true', help='use the same optional final-day continuation in collection and deployment')
        command.add_argument('--controller-config', type=Path, help='explicit final-day search configuration JSON')
        if kind == 'warmup':
            command.add_argument('--bc-run', type=Path, required=True)
            command.add_argument('--validation-games', type=int)
        else:
            command.add_argument('--critic-run', type=Path, required=True)
            command.add_argument('--teacher', type=Path, required=True, help='original fixed BC actor used by critic fitting')
    match = commands.add_parser('evaluate', help='paired screening or independent confirmation; no automatic deployment')
    match.add_argument('--run', type=Path, required=True)
    match.add_argument('--comparison', choices=('learning', 'controller'), default='learning')
    match.add_argument('--policy', type=Path, help='default: policy_latest_jax.pkl')
    match.add_argument('--baseline', type=Path, required=True)
    match.add_argument('--opponent', type=Path, required=True)
    match.add_argument('--seed', type=int, required=True)
    match.add_argument('--games', type=int, default=16, help='games per policy; total cost is twice this')
    match.add_argument('--phase', choices=('screen', 'confirmation'), default='screen')
    match.add_argument('--screen-report', type=Path)
    match.add_argument('--out', type=Path, required=True)
    candidate = commands.add_parser('candidate', help='create a controller candidate from frozen weights without training')
    candidate.add_argument('--policy', type=Path, required=True)
    candidate.add_argument('--out', type=Path, required=True)
    candidate.add_argument('--execution', dest='action_selection_contract',
                           choices=('official-prefix-full-action-ppo-v1', 'official-prefix-economic-full-action-ppo-v2'),
                           default='official-prefix-economic-full-action-ppo-v2')
    candidate.add_argument('--season-search', action='store_true')
    candidate.add_argument('--controller-config', type=Path)
    args = parser.parse_args()
    try:
        if args.command == 'doctor':
            from .ppo.provenance import OBJECTIVE_CONTRACT, PIPELINE_CONTRACT, source_identity
            from .ppo.sampling import ECONOMIC_ACTION_SELECTION_CONTRACT, SUPPORTED_ACTION_SELECTION_CONTRACTS
            from .ppo.native_backend import backend_identity
            import jax
            dependencies = {}
            for name in ('jax', 'numpy', 'optax', 'kaggle-environments'):
                try:
                    dependencies[name] = metadata.version(name)
                except metadata.PackageNotFoundError:
                    dependencies[name] = None
            try:
                native = {'available': True, **backend_identity('rust-batch')}
            except (ImportError, FileNotFoundError, ValueError, RuntimeError) as error:
                native = {'available': False, 'reason': str(error), 'build': 'python tools/build_action_native.py'}
            print(json.dumps({'contract': PIPELINE_CONTRACT, 'objective': OBJECTIVE_CONTRACT,
                              'execution': ECONOMIC_ACTION_SELECTION_CONTRACT,
                              'supported_execution': SUPPORTED_ACTION_SELECTION_CONTRACTS,
                              'collection_default': 'rust-batch', 'native': native, 'source': source_identity(),
                              'dependencies': dependencies, 'devices': [str(d) for d in jax.devices()],
                              'reference_checkout_required': False,
                              'deployment': 'candidate_only'}))
        elif args.command == 'evaluate':
            from .ppo.evaluation import evaluate
            evaluate(args)
        elif args.command == 'candidate':
            from .ppo.candidate import create_candidate
            print(json.dumps(create_candidate(args.policy, args.out,
                             action_selection_contract=args.action_selection_contract,
                             season_controller=controller_settings(args)), sort_keys=True))
        else:
            from .ppo.pipeline import warmup, train
            (warmup if args.command == 'warmup' else train)(args, settings(args))
    except (ValueError, RuntimeError, FileNotFoundError) as error:
        parser.error(str(error))


if __name__ == '__main__':
    main()
