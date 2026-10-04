"""Complete bounded single-device critic and PPO stages with resumable seed cursors."""
from __future__ import annotations

from dataclasses import fields
import json
import math
from pathlib import Path
import time

from .provenance import (OBJECTIVE_CONTRACT, PIPELINE_CONTRACT, bc_provenance,
                         check_seed_ranges, establish_run, file_hash, next_seeds,
                         source_identity, verify_bc_policy, require_training_state, write_json)
from .execution import collection_options, execution_identity, learner_options


def actor_hash(params: dict) -> str:
    from ..full_action.checkpoints import policy_hash
    return policy_hash({key: value for key, value in params.items() if key != 'value'})


def validate_settings(config: dict) -> None:
    from .sampling import ACTION_SELECTION_CONTRACT, validate_action_selection_contract
    validate_action_selection_contract(config.get('action_selection_contract', ACTION_SELECTION_CONTRACT))
    if config.get('collection_backend', 'official-python') not in ('official-python', 'rust-batch'):
        raise ValueError('collection_backend must be official-python or rust-batch')
    if config.get('season_controller') is not None:
        from .controllers import validate_config
        validate_config(config['season_controller'])
    for key in ('games', 'minibatch_size', 'epochs', 'seed', 'environment_seed_start', 'training_seed_limit'):
        value = config[key]
        if not isinstance(value, int) or isinstance(value, bool) or value < (0 if key == 'seed' else 1):
            raise ValueError(f'{key} must be an integer in its supported range')
    if config['minibatch_size'] < 2 or config['minibatch_size'] % 2:
        raise ValueError('minibatch_size must be even and >= 2 to preserve paired seats')
    if config['compute_dtype'] not in ('float32', 'bfloat16'):
        raise ValueError('compute_dtype must be float32 or bfloat16')
    if config['gamma'] != 1:
        raise ValueError('this terminal match-score objective requires gamma=1')
    for key, value in config.items():
        if isinstance(value, float) and not math.isfinite(value):
            raise ValueError(f'{key} must be finite')
    if 'validation_games' in config:
        for key in ('validation_games', 'validation_seed_start', 'validation_seed_limit', 'patience'):
            if not isinstance(config[key], int) or isinstance(config[key], bool) or config[key] < 1:
                raise ValueError(f'{key} must be a positive integer')


def require_device() -> None:
    import jax
    from ..replay_rules import require_pinned_rules
    if jax.process_count() != 1 or len(jax.local_devices()) != 1:
        raise ValueError('this PPO pipeline supports one process and one visible JAX device')
    require_pinned_rules()



def check_memory(config: dict) -> None:
    from ..full_action.memory import host_memory_status
    # Full float32 observations, metadata, and bool legal support per seat.
    per_transition = 264 * 124 * 4 + 20 * 500 + 10 * 1903 + 264 * 16 + 2048
    games = max(config['games'], config.get('validation_games', 0))
    rollout_bytes = 719 * 2 * games * per_transition
    estimated = rollout_bytes + config['minibatch_size'] * per_transition * 3 + 4 * 2**30
    memory = host_memory_status()
    print(json.dumps({'event': 'ppo_host_memory', **memory,
                      'rollout_budget_bytes': rollout_bytes,
                      'estimated_host_budget_bytes': estimated}), flush=True)
    available = memory['effective_available_bytes']
    if available is not None and estimated > available * .8:
        raise ValueError('PPO host memory budget exceeds available container memory; reduce games/minibatch in a new run')


def model_from(payload):
    from ..full_action.model import JaxModelConfig
    model = JaxModelConfig(**payload['model_config'])
    model.validate()
    if model.rope_correction_backend != 'partitioned' or not model.absolute_sell or model.dropout != 0:
        raise ValueError('PPO requires the project partitioned, absolute-SELL model with dropout=0')
    if 'value' not in payload['params']:
        raise ValueError('policy is missing the project value head')
    return model


def numerical_config(config: dict, *, critic=False):
    from .objective import PPOConfig
    keys = {field.name for field in fields(PPOConfig)}
    values = {key: value for key, value in config.items() if key in keys}
    if critic:
        values['warmup_steps'] = 0
    return PPOConfig(**values)


def flatten_for_learning(rollout, config: dict) -> dict:
    from .objective import generalized_advantage_estimate
    from .rollout import flatten_rollout
    from .sampling import ACTION_SELECTION_CONTRACT
    expected = config.get('action_selection_contract', ACTION_SELECTION_CONTRACT)
    actual = rollout.metadata.get('action_selection_contract')
    if actual != expected and (actual is not None or expected != ACTION_SELECTION_CONTRACT):
        raise ValueError('collected action contract differs from the configured learner')
    if config.get('season_controller') is not None:
        from .controllers import contract_identity
        if rollout.metadata.get('controller_identity') != contract_identity(config['season_controller']):
            raise ValueError('collected continuation controller differs from the configured learner')
    elif rollout.metadata.get('controller_identity') is not None:
        raise ValueError('unexpected external continuation controller in neural-only training')
    advantages, returns = generalized_advantage_estimate(
        rollout.values, rollout.terminal_scores, gamma=config['gamma'], gae_lambda=config['gae_lambda'])
    return flatten_rollout(rollout, advantages, returns)


def append_metric(output: Path, row: dict) -> None:
    with (output / 'metrics.jsonl').open('a') as stream:
        stream.write(json.dumps(row, sort_keys=True) + '\n')
    print(json.dumps(row, sort_keys=True), flush=True)


def common_identity(phase, initial, config, bc):
    return {'contract': PIPELINE_CONTRACT, 'phase': phase, 'objective': OBJECTIVE_CONTRACT,
            **execution_identity(config), 'source': source_identity(),
            'initial_sha256': file_hash(initial), 'settings': config, 'bc': bc,
            'deployment': 'candidate_only'}


def warmup(args, config: dict) -> None:
    import numpy as np
    from ..full_action.checkpoints import load_training_source
    from .rollout import collect_rollout, flatten_rollout
    from .trainer import PPOTrainer

    validate_settings(config)
    check_seed_ranges({'critic': (config['environment_seed_start'], config['training_seed_limit']),
                       'critic_validation': (config['validation_seed_start'], config['validation_seed_limit'])})
    require_device()
    check_memory(config)
    bc = bc_provenance(args.bc_run)
    membership = verify_bc_policy(args.bc_run, args.initial)
    initial = load_training_source(args.initial)
    if initial.get('action_selection_contract'):
        raise ValueError('critic stage starts from an original project BC inference policy')
    model = model_from(initial)
    frozen_hash = actor_hash(initial['params'])
    identity = common_identity('critic', args.initial, config, bc)
    if identity['initial_sha256'] != membership['policy_file_sha256']:
        raise ValueError('BC policy changed while initializing critic; use a frozen copy')
    identity['bc_policy_membership'] = membership
    identity.update(frozen_actor_sha256=frozen_hash,
                    reserved_training_ranges=[[config['environment_seed_start'], config['training_seed_limit']],
                                              [config['validation_seed_start'], config['validation_seed_limit']]])
    known = set(bc['known_game_seeds'])
    validation_seeds, _ = next_seeds(config['validation_seed_start'], config['validation_games'],
                                    config['validation_seed_limit'], known)
    identity['critic_validation_seeds'] = validation_seeds
    establish_run(args.out, identity, config, args.updates)
    numerical = numerical_config(config, critic=True)
    latest = args.out / 'latest_critic_state.pkl'
    require_training_state(args.out, latest.name, 'critic')
    if latest.exists():
        learner, meta = PPOTrainer.load(latest, model, numerical, identity)
    else:
        learner = PPOTrainer(model, numerical, initial['params'], initial['params'],
                             compute_dtype=config['compute_dtype'], **learner_options(config, identity))
        meta = {'completed_updates': 0, 'next_seed': config['environment_seed_start'],
                'training_seeds': [], 'validation_seeds': validation_seeds,
                'best_validation_mse': None, 'best_update': None, 'stale_updates': 0,
                'env_steps': 0}
    if actor_hash(learner.params) != frozen_hash:
        raise ValueError('critic state changed its frozen actor/trunk')

    def validate_current():
        # GAE is the bootstrapped training target; independent validation uses
        # the actual Monte Carlo terminal score to select the best fitted head.
        validation = collect_rollout(learner.params, model.to_dict(), validation_seeds,
                                     sampling_seed=config['seed'], compute_dtype=config['compute_dtype'],
                                     **collection_options(config))
        monte_carlo = np.broadcast_to(validation.terminal_scores, validation.values.shape).copy()
        valid_data = flatten_rollout(validation, np.zeros_like(monte_carlo), monte_carlo)
        metrics = learner.validate_values(valid_data, config['minibatch_size'])
        if not math.isfinite(float(metrics['value_mse'])):
            raise RuntimeError('nonfinite critic validation MSE')
        return metrics

    def export_best(iteration):
        learner.export_policy(args.out / 'policy_with_critic.pkl',
                              {'stage': 'critic', 'frozen_actor_sha256': frozen_hash,
                               'source_policy_sha256': identity['initial_sha256'],
                               'critic_integration_sha256': file_hash(args.out / 'integration.json'),
                               'best_update': iteration})
        meta['best_policy_sha256'] = file_hash(args.out / 'policy_with_critic.pkl')

    if meta['best_validation_mse'] is None:
        baseline = validate_current()
        meta.update(best_validation_mse=float(baseline['value_mse']), best_update=0)
        export_best(0)
        learner.save(latest, identity, meta)
        append_metric(args.out, {'event': 'critic_baseline', 'update': 0, 'validation': baseline,
                                 'frozen_actor_sha256': frozen_hash})
    for iteration in range(meta['completed_updates'] + 1, args.updates + 1):
        if meta['stale_updates'] >= config['patience']:
            print(json.dumps({'event': 'critic_patience', **meta}), flush=True)
            break
        started = time.monotonic()
        seeds, cursor = next_seeds(meta['next_seed'], config['games'], config['training_seed_limit'], known)
        rollout = collect_rollout(learner.params, model.to_dict(), seeds,
                                  sampling_seed=config['seed'] + iteration,
                                  compute_dtype=config['compute_dtype'], **collection_options(config))
        data = flatten_for_learning(rollout, config)
        train_metrics = learner.warmup_rollout(data, config['minibatch_size'], config['epochs'],
                                               config['seed'] + iteration)
        train_diagnostics = rollout.metadata
        del rollout, data
        if actor_hash(learner.params) != frozen_hash:
            raise RuntimeError('critic warmup changed the frozen actor/trunk')
        # Actor/sampling RNG stay fixed for the validation panel across iterations.
        validation_metrics = validate_current()
        validation_mse = float(validation_metrics['value_mse'])
        best = meta['best_validation_mse'] is None or validation_mse < meta['best_validation_mse']
        meta.update(completed_updates=iteration, next_seed=cursor,
                    training_seeds=meta['training_seeds'] + seeds,
                    env_steps=meta['env_steps'] + 1438 * len(seeds),
                    stale_updates=0 if best else meta['stale_updates'] + 1)
        if best:
            meta.update(best_validation_mse=validation_mse, best_update=iteration)
            export_best(iteration)
        learner.save(latest, identity, meta)
        append_metric(args.out, {'event': 'critic_update', 'update': iteration,
                                 'seconds': time.monotonic() - started, 'training': train_metrics,
                                 'validation': validation_metrics, 'best_update': meta['best_update'],
                                 'frozen_actor_sha256': frozen_hash, 'seeds': seeds,
                                 'rollout_diagnostics': train_diagnostics})
    if not (args.out / 'policy_with_critic.pkl').exists():
        raise RuntimeError('critic stage has no completed best policy')
    write_json(args.out / 'receipt.json', {'contract': PIPELINE_CONTRACT, 'phase': 'critic',
               'objective': OBJECTIVE_CONTRACT, 'deployment': 'candidate_only', **meta,
               'frozen_actor_sha256': frozen_hash, 'actor_unchanged': True,
               'policy_sha256': file_hash(args.out / 'policy_with_critic.pkl')})


def train(args, config: dict) -> None:
    from ..full_action.checkpoints import load_training_source, policy_hash
    from .rollout import collect_rollout
    from .trainer import PPOTrainer, checkpoint_metadata

    validate_settings(config)
    require_device()
    check_memory(config)
    critic_identity = json.loads((args.critic_run / 'integration.json').read_text())
    if (critic_identity.get('phase') != 'critic' or critic_identity.get('contract') != PIPELINE_CONTRACT
            or critic_identity.get('source') != source_identity()):
        raise ValueError('critic contract/source mismatch; use its matching pipeline')
    critic_meta = checkpoint_metadata(args.critic_run / 'latest_critic_state.pkl', expected_identity=critic_identity)
    receipt_path = args.critic_run / 'receipt.json'
    if not receipt_path.exists():
        raise ValueError('critic stage is incomplete; wait for its receipt before starting PPO')
    critic_receipt = json.loads(receipt_path.read_text())
    initial, teacher = load_training_source(args.initial), load_training_source(args.teacher)
    if (file_hash(args.teacher) != critic_identity['initial_sha256']
            or initial.get('source_policy_sha256') != file_hash(args.teacher)
            or initial.get('critic_integration_sha256') != file_hash(args.critic_run / 'integration.json')
            or file_hash(args.initial) != critic_receipt['policy_sha256']
            or file_hash(args.initial) != critic_meta['best_policy_sha256']):
        raise ValueError('critic initial/teacher/receipt identity mismatch; use the original BC and best critic policy')
    model = model_from(initial)
    if initial['model_config'] != teacher['model_config']:
        raise ValueError('critic and BC teacher models differ')
    if actor_hash(initial['params']) != actor_hash(teacher['params']):
        raise ValueError('critic policy changed the frozen BC actor/trunk')
    c = critic_identity['settings']
    from .sampling import ACTION_SELECTION_CONTRACT
    if (config.get('action_selection_contract', ACTION_SELECTION_CONTRACT) != critic_identity['execution']
            or config.get('season_controller') != c.get('season_controller')):
        raise ValueError('critic and PPO continuation execution differ; fit a new critic with the matching rules/controller')
    check_seed_ranges({'ppo': (config['environment_seed_start'], config['training_seed_limit']),
                       'critic': (c['environment_seed_start'], c['training_seed_limit']),
                       'critic_validation': (c['validation_seed_start'], c['validation_seed_limit'])})
    identity = common_identity('ppo', args.initial, config, critic_identity['bc'])
    identity.update(teacher_sha256=file_hash(args.teacher),
                    teacher_policy_sha256=policy_hash(teacher['params']),
                    critic_integration_sha256=file_hash(args.critic_run / 'integration.json'),
                    critic_receipt_sha256=file_hash(receipt_path),
                    critic_training_seeds=critic_meta['training_seeds'],
                    critic_validation_seeds=critic_meta['validation_seeds'],
                    reserved_training_ranges=[[config['environment_seed_start'], config['training_seed_limit']],
                                              [c['environment_seed_start'], c['training_seed_limit']],
                                              [c['validation_seed_start'], c['validation_seed_limit']]])
    establish_run(args.out, identity, config, args.updates)
    numerical = numerical_config(config)
    latest = args.out / 'latest_ppo_state.pkl'
    require_training_state(args.out, latest.name, 'ppo')
    if latest.exists():
        learner, meta = PPOTrainer.load(latest, model, numerical, identity)
    else:
        learner = PPOTrainer(model, numerical, initial['params'], teacher['params'],
                             compute_dtype=config['compute_dtype'], **learner_options(config, identity))
        meta = {'completed_updates': 0, 'next_seed': config['environment_seed_start'],
                'training_seeds': [], 'env_steps': 0}
    excluded = (set(identity['bc']['known_game_seeds']) | set(identity['critic_training_seeds'])
                | set(identity['critic_validation_seeds']))
    for iteration in range(meta['completed_updates'] + 1, args.updates + 1):
        started = time.monotonic()
        seeds, cursor = next_seeds(meta['next_seed'], config['games'], config['training_seed_limit'], excluded)
        # Both seats use exactly the policy version at this rollout boundary.
        continuation_version = policy_hash(learner.params)
        rollout = collect_rollout(learner.params, model.to_dict(), seeds,
                                  sampling_seed=config['seed'] + iteration,
                                  compute_dtype=config['compute_dtype'], **collection_options(config))
        data = flatten_for_learning(rollout, config)
        metrics = learner.update_rollout(data, config['minibatch_size'], config['epochs'],
                                          config['seed'] + iteration)
        meta.update(completed_updates=iteration, next_seed=cursor,
                    training_seeds=meta['training_seeds'] + seeds,
                    env_steps=meta['env_steps'] + 1438 * len(seeds),
                    continuation_policy_version=continuation_version)
        policy_metadata = {'stage': 'ppo', 'completed_rollout_updates': iteration,
                           'env_steps': meta['env_steps'],
                           'run_integration_sha256': file_hash(args.out / 'integration.json')}
        learner.export_policy(args.out / 'policy_latest_jax.pkl', policy_metadata)
        learner.export_policy(args.out / f'policy-update-{iteration}.pkl', policy_metadata)
        learner.save(latest, identity, meta)
        append_metric(args.out, {'event': 'ppo_update', 'update': iteration,
                                 'seconds': time.monotonic() - started, 'training': metrics,
                                 'seeds': seeds, 'env_steps': meta['env_steps'],
                                 'rollout_env_steps': 1438 * len(seeds),
                                 'continuation_policy_version': continuation_version,
                                 'rollout_diagnostics': rollout.metadata})
        del rollout, data
    if not latest.exists():
        raise RuntimeError('PPO stage has no completed update')
    write_json(args.out / 'receipt.json', {'contract': PIPELINE_CONTRACT, 'phase': 'ppo',
               'objective': OBJECTIVE_CONTRACT, 'deployment': 'candidate_only', **meta,
               'policy_sha256': file_hash(args.out / 'policy_latest_jax.pkl')})
