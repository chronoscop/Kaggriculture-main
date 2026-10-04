"""Paired candidate/baseline screening and independent confirmation; no deployment."""
from __future__ import annotations

from contextlib import ExitStack
import json
import hashlib
import math
from pathlib import Path
import select
import subprocess
import sys
import tempfile

from ..replay_rules import require_pinned_rules
from .provenance import (OBJECTIVE_CONTRACT, PIPELINE_CONTRACT, file_hash,
                         seed_exclusions, source_identity, write_json)


class AgentProcess:
    def __init__(self, kind: str, path: Path, source_context: Path | None = None):
        command = [sys.executable, '-u', '-m', 'route_rl.ppo.agent_worker', kind, str(path.resolve())]
        if source_context is not None:
            command.append(str(source_context.resolve()))
        self.process = subprocess.Popen(
            command,
            stdin=subprocess.PIPE, stdout=subprocess.PIPE, text=True, bufsize=1)
        try:
            if not select.select([self.process.stdout], [], [], 120)[0]:
                raise TimeoutError('agent initialization timed out')
            line = self.process.stdout.readline()
            if not line or json.loads(line) != {'ready': True}:
                raise RuntimeError('agent initialization failed; see stderr')
        except Exception:
            self.close()
            raise

    def __call__(self, observation, configuration):
        self.process.stdin.write(json.dumps({'observation': observation, 'configuration': configuration}) + '\n')
        self.process.stdin.flush()
        if not select.select([self.process.stdout], [], [], 120)[0]:
            raise TimeoutError('agent response timed out')
        line = self.process.stdout.readline()
        if not line:
            raise RuntimeError(f'agent exited with status {self.process.poll()}')
        response = json.loads(line)
        if 'error' in response:
            raise RuntimeError(response['error'])
        return response['action']

    def close(self):
        if self.process.stdin:
            self.process.stdin.close()
        if self.process.poll() is None:
            self.process.terminate()
        try:
            self.process.wait(timeout=5)
        except subprocess.TimeoutExpired:
            self.process.kill()
            self.process.wait()
        if self.process.stdout:
            self.process.stdout.close()


def score(own, other) -> float:
    if own is None or other is None or not math.isfinite(own) or not math.isfinite(other):
        raise ValueError('terminal rewards must be finite')
    return 1.0 if own > other else 0.5 if own == other else 0.0


def check_panel(seeds: list[int], identity: dict, metadata: dict, prior: dict | None) -> None:
    if not seeds or len(seeds) != len(set(seeds)) or min(seeds) <= 0 or max(seeds) >= 2**31:
        raise ValueError('evaluation requires unique nonzero seeds in the supported range')
    if set(seeds) & seed_exclusions(identity, metadata):
        raise ValueError('evaluation seeds overlap demonstration/critic/PPO data')
    # Entire training ranges are reserved, including future continuations.
    for start, stop in identity.get('reserved_training_ranges', []):
        if any(start <= seed < stop for seed in seeds):
            raise ValueError('evaluation seeds overlap a reserved training range')
    if prior is not None and set(seeds) & set(prior['seeds']):
        raise ValueError('confirmation must use new seeds independent of screening')


def freeze_inputs(policy: Path, baseline: Path, opponent: Path, destination: Path) -> tuple[dict, dict]:
    """Capture the exact bytes used throughout an entire paired report."""
    paths, hashes = {}, {}
    for name, source in (('policy', policy), ('baseline', baseline), ('opponent', opponent)):
        data = source.read_bytes()
        frozen = destination / (name + ('.py' if name == 'opponent' else '.pkl'))
        frozen.write_bytes(data)
        paths[name] = frozen
        hashes[f'{name}_sha256'] = hashlib.sha256(data).hexdigest()
    return paths, hashes


def evaluate(args) -> dict:
    # Latest candidates may be replaced while BC/PPO training continues. Every
    # worker in this report loads the captured bytes, including the opponent
    # entry source; its original directory remains the context for resources.
    with tempfile.TemporaryDirectory(prefix='kaggriculture-paired-') as directory:
        return _evaluate(args, Path(directory))


def controller_comparison_lineage(args, identity: dict, frozen: dict, input_hashes: dict) -> tuple[dict, dict]:
    """Bind both controllers to the same exact frozen model and owned run."""
    from ..full_action.checkpoints import load_training_source
    from .provenance import bc_provenance
    from .trainer import checkpoint_metadata

    candidate = load_training_source(frozen['policy'])
    baseline = load_training_source(frozen['baseline'])
    if candidate.get('candidate_origin') != 'frozen-controller-adaptation-v1':
        raise ValueError('controller comparison requires an explicit frozen-controller candidate')
    if candidate.get('candidate_source') != source_identity():
        raise ValueError('controller candidate source changed; create a new candidate')
    if candidate['policy_sha256'] != baseline['policy_sha256'] or candidate['model_config'] != baseline['model_config']:
        raise ValueError('controller comparisons require identical frozen weights and model configuration')
    base_hash = candidate['base_policy_file_sha256']
    if baseline.get('candidate_origin') == 'frozen-controller-adaptation-v1':
        if baseline.get('candidate_source') != source_identity() or baseline.get('base_policy_file_sha256') != base_hash:
            raise ValueError('both controller candidates must use the same frozen source policy')
    elif input_hashes['baseline_sha256'] != base_hash:
        raise ValueError('baseline must be the exact frozen source policy or its explicit controller candidate')
    members = []
    for path in args.run.glob('*.pkl'):
        allowed = (path.name in ('final_student_jax.pkl', 'policy_latest_jax.pkl')
                   or path.name.startswith('epoch-') and path.name.endswith('-policy.pkl')
                   or path.name.startswith('policy-update-'))
        if allowed and path.is_file() and not path.is_symlink() and file_hash(path) == base_hash:
            members.append(path)
    if not members:
        raise ValueError('frozen source policy does not belong to the supplied training run')
    origin = load_training_source(members[0])
    if origin['policy_sha256'] != candidate['policy_sha256'] or origin['model_config'] != candidate['model_config']:
        raise ValueError('controller candidate changed the source model')
    if identity.get('phase') == 'ppo':
        if identity.get('contract') != PIPELINE_CONTRACT or identity.get('source') != source_identity():
            raise ValueError('PPO source changed; restore the run source before evaluating')
        if origin.get('run_integration_sha256') != file_hash(args.run / 'integration.json'):
            raise ValueError('frozen source policy has a different PPO lineage')
        metadata = checkpoint_metadata(args.run / 'latest_ppo_state.pkl', expected_identity=identity)
    else:
        bc = bc_provenance(args.run)
        identity = {**identity, 'bc': bc, 'reserved_training_ranges': []}
        metadata = {'training_seeds': []}
    return identity, metadata


def _evaluate(args, snapshots: Path) -> dict:
    from .trainer import checkpoint_metadata
    from ..full_action.checkpoints import load_training_source

    if args.games < 2 or args.games % 2:
        raise ValueError('games is even and >= 2; each seed is played in both seats by each policy')
    identity = json.loads((args.run / 'integration.json').read_text())
    policy = args.policy or args.run / 'policy_latest_jax.pkl'
    frozen, input_hashes = freeze_inputs(policy, args.baseline, args.opponent, snapshots)
    candidate_payload = load_training_source(frozen['policy'])
    comparison = getattr(args, 'comparison', 'learning')
    if comparison == 'controller':
        identity, metadata = controller_comparison_lineage(args, identity, frozen, input_hashes)
    elif comparison == 'learning':
        if identity['contract'] != PIPELINE_CONTRACT or identity['source'] != source_identity():
            raise ValueError('PPO source changed; restore the run source before evaluating')
        metadata = checkpoint_metadata(args.run / 'latest_ppo_state.pkl', expected_identity=identity)
        if candidate_payload.get('run_integration_sha256') != file_hash(args.run / 'integration.json'):
            raise ValueError('candidate does not belong to this PPO run; evaluate with its own lineage')
        if candidate_payload.get('candidate_origin') == 'frozen-controller-adaptation-v1':
            raise ValueError('controller adaptations require --comparison controller, not learning evaluation')
        if (candidate_payload.get('action_selection_contract') != identity.get('execution')
                or candidate_payload.get('season_controller') != identity.get('settings', {}).get('season_controller')
                or candidate_payload.get('controller_identity') != identity.get('controller_identity')):
            raise ValueError('candidate execution/continuation differs from its training run; use --comparison controller')
        if input_hashes['baseline_sha256'] not in {identity['teacher_sha256'], identity['initial_sha256']}:
            raise ValueError("baseline must be this run's original BC teacher or frozen critic initialization")
    else:
        raise ValueError('comparison must be learning or controller')
    raw = args.out.with_suffix('.games.json')
    if args.out.exists() or raw.exists():
        raise ValueError('evaluation output already exists; choose a new path')
    hashes = {**input_hashes, 'run_integration_sha256': file_hash(args.run / 'integration.json'),
              'comparison': comparison}
    prior = None
    if args.phase == 'confirmation':
        if args.screen_report is None:
            raise ValueError('confirmation requires --screen-report from the same frozen policies')
        prior = json.loads(args.screen_report.read_text())
        if prior.get('phase') != 'screen' or prior.get('objective') != OBJECTIVE_CONTRACT:
            raise ValueError('screen report has the wrong phase or objective')
        if any(prior.get(key) != value for key, value in hashes.items()):
            raise ValueError('candidate/baseline/opponent/run changed since screening; screen again')
        if prior['paired_score_delta'] <= 0:
            raise ValueError('screening did not improve match score; candidate remains unaccepted')
    elif args.screen_report is not None:
        raise ValueError('--screen-report is only used for confirmation')
    seeds = list(range(args.seed, args.seed + args.games // 2))
    check_panel(seeds, identity, metadata, prior)
    require_pinned_rules()
    from kaggle_environments import make

    args.out.parent.mkdir(parents=True, exist_ok=True)
    records = []
    # Workers are separate per policy and reset their tracker at each new game.
    for seed in seeds:
        for seat in (0, 1):
            for name, candidate in (('candidate', frozen['policy']), ('baseline', frozen['baseline'])):
                with ExitStack() as stack:
                    own = AgentProcess('policy', candidate)
                    stack.callback(own.close)
                    opponent = AgentProcess('entry', frozen['opponent'], source_context=args.opponent)
                    stack.callback(opponent.close)
                    workers = [own, opponent] if seat == 0 else [opponent, own]
                    environment = make('kaggriculture', configuration={'seed': seed}, debug=False)
                    environment.run(workers)
                    last = environment.steps[-1]
                    row = {'policy': name, 'seed': seed, 'seat': seat,
                           'turns': len(environment.steps), 'statuses': [s.status for s in last],
                           'rewards': [s.reward for s in last]}
                    records.append(row)
                    write_json(raw, {'games': records})
                    if row['turns'] != 720 or row['statuses'] != ['DONE', 'DONE']:
                        raise ValueError('failed/incomplete paired game; inspect raw report')
                    row['score'] = score(row['rewards'][seat], row['rewards'][1 - seat])
                    print(json.dumps({'event': 'paired_game', **row}), flush=True)
    own_scores = [row['score'] for row in records if row['policy'] == 'candidate']
    baseline_scores = [row['score'] for row in records if row['policy'] == 'baseline']
    deltas = [a - b for a, b in zip(own_scores, baseline_scores)]
    delta = sum(deltas) / len(deltas)
    result = {'contract': PIPELINE_CONTRACT, 'objective': OBJECTIVE_CONTRACT, **hashes,
              'phase': args.phase, 'seeds': seeds, 'games_per_policy': len(own_scores),
              'total_games': len(records), 'games': records,
              'candidate_score_rate': sum(own_scores) / len(own_scores),
              'baseline_score_rate': sum(baseline_scores) / len(baseline_scores),
              'paired_score_deltas': deltas, 'paired_score_delta': delta,
              'eligible_for_manual_review': args.phase == 'confirmation' and delta > 0,
              'deployment': 'candidate_only',
              'demonstration_games_without_seed': identity['bc']['games_without_seed'],
              'inference': 'checkpoint-specific BC, prefix/economic PPO, or explicit final-day controller',
              'controller_identity': candidate_payload.get('controller_identity'),
              'action_selection_contract': candidate_payload.get('action_selection_contract', 'legacy-bc-greedy-v1'),
              'continuation_policy_version': hashes['policy_sha256']}
    if prior is not None:
        result['screen_report_sha256'] = file_hash(args.screen_report)
    write_json(raw, {'games': records})
    write_json(args.out, result)
    print(json.dumps({key: value for key, value in result.items()
                      if key not in ('games', 'seeds', 'paired_score_deltas')}), flush=True)
    return result
