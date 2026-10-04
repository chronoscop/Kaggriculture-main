"""Create explicit controller candidates from frozen weights without training."""
from __future__ import annotations

import hashlib
from pathlib import Path
import tempfile

from ..full_action.checkpoints import atomic_pickle, load_training_source
from .provenance import OBJECTIVE_CONTRACT, source_identity
from .sampling import validate_action_selection_contract


def create_candidate(policy: Path, output: Path, *, action_selection_contract: str,
                     season_controller: dict | None = None) -> dict:
    if output.exists():
        raise FileExistsError('candidate output already exists; choose a new path')
    validate_action_selection_contract(action_selection_contract)
    data = Path(policy).read_bytes()
    with tempfile.TemporaryDirectory(prefix='kaggriculture-candidate-') as directory:
        frozen = Path(directory) / 'base.pkl'
        frozen.write_bytes(data)
        payload = load_training_source(frozen)
    if payload.get('season_controller') is not None:
        raise ValueError('start a controller comparison from the frozen neural policy, without an existing handoff')
    identity = None
    if season_controller is not None:
        from .controllers import contract_identity
        identity = contract_identity(season_controller)
    result = {**payload, 'action_selection_contract': action_selection_contract,
              'learning_objective_contract': OBJECTIVE_CONTRACT,
              'season_controller': season_controller, 'controller_identity': identity,
              'candidate_origin': 'frozen-controller-adaptation-v1',
              'candidate_source': source_identity(),
              'base_policy_file_sha256': hashlib.sha256(data).hexdigest(),
              'base_policy_sha256': payload['policy_sha256'],
              'base_action_selection_contract': payload.get('action_selection_contract', 'legacy-bc-greedy-v1'),
              'deployment': 'candidate_only'}
    atomic_pickle(output, result)
    return {'candidate': str(output), 'base_policy_file_sha256': result['base_policy_file_sha256'],
            'policy_sha256': result['policy_sha256'], 'action_selection_contract': action_selection_contract,
            'controller_identity': identity, 'deployment': 'candidate_only', 'trained': False}
