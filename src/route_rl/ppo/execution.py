"""Explicit collection and continuation identities for new candidate runs."""
from __future__ import annotations

from .sampling import ACTION_SELECTION_CONTRACT


def collection_options(config: dict) -> dict:
    """Omitted fields retain the original official-Python/v1 semantics."""
    return {name: config[name] for name in
            ('collection_backend', 'action_selection_contract', 'season_controller') if name in config}


def execution_identity(config: dict) -> dict:
    from .sampling import validate_action_selection_contract
    contract = validate_action_selection_contract(config.get('action_selection_contract', ACTION_SELECTION_CONTRACT))
    backend = config.get('collection_backend', 'official-python')
    if backend not in ('official-python', 'rust-batch'):
        raise ValueError('collection_backend must be official-python or rust-batch')
    if backend == 'rust-batch':
        from .native_backend import backend_identity
        collection = backend_identity('rust-batch')
    else:
        collection = {'backend': 'official-python', 'rules': 'kaggle-environments==1.32.7'}
    controller = config.get('season_controller')
    result = {'execution': contract, 'collection': collection}
    if controller is not None:
        from .controllers import contract_identity, validate_config
        validate_config(controller)
        result['controller_identity'] = contract_identity(controller)
    return result


def learner_options(config: dict, identity: dict) -> dict:
    return {'action_selection_contract': identity['execution'],
            'season_controller': config.get('season_controller'),
            'controller_identity': identity.get('controller_identity')}
