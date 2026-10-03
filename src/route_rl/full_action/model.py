# Adapted from msdsm/kaggriculture-solution, commit 84057a0fda4238ccdebc46f9bf5496c6c4b2e00d.
# Source: model/policy.py (manual path); see docs/action_bc_sources.md.
"""Entity Transformer with same-farm 2D RoPE and full-action heads.

The project maintains the manual attention path. Experimental kernels, search
patches and the reference package are outside this BC implementation.
"""
from __future__ import annotations
import math
from dataclasses import dataclass
from typing import Any
import jax
import jax.numpy as jnp
from .catalog import MARKET_ACTIONS, UNIT_ACTIONS
from .sell_quantity import PRODUCT_COUNT, QUANTITY_COUNT, compose_market_logits
from .features import FEATURE_INDEX, MEMORY_PACK_FEATURE_INDICES, TOKEN_ADAPTER_FEATURES

Array = jax.Array
Params = dict[str, Any]
LAYER_NORM_EPSILON = 1e-5
PARTITION_GROUP_TOKENS = 120
PARTITION_NONSPATIAL_TOKENS = 24
PARTITION_FIXED_TOKENS = 264
GRID_MAX_COORDINATE = 9.0
VALUE_COST_DIFFERENCE_REFERENCE = 20_000.0
VALUE_MONEY_ENCODING_REFERENCE = 1_000_000.0
VALUE_HEAD_INIT_SEED = 42_042

@dataclass(frozen=True)
class JaxModelConfig:
    d_model: int = 256
    layers: int = 6
    heads: int = 8
    ffn_dim: int = 1024
    dropout: float = 0.1
    rope_dim: int = 16
    rope_base: float = 100.0
    attention_backend: str = "manual"
    rope_correction_backend: str = "dense"
    legal_mask: bool = False
    absolute_sell: bool = False
    sequential_patch: bool = False

    def validate(self) -> None:
        if self.legal_mask or self.sequential_patch or not self.absolute_sell:
            raise ValueError("project BC requires unmasked full actions and absolute SELL; patches are unsupported")
        if min(self.d_model, self.layers, self.heads, self.ffn_dim) <= 0:
            raise ValueError("model dimensions must be positive")
        if self.sequential_patch and not self.absolute_sell:
            raise ValueError("sequential patch shed sales require absolute quantities")
        if self.sequential_patch and self.legal_mask:
            raise ValueError("sequential patch uses its own support, not independent legal masks")
        if self.absolute_sell and self.legal_mask:
            raise ValueError("absolute SELL currently requires the unmasked policy")
        if self.d_model % self.heads:
            raise ValueError("d_model must be divisible by heads")
        head_dim = self.d_model // self.heads
        if self.rope_dim > head_dim or self.rope_dim % 4:
            raise ValueError("rope_dim must fit in one head and be divisible by four")
        if self.rope_base <= 1.0:
            raise ValueError("rope base must be greater than one")
        if not 0.0 <= self.dropout < 1.0:
            raise ValueError("dropout must be in [0, 1)")
        if self.attention_backend != "manual":
            raise ValueError("project BC supports the manual attention backend")
        if self.rope_correction_backend not in ("dense", "partitioned"):
            raise ValueError("RoPE correction backend must be dense or partitioned")

    def to_dict(self) -> dict[str, int | float | str]:
        return {
            "d_model": self.d_model,
            "layers": self.layers,
            "heads": self.heads,
            "ffn_dim": self.ffn_dim,
            "dropout": self.dropout,
            "rope_dim": self.rope_dim,
            "rope_base": self.rope_base,
            "attention_backend": self.attention_backend,
            "rope_correction_backend": self.rope_correction_backend,
            "legal_mask": self.legal_mask,
            **({"sequential_patch": True} if self.sequential_patch else {}),
            **({"absolute_sell": True} if self.absolute_sell else {}),
        }


def init_dense(key: Array, input_dim: int, output_dim: int) -> Params:
    limit = 1.0 / math.sqrt(input_dim)
    return {
        "kernel": jax.random.uniform(
            key,
            (input_dim, output_dim),
            minval=-limit,
            maxval=limit,
            dtype=jnp.float32,
        ),
        "bias": jax.random.uniform(
            jax.random.fold_in(key, 1),
            (output_dim,),
            minval=-limit,
            maxval=limit,
            dtype=jnp.float32,
        ),
    }


def init_norm(width: int) -> Params:
    return {
        "scale": jnp.ones((width,), dtype=jnp.float32),
        "bias": jnp.zeros((width,), dtype=jnp.float32),
    }


def initialize_params(key: Array, config: JaxModelConfig) -> Params:
    config.validate()

    def next_key() -> Array:
        nonlocal key
        key, result = jax.random.split(key)
        return result

    adapters = {
        token_type: init_dense(next_key(), len(feature_names), config.d_model)
        for token_type, feature_names in TOKEN_ADAPTER_FEATURES.items()
    }
    blocks = []
    for _ in range(config.layers):
        blocks.append(
            {
                "attention_norm": init_norm(config.d_model),
                "qkv": init_dense(next_key(), config.d_model, config.d_model * 3),
                "attention_output": init_dense(next_key(), config.d_model, config.d_model),
                "ffn_norm": init_norm(config.d_model),
                "ffn_input": init_dense(next_key(), config.d_model, config.ffn_dim),
                "ffn_output": init_dense(next_key(), config.ffn_dim, config.d_model),
            }
        )
    params = {
        "adapters": adapters,
        "input_norm": init_norm(config.d_model),
        "blocks": tuple(blocks),
        "final_norm": init_norm(config.d_model),
        "unit_action": init_dense(next_key(), config.d_model, len(UNIT_ACTIONS)),
        "market_action": init_dense(next_key(), config.d_model, len(MARKET_ACTIONS)),
    }
    if config.absolute_sell:
        params["sell_quantity"] = init_dense(next_key(), config.d_model, PRODUCT_COUNT * QUANTITY_COUNT)
    return params


def add_zero_value_head(params: Params, config: JaxModelConfig) -> Params:
    """Add a GELU critic whose zero output preserves the behavior-cloned policy."""
    if "value" in params:
        return params
    return {
        **params,
        "value": {
            "hidden": init_dense(
                jax.random.PRNGKey(VALUE_HEAD_INIT_SEED),
                config.d_model,
                config.d_model,
            ),
            "output": {
                "kernel": jnp.zeros((config.d_model, 1), dtype=jnp.float32),
                "bias": jnp.zeros((1,), dtype=jnp.float32),
            },
        },
    }


def dense(inputs: Array, params: Params, dtype: jnp.dtype) -> Array:
    return jnp.matmul(inputs.astype(dtype), params["kernel"].astype(dtype)) + params["bias"].astype(dtype)


def policy_head(inputs: Array, params: Params, dtype: jnp.dtype) -> Array:
    # BF16 GEMM+BIAS fusion otherwise changes rounding between sampling and selected-log-prob graphs.
    logits = jnp.matmul(inputs.astype(dtype), params["kernel"].astype(dtype), preferred_element_type=jnp.float32)
    return logits + params["bias"].astype(jnp.float32)


def layer_norm(inputs: Array, params: Params, dtype: jnp.dtype) -> Array:
    values = inputs.astype(jnp.float32)
    mean = jnp.mean(values, axis=-1, keepdims=True)
    variance = jnp.mean(jnp.square(values - mean), axis=-1, keepdims=True)
    normalized = (values - mean) * jax.lax.rsqrt(variance + LAYER_NORM_EPSILON)
    output = normalized * params["scale"] + params["bias"]
    return output.astype(dtype)


def apply_dropout(inputs: Array, key: Array | None, rate: float, training: bool) -> Array:
    if not training or rate == 0.0:
        return inputs
    if key is None:
        raise ValueError("training with dropout requires a PRNG key")
    keep_probability = 1.0 - rate
    keep = jax.random.bernoulli(key, keep_probability, inputs.shape)
    return jnp.where(keep, inputs / keep_probability, 0.0).astype(inputs.dtype)


def rotate_axis(
    tensor: Array,
    positions: Array,
    spatial_mask: Array,
    start: int,
    width: int,
    base: float,
) -> Array:
    section = tensor[..., start : start + width]
    frequency_index = jnp.arange(0, width, 2, dtype=jnp.float32)
    inverse_frequency = 1.0 / (base ** (frequency_index / width))
    angles = positions.astype(jnp.float32)[:, None, :, None] * inverse_frequency[None, None, None, :]
    cosine = jnp.cos(angles).astype(section.dtype)
    sine = jnp.sin(angles).astype(section.dtype)
    even = section[..., 0::2]
    odd = section[..., 1::2]
    rotated = jnp.stack((even * cosine - odd * sine, even * sine + odd * cosine), axis=-1).reshape(section.shape)
    selected = jnp.where(spatial_mask[:, None, :, None], rotated, section)
    return tensor.at[..., start : start + width].set(selected)


def apply_2d_rope(
    query: Array,
    key: Array,
    coordinates: Array,
    spatial_mask: Array,
    config: JaxModelConfig,
) -> tuple[Array, Array]:
    axis_dim = config.rope_dim // 2
    x = coordinates[..., 0]
    y = coordinates[..., 1]
    query = rotate_axis(query, x, spatial_mask, 0, axis_dim, config.rope_base)
    query = rotate_axis(query, y, spatial_mask, axis_dim, axis_dim, config.rope_base)
    key = rotate_axis(key, x, spatial_mask, 0, axis_dim, config.rope_base)
    key = rotate_axis(key, y, spatial_mask, axis_dim, axis_dim, config.rope_base)
    return query, key


def dense_rope_correction(
    raw_query: Array,
    raw_key: Array,
    rotated_query: Array,
    rotated_key: Array,
    rope_groups: Array,
    rope_dim: int,
    scale: float,
) -> Array:
    raw_rotary = jnp.einsum(
        "bhtd,bhsd->bhts",
        raw_query[..., :rope_dim],
        raw_key[..., :rope_dim],
    ).astype(jnp.float32)
    rotated_rotary = jnp.einsum(
        "bhtd,bhsd->bhts",
        rotated_query[..., :rope_dim],
        rotated_key[..., :rope_dim],
    ).astype(jnp.float32)
    query_group = rope_groups[:, None, :, None]
    key_group = rope_groups[:, None, None, :]
    same_farm_pair = (query_group != 0) & (query_group == key_group)
    return jnp.where(~same_farm_pair, (raw_rotary - rotated_rotary) * scale, 0.0)


def partition_policy_tokens(
    hidden: Array,
    coordinates: Array,
    spatial_mask: Array,
    rope_groups: Array,
    source_token_mask: Array,
) -> tuple[Array, Array, Array, Array, Array, Array, Array, int]:
    """Reorder compact observations into two fixed farm blocks plus global tokens."""
    if hidden.shape[1] != PARTITION_FIXED_TOKENS:
        raise ValueError(f"partitioned attention requires {PARTITION_FIXED_TOKENS} tokens")
    batch_size = hidden.shape[0]
    group = PARTITION_GROUP_TOKENS
    unit_slots = group - 100
    unit_offsets = jnp.arange(unit_slots, dtype=jnp.int32)[None, :]
    own_counts = jnp.sum(rope_groups == 1, axis=-1, dtype=jnp.int32) - 100
    opponent_counts = jnp.sum(rope_groups == 2, axis=-1, dtype=jnp.int32) - 100
    own_units = jnp.where(unit_offsets < own_counts[:, None], 201 + unit_offsets, 0)
    opponent_units = jnp.where(
        unit_offsets < opponent_counts[:, None],
        201 + own_counts[:, None] + unit_offsets,
        0,
    )
    self_cells = jnp.broadcast_to(jnp.arange(1, 101, dtype=jnp.int32), (batch_size, 100))
    opponent_cells = jnp.broadcast_to(jnp.arange(101, 201, dtype=jnp.int32), (batch_size, 100))
    tail_offsets = jnp.arange(PARTITION_NONSPATIAL_TOKENS - 1, dtype=jnp.int32)[None, :]
    tail = 201 + own_counts[:, None] + opponent_counts[:, None] + tail_offsets
    global_token = jnp.zeros((batch_size, 1), dtype=jnp.int32)
    indices = jnp.concatenate(
        (self_cells, own_units, opponent_cells, opponent_units, global_token, tail),
        axis=1,
    )

    def gather(values: Array) -> Array:
        gather_indices = indices.reshape((*indices.shape, *((1,) * (values.ndim - 2))))
        gather_indices = jnp.broadcast_to(gather_indices, (*indices.shape, *values.shape[2:]))
        return jnp.take_along_axis(values, gather_indices, axis=1)

    valid_units = jnp.arange(unit_slots)[None, :]
    token_mask = jnp.concatenate(
        (
            jnp.ones((batch_size, 100), dtype=jnp.bool_),
            valid_units < own_counts[:, None],
            jnp.ones((batch_size, 100), dtype=jnp.bool_),
            valid_units < opponent_counts[:, None],
            jnp.ones((batch_size, PARTITION_NONSPATIAL_TOKENS), dtype=jnp.bool_),
        ),
        axis=1,
    )
    token_mask &= gather(source_token_mask)
    hidden = gather(hidden) * token_mask[..., None].astype(hidden.dtype)
    unit_indices = jnp.where(
        valid_units < own_counts[:, None],
        jnp.arange(100, 120, dtype=jnp.int32)[None, :],
        240,
    )
    market_indices = jnp.broadcast_to(jnp.arange(254, 264, dtype=jnp.int32), (batch_size, 10))
    return (
        hidden,
        gather(coordinates),
        gather(spatial_mask),
        gather(rope_groups),
        token_mask,
        unit_indices,
        market_indices,
        240,
    )


def metadata_from_features(features: Array) -> tuple[Array, Array, Array, Array]:
    """Recover exact spatial metadata already encoded in the dense feature rows."""
    token_columns = jnp.asarray(
        [FEATURE_INDEX[f"token:{token_type}"] for token_type in TOKEN_ADAPTER_FEATURES],
        dtype=jnp.int32,
    )
    token_mask = jnp.any(jnp.take(features, token_columns, axis=-1) > 0.5, axis=-1)
    cell_mask = features[..., FEATURE_INDEX["token:CELL"]] > 0.5
    unit_mask = features[..., FEATURE_INDEX["token:UNIT"]] > 0.5
    spatial_mask = cell_mask | unit_mask
    self_mask = features[..., FEATURE_INDEX["farm:SELF"]] > 0.5
    opponent_mask = features[..., FEATURE_INDEX["farm:OPPONENT"]] > 0.5
    rope_groups = jnp.where(spatial_mask & self_mask, 1, jnp.where(spatial_mask & opponent_mask, 2, 0))
    x = features[..., FEATURE_INDEX["cell_x"]] + features[..., FEATURE_INDEX["unit_x"]]
    y = features[..., FEATURE_INDEX["cell_y"]] + features[..., FEATURE_INDEX["unit_y"]]
    coordinates = jnp.stack((x, y), axis=-1) * GRID_MAX_COORDINATE
    return coordinates, spatial_mask, rope_groups, token_mask


def own_unit_mask_from_features(features: Array) -> Array:
    """Recover the fixed policy-head mask from self Unit feature rows."""
    unit_rows = features[..., FEATURE_INDEX["token:UNIT"]] > 0.5
    self_rows = features[..., FEATURE_INDEX["farm:SELF"]] > 0.5
    unit_counts = jnp.sum(unit_rows & self_rows, axis=-1, dtype=jnp.int32)
    return jnp.arange(PARTITION_GROUP_TOKENS - 100)[None, :] < unit_counts[:, None]


def packed_memory_features(features: Array) -> Array:
    memory_rows = jnp.argmax(features[..., FEATURE_INDEX["token:MEMORY"]], axis=1)
    memory_indices = jnp.asarray(MEMORY_PACK_FEATURE_INDICES, dtype=jnp.int32)
    packed = jnp.take(features, memory_indices, axis=-1)
    return jnp.take_along_axis(packed, memory_rows[:, None, None], axis=1)[:, 0]


def typed_input_projection(params: Params, features: Array, memory_features: Array, dtype: jnp.dtype) -> Array:
    projected = jnp.zeros((*features.shape[:2], params["input_norm"]["scale"].shape[0]), dtype=dtype)
    for token_type, feature_names in TOKEN_ADAPTER_FEATURES.items():
        if token_type == "MEMORY":
            token_projection = dense(memory_features, params["adapters"][token_type], dtype)[:, None, :]
        else:
            indices = jnp.asarray([FEATURE_INDEX[name] for name in feature_names], dtype=jnp.int32)
            active_features = jnp.take(features, indices, axis=-1)
            token_projection = dense(active_features, params["adapters"][token_type], dtype)
        type_mask = features[..., FEATURE_INDEX[f"token:{token_type}"]].astype(dtype)[..., None]
        projected += token_projection * type_mask
    return projected


def parameter_count(params: Params) -> int:
    return sum(int(leaf.size) for leaf in jax.tree_util.tree_leaves(params))


def self_attention(
    hidden: Array,
    coordinates: Array,
    spatial_mask: Array,
    rope_groups: Array,
    token_mask: Array,
    params: Params,
    config: JaxModelConfig,
    dtype: jnp.dtype,
    dropout_key: Array | None,
    training: bool,
) -> Array:
    batch, tokens, _ = hidden.shape
    head_dim = config.d_model // config.heads
    with jax.named_scope("attention_qkv"):
        qkv = dense(hidden, params["qkv"], dtype).reshape(batch, tokens, 3, config.heads, head_dim)
    query, key, value = jnp.moveaxis(qkv, 2, 0)
    query = jnp.transpose(query, (0, 2, 1, 3))
    key = jnp.transpose(key, (0, 2, 1, 3))
    value = jnp.transpose(value, (0, 2, 1, 3))
    raw_query, raw_key = query, key
    with jax.named_scope("attention_rope"):
        query, key = apply_2d_rope(query, key, coordinates, spatial_mask, config)

    scale = 1.0 / math.sqrt(head_dim)
    with jax.named_scope("attention_rope_correction_dense"):
        correction = dense_rope_correction(
            raw_query,
            raw_key,
            query,
            key,
            rope_groups,
            config.rope_dim,
            scale,
        )
    scores = jnp.einsum("bhtd,bhsd->bhts", query, key).astype(jnp.float32) * scale
    scores += correction
    scores = jnp.where(token_mask[:, None, None, :], scores, jnp.finfo(jnp.float32).min)
    attention = jax.nn.softmax(scores, axis=-1).astype(dtype)
    attention = apply_dropout(attention, dropout_key, config.dropout, training)
    attended = jnp.einsum("bhts,bhsd->bhtd", attention, value)
    attended = jnp.transpose(attended, (0, 2, 1, 3)).reshape(batch, tokens, config.d_model)
    output = dense(attended, params["attention_output"], dtype)
    return output * token_mask[..., None].astype(dtype)


def policy_forward(
    params: Params,
    batch: dict[str, Array],
    config: JaxModelConfig,
    dtype: jnp.dtype = jnp.float32,
    *,
    rng: Array | None = None,
    training: bool = False,
) -> dict[str, Array]:
    """The same feature/action contract is used for BC and actual inference."""
    partitioned = config.rope_correction_backend == "partitioned"
    features = batch["features"]
    memory = packed_memory_features(features) if partitioned else batch["memory_features"]
    hidden = layer_norm(typed_input_projection(params, features, memory, dtype), params["input_norm"], dtype)
    if partitioned:
        coordinates, spatial_mask, rope_groups, token_mask = metadata_from_features(features)
    else:
        coordinates, spatial_mask = batch["coordinates"], batch["spatial_mask"]
        rope_groups, token_mask = batch["rope_groups"], batch["token_mask"]
    hidden *= token_mask[..., None].astype(dtype)
    if partitioned:
        hidden, coordinates, spatial_mask, rope_groups, token_mask, unit_indices, market_indices, global_index = (
            partition_policy_tokens(hidden, coordinates, spatial_mask, rope_groups, token_mask)
        )
    else:
        unit_indices, market_indices, global_index = batch["unit_indices"], batch["market_indices"], 0
    if training and config.dropout:
        if rng is None:
            raise ValueError("training with dropout requires a PRNG key")
        keys = jax.random.split(rng, config.layers * 3)
    else:
        keys = None
    for index, block in enumerate(params["blocks"]):
        attention_key, hidden_key, output_key = (keys[index * 3:index * 3 + 3] if keys is not None
                                                else (None, None, None))
        hidden += self_attention(layer_norm(hidden, block["attention_norm"], dtype), coordinates,
                                 spatial_mask, rope_groups, token_mask, block, config, dtype,
                                 attention_key, training)
        normalized = layer_norm(hidden, block["ffn_norm"], dtype)
        # Retain the numerical path used by the public BC implementation.
        feed_forward = jax.nn.gelu(dense(normalized, block["ffn_input"], dtype), approximate="none")
        feed_forward = apply_dropout(feed_forward, hidden_key, config.dropout, training)
        feed_forward = dense(feed_forward, block["ffn_output"], dtype)
        feed_forward = apply_dropout(feed_forward, output_key, config.dropout, training)
        hidden += feed_forward * token_mask[..., None].astype(dtype)
    hidden = layer_norm(hidden, params["final_norm"], dtype)
    units = jnp.take_along_axis(hidden, unit_indices[..., None], axis=1)
    market = jnp.take_along_axis(hidden, market_indices[..., None], axis=1)
    quantity = policy_head(market, params["sell_quantity"], dtype).reshape(
        (*market.shape[:-1], PRODUCT_COUNT, QUANTITY_COUNT)
    )
    outputs = {
        "unit_action": policy_head(units, params["unit_action"], dtype),
        "market_action": compose_market_logits(policy_head(market, params["market_action"], dtype), quantity),
    }
    if "value" in params:
        value_hidden = jax.nn.gelu(dense(hidden[:, global_index], params["value"]["hidden"], dtype)
                                   .astype(jnp.float32), approximate=False).astype(dtype)
        value = dense(value_hidden, params["value"]["output"], dtype).astype(jnp.float32)[..., 0]
        if "linear_cost" in params["value"]:
            global_features = features[:, 0].astype(jnp.float32)
            scale = jnp.log1p(jnp.asarray(VALUE_MONEY_ENCODING_REFERENCE, jnp.float32))
            own_money = jnp.expm1(global_features[:, FEATURE_INDEX["self_money"]] * scale)
            other_money = jnp.expm1(global_features[:, FEATURE_INDEX["opponent_money"]] * scale)
            difference = batch.get("value_cost_difference", (own_money - other_money) / VALUE_COST_DIFFERENCE_REFERENCE)
            time = batch.get("value_time", global_features[:, FEATURE_INDEX["season_progress"]])
            w0, w1 = params["value"]["linear_cost"]
            value += (w0 + w1 * time) * difference
        outputs["value"] = value
    return outputs

