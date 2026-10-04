//! pyo3 bindings: `RustEnv`, a drop-in replacement for FastEnv (fast_env.py).
//!
//! Observations are freshly built Python dicts each call (the parity harness
//! only compares values, so live-reference aliasing like FastEnv's is not
//! reproduced). Scalar Python types are preserved exactly: money is float,
//! counts are int, flags are bool — the harness compares by type.

use numpy::ndarray::{Array2, Array3};
use numpy::{
    IntoPyArray, PyArray2, PyArray3, PyReadonlyArray1, PyReadonlyArray2, PyReadonlyArray3,
    PyReadwriteArray2, PyReadwriteArray3, PyUntypedArrayMethods,
};
use pyo3::exceptions::{PyRuntimeError, PyValueError};
use pyo3::intern;
use pyo3::prelude::*;
use pyo3::types::{PyDict, PyList, PyString};
use rayon::prelude::*;

use crate::catalog::{
    FixedMarketHistory, MARKET_SLOTS as ACTION_MARKET_SLOTS, MAX_OWN_UNITS as ACTION_MAX_OWN_UNITS,
    decode_fixed_unit_labels, decode_player_action_ids, encode_fixed_market_history,
    sell_stocks_before_slots,
};
use crate::engine::Engine;
use crate::exp28_features::{
    FEATURE_DIM as EXP28_FEATURE_DIM, FIXED_TOKENS as EXP28_FIXED_TOKENS,
    MEMORY_DIM as EXP28_MEMORY_DIM, encode_features_into as encode_exp28_features_into,
};
use crate::features::{
    FEATURE_DIM, FIXED_TOKENS, FeatureCounts, MARKET_SLOTS, MAX_OWN_UNITS, MEMORY_DIM,
    encode_fixed, encode_partitioned_features_into,
};
use crate::fixed_tape_features::{
    GLOBAL_DIM as TAPE_GLOBAL_DIM, MARKET_LABEL_SLOTS, RESOURCE_DIM as TAPE_RESOURCE_DIM,
    RESOURCE_TOKENS as TAPE_RESOURCE_TOKENS, SHOP_DIM as TAPE_SHOP_DIM,
    SHOP_TOKENS as TAPE_SHOP_TOKENS, TILE_DIM as TAPE_TILE_DIM, TILE_TOKENS as TAPE_TILE_TOKENS,
    TapeFeatureOutput, TapeFeatures, TapeHistory, UNIT_DIM as TAPE_UNIT_DIM,
    UNIT_TOKENS as TAPE_UNIT_TOKENS, encode_fixed_tape, encode_fixed_tape_into,
};
use crate::legal_masks::{MARKET_MASK_SIZE, UNIT_MASK_SIZE, write_action_masks_into};
use crate::state::*;
use crate::tracker::{OpponentTracker, infer_all_public_flows};

type PartitionedBatchOutput = (Py<PyArray3<f32>>, Vec<usize>, Vec<usize>);
type MixedStepOutput = (Vec<bool>, Py<PyArray2<i32>>, Vec<i32>, Vec<i32>);

// ---------------------------------------------------------------------------
// Python -> engine conversion
// ---------------------------------------------------------------------------

/// Python `int(x)` over config values (int/bool, float truncation, numeric str).
fn py_int(v: &Bound<'_, PyAny>) -> PyResult<i64> {
    if let Ok(i) = v.extract::<i64>() {
        return Ok(i);
    }
    if let Ok(f) = v.extract::<f64>() {
        return Ok(f.trunc() as i64);
    }
    if let Ok(s) = v.extract::<String>()
        && let Ok(i) = s.trim().parse::<i64>()
    {
        return Ok(i);
    }
    Err(PyValueError::new_err(format!(
        "expected an int-like config value, got {v:?}"
    )))
}

fn py_float(v: &Bound<'_, PyAny>) -> PyResult<f64> {
    if let Ok(f) = v.extract::<f64>() {
        return Ok(f);
    }
    if let Ok(s) = v.extract::<String>()
        && let Ok(f) = s.trim().parse::<f64>()
    {
        return Ok(f);
    }
    Err(PyValueError::new_err(format!(
        "expected a float-like config value, got {v:?}"
    )))
}

/// int stays Int, float stays Float — the obs must echo the exact Python type.
fn py_num(v: &Bound<'_, PyAny>) -> PyResult<Num> {
    if let Ok(i) = v.extract::<i64>() {
        return Ok(Num::Int(i));
    }
    if let Ok(f) = v.extract::<f64>() {
        return Ok(Num::Float(f));
    }
    Err(PyValueError::new_err(format!(
        "unsupported market param override value: {v:?}"
    )))
}

fn parse_market_overrides(v: &Bound<'_, PyAny>) -> PyResult<Option<[MarketParam; N_PRODUCTS]>> {
    if v.is_none() {
        return Ok(None);
    }
    let d = v
        .downcast::<PyDict>()
        .map_err(|_| PyValueError::new_err("marketParams must be a dict or None"))?;
    // The reference emits obs "params" whenever the overrides dict is truthy,
    // even if no patch matches a known product.
    if d.is_empty() {
        return Ok(None);
    }
    let mut params = default_market_params();
    for (key, patch) in d.iter() {
        let Ok(name) = key.extract::<String>() else {
            continue;
        };
        let Some(pid) = item_id(&name).filter(|&i| i < N_PRODUCTS) else {
            continue;
        };
        let Ok(pd) = patch.downcast::<PyDict>() else {
            continue;
        };
        let p = &mut params[pid];
        if let Some(x) = pd.get_item(intern!(v.py(), "base"))? {
            p.base = py_num(&x)?;
        }
        if let Some(x) = pd.get_item(intern!(v.py(), "I0"))? {
            p.i0 = py_num(&x)?;
        }
        if let Some(x) = pd.get_item(intern!(v.py(), "T"))? {
            p.t = py_num(&x)?;
        }
        if let Some(x) = pd.get_item(intern!(v.py(), "below_func"))? {
            p.below_func = Func::parse(&x.extract::<String>()?);
        }
        if let Some(x) = pd.get_item(intern!(v.py(), "below_target"))? {
            p.below_target = py_num(&x)?;
        }
        if let Some(x) = pd.get_item(intern!(v.py(), "above_func"))? {
            p.above_func = Func::parse(&x.extract::<String>()?);
        }
        if let Some(x) = pd.get_item(intern!(v.py(), "above_target"))? {
            p.above_target = py_num(&x)?;
        }
    }
    Ok(Some(params))
}

fn parse_config(overrides: Option<&Bound<'_, PyDict>>) -> PyResult<Config> {
    let mut cfg = Config::default();
    let Some(d) = overrides else { return Ok(cfg) };
    if let Some(v) = d.get_item("episodeSteps")? {
        cfg.episode_steps = py_int(&v)?;
    }
    if let Some(v) = d.get_item("boardSize")? {
        cfg.board_size = py_int(&v)?;
    }
    if let Some(v) = d.get_item("startingMoney")? {
        cfg.starting_money = py_int(&v)?;
    }
    if let Some(v) = d.get_item("maxMarketOrdersPerTurn")? {
        cfg.max_orders = py_int(&v)?.max(1) as usize;
    }
    if let Some(v) = d.get_item("turnsPerDay")? {
        cfg.turns_per_day = py_int(&v)?;
    }
    if let Some(v) = d.get_item("shedCapacity")? {
        cfg.shed_capacity = py_int(&v)?;
    }
    if let Some(v) = d.get_item("weedSpawnChance")? {
        cfg.weed_chance = py_float(&v)?;
    }
    if let Some(v) = d.get_item("townShopUnlockInterval")? {
        cfg.shop_unlock_interval = py_int(&v)?;
    }
    if let Some(v) = d.get_item("townShopSellInterval")? {
        cfg.shop_sell_interval = py_int(&v)?;
    }
    if let Some(v) = d.get_item("townCenterSellInterval")? {
        cfg.center_sell_interval = py_int(&v)?;
    }
    if let Some(v) = d.get_item("farmHandCostMult")? {
        cfg.hire_mult = py_int(&v)?;
    }
    if let Some(v) = d.get_item("marketParams")? {
        cfg.market_overrides = parse_market_overrides(&v)?;
    }
    Ok(cfg)
}

fn to_token(v: &Bound<'_, PyAny>) -> Token {
    if let Ok(s) = v.downcast::<PyString>() {
        return match s.extract::<String>() {
            Ok(s) => Token::Str(s),
            Err(_) => Token::Other,
        };
    }
    if let Ok(i) = v.extract::<i64>() {
        return Token::Int(i);
    }
    if let Ok(f) = v.extract::<f64>() {
        return Token::Float(f);
    }
    Token::Other
}

/// Mirrors `isinstance(x, list)`: only real lists parse, anything else no-ops.
fn to_token_list(v: &Bound<'_, PyAny>) -> TokenList {
    let list = v.downcast::<PyList>().ok()?;
    Some(
        list.iter()
            .map(|item| to_token(&item))
            .collect::<Vec<_>>()
            .into(),
    )
}

fn to_player_action(v: &Bound<'_, PyAny>) -> PyResult<PlayerAction> {
    // Framework schema validation is collapsed to: non-dict action -> all-PASS.
    let Ok(d) = v.downcast::<PyDict>() else {
        return Ok(PlayerAction::empty());
    };
    let farmer = match d.get_item(intern!(v.py(), "farmer"))? {
        Some(f) => to_token_list(&f),
        None => Some(vec![Token::Str("PASS".to_string())].into()),
    };
    let hands = match d.get_item(intern!(v.py(), "hands"))? {
        Some(h) => match h.downcast::<PyList>() {
            Ok(list) => list.iter().map(|a| to_token_list(&a)).collect(),
            Err(_) => Vec::new(),
        },
        None => Vec::new(),
    };
    let market = match d.get_item(intern!(v.py(), "market"))? {
        Some(m) => match m.downcast::<PyList>() {
            Ok(list) => list.iter().map(|o| to_token_list(&o)).collect(),
            Err(_) => Vec::new(),
        },
        None => Vec::new(),
    };
    Ok(PlayerAction {
        farmer,
        hands,
        market,
    })
}

// ---------------------------------------------------------------------------
// Engine -> Python observation
// ---------------------------------------------------------------------------

fn num_to_py(py: Python<'_>, n: Num) -> Py<PyAny> {
    match n {
        Num::Int(i) => i.into_pyobject(py).unwrap().into_any().unbind(),
        Num::Float(f) => f.into_pyobject(py).unwrap().into_any().unbind(),
    }
}

fn tile_to_py(py: Python<'_>, tile: &Tile) -> PyResult<Py<PyAny>> {
    Ok(match tile {
        Tile::Empty => py.None(),
        Tile::Locked => intern!(py, "LOCKED").clone().into_any().unbind(),
        Tile::Weed => {
            let d = PyDict::new(py);
            d.set_item(intern!(py, "kind"), intern!(py, "WEED"))?;
            d.into_any().unbind()
        }
        Tile::Structure(s) => {
            let d = PyDict::new(py);
            d.set_item(intern!(py, "kind"), s.name())?;
            d.into_any().unbind()
        }
        Tile::Plant(p) => {
            let d = PyDict::new(py);
            d.set_item(intern!(py, "kind"), intern!(py, "PLANT"))?;
            d.set_item(intern!(py, "crop"), ITEM_NAMES[p.crop])?;
            d.set_item(intern!(py, "planted_day"), p.planted_day)?;
            d.set_item(intern!(py, "watered_today"), p.watered_today)?;
            d.set_item(
                intern!(py, "consecutive_unwatered"),
                p.consecutive_unwatered,
            )?;
            d.set_item(intern!(py, "yield_units"), p.yield_units)?;
            d.set_item(intern!(py, "max_lifespan_step"), p.max_lifespan_step)?;
            d.set_item(intern!(py, "fertilized_until_day"), p.fertilized_until_day)?;
            d.into_any().unbind()
        }
        Tile::Animal(a) => {
            let d = PyDict::new(py);
            d.set_item(intern!(py, "kind"), ANIMALS[a.animal].structure.name())?;
            d.set_item(intern!(py, "animal"), ITEM_NAMES[FIRST_ANIMAL + a.animal])?;
            d.set_item(intern!(py, "placed_day"), a.placed_day)?;
            d.set_item(intern!(py, "yield_units"), a.yield_units)?;
            d.set_item(intern!(py, "consecutive_unfed"), a.consecutive_unfed)?;
            d.set_item(intern!(py, "fed_today"), a.fed_today)?;
            d.set_item(intern!(py, "cared_today"), a.cared_today)?;
            d.set_item(intern!(py, "fertilizer_available"), a.fertilizer_available)?;
            d.set_item(intern!(py, "pending_care_bonus"), a.pending_care_bonus)?;
            d.into_any().unbind()
        }
    })
}

fn farm_to_py<'py>(py: Python<'py>, farm: &Farm) -> PyResult<Bound<'py, PyDict>> {
    let d = PyDict::new(py);
    d.set_item(intern!(py, "money"), farm.money)?;
    let tiles = PyList::empty(py);
    for row in &farm.tiles {
        let r = PyList::empty(py);
        for tile in row {
            r.append(tile_to_py(py, tile)?)?;
        }
        tiles.append(r)?;
    }
    d.set_item(intern!(py, "tiles"), tiles)?;
    d.set_item(intern!(py, "farmer"), vec![farm.farmer.0, farm.farmer.1])?;
    let hands: Vec<Vec<i64>> = farm.hands.iter().map(|(x, y)| vec![*x, *y]).collect();
    d.set_item(intern!(py, "hands"), hands)?;
    d.set_item(
        intern!(py, "unlocked_quadrants"),
        farm.unlocked_quadrants.clone(),
    )?;
    d.set_item(intern!(py, "hires_today"), farm.hires_today)?;
    Ok(d)
}

fn private_to_py<'py>(py: Python<'py>, private: &Private) -> PyResult<Bound<'py, PyDict>> {
    let d = PyDict::new(py);
    let shed = PyDict::new(py);
    for (item, count) in private.shed.iter().enumerate() {
        shed.set_item(ITEM_NAMES[item], *count)?;
    }
    d.set_item(intern!(py, "shed"), shed)?;
    let seeds = PyDict::new(py);
    for (crop, count) in private.seeds.iter().enumerate() {
        seeds.set_item(ITEM_NAMES[crop], *count)?;
    }
    d.set_item(intern!(py, "seeds"), seeds)?;
    let inventories = PyList::empty(py);
    for inv in &private.inventories {
        let i = PyDict::new(py);
        for (item, n) in &inv.0 {
            i.set_item(ITEM_NAMES[*item], *n)?;
        }
        inventories.append(i)?;
    }
    d.set_item(intern!(py, "inventories"), inventories)?;
    Ok(d)
}

fn market_to_py<'py>(py: Python<'py>, market: &Market) -> PyResult<Bound<'py, PyDict>> {
    let d = PyDict::new(py);
    let inventory = PyDict::new(py);
    let prices = PyDict::new(py);
    for (item, name) in ITEM_NAMES.iter().take(N_PRODUCTS).enumerate() {
        inventory.set_item(name, market.inventory[item])?;
        prices.set_item(name, num_to_py(py, market.prices[item]))?;
    }
    d.set_item(intern!(py, "inventory"), inventory)?;
    d.set_item(intern!(py, "prices"), prices)?;
    if market.params_overridden {
        let params = PyDict::new(py);
        for (item, p) in market.params.iter().enumerate() {
            let pd = PyDict::new(py);
            pd.set_item(intern!(py, "base"), num_to_py(py, p.base))?;
            pd.set_item(intern!(py, "I0"), num_to_py(py, p.i0))?;
            pd.set_item(intern!(py, "T"), num_to_py(py, p.t))?;
            pd.set_item(intern!(py, "below_func"), p.below_func.as_str())?;
            pd.set_item(intern!(py, "below_target"), num_to_py(py, p.below_target))?;
            pd.set_item(intern!(py, "above_func"), p.above_func.as_str())?;
            pd.set_item(intern!(py, "above_target"), num_to_py(py, p.above_target))?;
            params.set_item(ITEM_NAMES[item], pd)?;
        }
        d.set_item(intern!(py, "params"), params)?;
    }
    Ok(d)
}

type PublicObservationParts<'py> = (
    Bound<'py, PyList>,
    Bound<'py, PyDict>,
    Bound<'py, PyDict>,
    i64,
    i64,
);

fn public_observation_parts<'py>(
    py: Python<'py>,
    engine: &Engine,
) -> PyResult<PublicObservationParts<'py>> {
    let farms = PyList::empty(py);
    for farm in &engine.farms {
        farms.append(farm_to_py(py, farm)?)?;
    }
    let market = market_to_py(py, &engine.market)?;
    let town = PyDict::new(py);
    let shops: Vec<&str> = engine
        .town
        .unlocked_shops
        .iter()
        .map(|&s| SHOP_NAMES[s])
        .collect();
    town.set_item(intern!(py, "unlocked_shops"), shops)?;
    let (day, hour) = engine.day_hour();
    Ok((farms, market, town, day, hour))
}

#[allow(clippy::too_many_arguments)]
fn player_observation_from_parts<'py>(
    py: Python<'py>,
    engine: &Engine,
    player: usize,
    farms: &Bound<'py, PyList>,
    market: &Bound<'py, PyDict>,
    town: &Bound<'py, PyDict>,
    day: i64,
    hour: i64,
) -> PyResult<Bound<'py, PyDict>> {
    let observation = PyDict::new(py);
    observation.set_item(intern!(py, "remainingOverageTime"), REMAINING_OVERAGE_TIME)?;
    observation.set_item(intern!(py, "step"), engine.step_no)?;
    observation.set_item(intern!(py, "player"), player)?;
    observation.set_item(intern!(py, "farms"), farms)?;
    observation.set_item(
        intern!(py, "private"),
        private_to_py(py, &engine.privates[player])?,
    )?;
    observation.set_item(intern!(py, "market"), market)?;
    observation.set_item(intern!(py, "town"), town)?;
    observation.set_item(intern!(py, "day"), day)?;
    observation.set_item(intern!(py, "hour"), hour)?;
    Ok(observation)
}

fn build_player_obs(py: Python<'_>, engine: &Engine, player: usize) -> PyResult<Py<PyDict>> {
    let (farms, market, town, day, hour) = public_observation_parts(py, engine)?;
    Ok(
        player_observation_from_parts(py, engine, player, &farms, &market, &town, day, hour)?
            .unbind(),
    )
}

fn build_obs(py: Python<'_>, engine: &Engine) -> PyResult<Py<PyList>> {
    let (farms, market, town, day, hour) = public_observation_parts(py, engine)?;

    let out = PyList::empty(py);
    for player in 0..2 {
        out.append(player_observation_from_parts(
            py, engine, player, &farms, &market, &town, day, hour,
        )?)?;
    }
    Ok(out.unbind())
}

// ---------------------------------------------------------------------------
// RustEnv
// ---------------------------------------------------------------------------

/// Fast replica of the kaggriculture interpreter for 2 players.
///
/// Usage: env = RustEnv(seed); obs = env.reset(seed); obs, done = env.step(actions).
#[pyclass]
pub struct RustEnv {
    engine: Engine,
    trackers: [OpponentTracker; 2],
}

impl RustEnv {
    fn advance_parsed(&mut self, parsed: &[PlayerAction; 2]) -> Result<bool, String> {
        let before = self.engine.clone();
        let done = self.engine.step(parsed)?;
        let public_flows = infer_all_public_flows(&before, &self.engine);
        for (observer, own_action) in parsed.iter().enumerate() {
            self.trackers[observer].advance(&before, &self.engine, own_action, &public_flows);
        }
        Ok(done)
    }
}

/// Owns an evaluation batch so Rust can parallelize without repeated PyRef traffic.
#[pyclass]
struct RustBatchEnv {
    environments: Vec<RustEnv>,
}

fn write_batch_action_masks(
    py: Python<'_>,
    environments: &[RustEnv],
    players: Option<PyReadonlyArray1<'_, i32>>,
    mut unit_masks: PyReadwriteArray3<'_, bool>,
    mut market_masks: PyReadwriteArray2<'_, bool>,
    parallel: bool,
) -> PyResult<()> {
    let games = environments.len();
    let seats = if players.is_some() { 1 } else { 2 };
    let players = players
        .as_ref()
        .map(|values| {
            if values.shape() != [games] {
                return Err(PyValueError::new_err(
                    "players must have one entry per environment",
                ));
            }
            let values = values
                .as_slice()
                .map_err(|_| PyValueError::new_err("players must be C-contiguous"))?;
            if values.iter().any(|&player| !(0..2).contains(&player)) {
                return Err(PyValueError::new_err("player must be 0 or 1"));
            }
            Ok(values)
        })
        .transpose()?;
    if unit_masks.shape()
        != [
            games * seats,
            ACTION_MAX_OWN_UNITS,
            UNIT_MASK_SIZE / ACTION_MAX_OWN_UNITS,
        ]
        || market_masks.shape() != [games * seats, MARKET_MASK_SIZE]
    {
        return Err(PyValueError::new_err(
            "mask outputs must have shapes [rows,20,500] and [rows,1075]",
        ));
    }
    let mut units_view = unit_masks.as_array_mut();
    let units = units_view
        .as_slice_mut()
        .ok_or_else(|| PyValueError::new_err("unit masks must be C-contiguous"))?;
    let mut market_view = market_masks.as_array_mut();
    let market = market_view
        .as_slice_mut()
        .ok_or_else(|| PyValueError::new_err("market masks must be C-contiguous"))?;
    // Buffers stay borrowed for the complete operation; no Python objects cross Rayon.
    py.allow_threads(|| {
        let write_game = |(game, ((units, market), environment)): (
            usize,
            ((&mut [bool], &mut [bool]), &RustEnv),
        )| {
            for seat in 0..seats {
                let player = players.map_or(seat, |values| values[game] as usize);
                write_action_masks_into(
                    &environment.engine,
                    player,
                    &mut units[seat * UNIT_MASK_SIZE..(seat + 1) * UNIT_MASK_SIZE],
                    &mut market[seat * MARKET_MASK_SIZE..(seat + 1) * MARKET_MASK_SIZE],
                )?;
            }
            Ok::<(), &'static str>(())
        };
        // The small mask predicate workload can cost less than Rayon dispatch.
        // Keep this choice independent of the feature encoder's thread pool.
        if parallel {
            units
                .par_chunks_mut(seats * UNIT_MASK_SIZE)
                .zip(market.par_chunks_mut(seats * MARKET_MASK_SIZE))
                .zip(environments.par_iter())
                .enumerate()
                .try_for_each(write_game)
        } else {
            units
                .chunks_mut(seats * UNIT_MASK_SIZE)
                .zip(market.chunks_mut(seats * MARKET_MASK_SIZE))
                .zip(environments.iter())
                .enumerate()
                .try_for_each(write_game)
        }
    })
    .map_err(PyValueError::new_err)
}

#[pymethods]
impl RustBatchEnv {
    fn owned_prefix(&self, game: usize, player: usize) -> PyResult<NativePrefixResolver> {
        if player > 1 || game >= self.environments.len() { return Err(PyValueError::new_err("invalid game or player")); }
        Ok(NativePrefixResolver { state: crate::owned_prefix::PrefixState::new(&self.environments[game].engine, player).map_err(PyValueError::new_err)? })
    }

    /// Execute exact owned catalog requests; preserve NOOP slots and absolute SELL amounts.
    fn step_owned_ids(&mut self, py: Python<'_>, units: PyReadonlyArray2<'_, i32>, markets: PyReadonlyArray2<'_, i32>) -> PyResult<Vec<bool>> {
        let games = self.environments.len();
        if units.shape() != [games*2, 20] || markets.shape() != [games*2, 10] { return Err(PyValueError::new_err("owned action ids require both seats")); }
        let units = units.as_slice()?;
        let markets = markets.as_slice()?;
        py.allow_threads(|| self.environments.par_iter_mut().enumerate().map(|(game, environment)| {
            let actions = [
                crate::owned_prefix::exact_action(&environment.engine, 0, &units[game*40..game*40+20], &markets[game*20..game*20+10])?,
                crate::owned_prefix::exact_action(&environment.engine, 1, &units[game*40+20..game*40+40], &markets[game*20+10..game*20+20])?,
            ];
            environment.advance_parsed(&actions)
        }).collect::<Result<Vec<_>, String>>()).map_err(PyRuntimeError::new_err)
    }

    /// External controllers can emit requests outside the neural catalog.
    fn step_owned_actions(&mut self, py: Python<'_>, actions: &Bound<'_, PyList>) -> PyResult<Vec<bool>> {
        if actions.len() != self.environments.len()*2 { return Err(PyValueError::new_err("actions require both seats")); }
        let parsed = actions.iter().map(|a| to_player_action(&a)).collect::<PyResult<Vec<_>>>()?;
        py.allow_threads(|| self.environments.par_iter_mut().enumerate().map(|(game, environment)|
            environment.advance_parsed(&[parsed[game*2].clone(), parsed[game*2+1].clone()])
        ).collect::<Result<Vec<_>, String>>()).map_err(PyRuntimeError::new_err)
    }

    /// Encode both seats into caller-owned, reusable full-action buffers.
    #[allow(clippy::too_many_arguments)]
    fn write_owned_features(&self, py: Python<'_>, mut features: PyReadwriteArray3<'_, f32>,
        mut memory: PyReadwriteArray2<'_, f32>, mut coordinates: PyReadwriteArray3<'_, f32>,
        mut spatial: PyReadwriteArray2<'_, bool>, mut rope: PyReadwriteArray2<'_, i32>,
        mut token: PyReadwriteArray2<'_, bool>, mut units: PyReadwriteArray2<'_, i32>,
        mut markets: PyReadwriteArray2<'_, i32>) -> PyResult<Vec<usize>> {
        let rows = self.environments.len()*2;
        if features.shape() != [rows, FIXED_TOKENS, FEATURE_DIM] || memory.shape() != [rows, MEMORY_DIM]
            || coordinates.shape() != [rows, FIXED_TOKENS, 2] || spatial.shape() != [rows, FIXED_TOKENS]
            || rope.shape() != [rows, FIXED_TOKENS] || token.shape() != [rows, FIXED_TOKENS]
            || units.shape() != [rows, MAX_OWN_UNITS] || markets.shape() != [rows, MARKET_SLOTS] {
            return Err(PyValueError::new_err("owned fixed feature buffer shapes mismatch"));
        }
        let encoded = py.allow_threads(|| self.environments.par_iter().map(|environment| {
            let engine = &environment.engine;
            if 2 + engine.farms[0].hands.len() + engine.farms[1].hands.len() > 40 || engine.farms.iter().any(|f| 1+f.hands.len()>20) {
                return Err("PPO observation exceeds fixed unit capacity; truncation is forbidden");
            }
            Ok([encode_fixed(engine, 0, &environment.trackers[0].encoded_memory())?,
                encode_fixed(engine, 1, &environment.trackers[1].encoded_memory())?])
        }).collect::<Result<Vec<_>, &'static str>>()).map_err(PyValueError::new_err)?;
        let f = features.as_slice_mut()?; let m = memory.as_slice_mut()?; let c = coordinates.as_slice_mut()?;
        let s = spatial.as_slice_mut()?; let r = rope.as_slice_mut()?; let t = token.as_slice_mut()?;
        let u = units.as_slice_mut()?; let a = markets.as_slice_mut()?;
        let mut counts = Vec::with_capacity(rows);
        for (row, encoded) in encoded.into_iter().flatten().enumerate() {
            f[row*FIXED_TOKENS*FEATURE_DIM..(row+1)*FIXED_TOKENS*FEATURE_DIM].copy_from_slice(&encoded.features);
            m[row*MEMORY_DIM..(row+1)*MEMORY_DIM].copy_from_slice(&encoded.memory_features);
            c[row*FIXED_TOKENS*2..(row+1)*FIXED_TOKENS*2].copy_from_slice(&encoded.coordinates);
            s[row*FIXED_TOKENS..(row+1)*FIXED_TOKENS].copy_from_slice(&encoded.spatial_mask);
            r[row*FIXED_TOKENS..(row+1)*FIXED_TOKENS].copy_from_slice(&encoded.rope_groups);
            t[row*FIXED_TOKENS..(row+1)*FIXED_TOKENS].copy_from_slice(&encoded.token_mask);
            u[row*MAX_OWN_UNITS..(row+1)*MAX_OWN_UNITS].copy_from_slice(&encoded.unit_indices);
            a[row*MARKET_SLOTS..(row+1)*MARKET_SLOTS].copy_from_slice(&encoded.market_indices);
            counts.push(encoded.actual_own_units);
        }
        Ok(counts)
    }

    #[pyo3(signature = (units, market, unit_valid, market_valid, players=None))]
    fn patch_action_ids(
        &self, py: Python<'_>, mut units: PyReadwriteArray2<'_, i32>,
        mut market: PyReadwriteArray2<'_, i32>, mut unit_valid: PyReadwriteArray2<'_, bool>,
        mut market_valid: PyReadwriteArray2<'_, bool>, players: Option<PyReadonlyArray1<'_, i32>>,
    ) -> PyResult<()> {
        let seats = if players.is_some() { 1 } else { 2 };
        let rows = self.environments.len() * seats;
        if units.shape() != [rows, ACTION_MAX_OWN_UNITS] || unit_valid.shape() != units.shape()
            || market.shape() != [rows, ACTION_MARKET_SLOTS] || market_valid.shape() != market.shape() {
            return Err(PyValueError::new_err("patch actions require [rows,20] units and [rows,10] market"));
        }
        let players = players.as_ref().map(|p| p.as_slice()).transpose()?;
        if players.is_some_and(|p| p.len() != self.environments.len() || p.iter().any(|&v| !(0..2).contains(&v))) {
            return Err(PyValueError::new_err("invalid patch player indices"));
        }
        let units = units.as_slice_mut()?;
        let market = market.as_slice_mut()?;
        let unit_valid = unit_valid.as_slice_mut()?;
        let market_valid = market_valid.as_slice_mut()?;
        py.allow_threads(|| {
            for row in 0..rows {
                let game = row / seats;
                let seat = players.map_or(row % seats, |p| p[game] as usize);
                let u = row * ACTION_MAX_OWN_UNITS;
                let m = row * ACTION_MARKET_SLOTS;
                crate::patch_sales::patch_action_ids(
                    &self.environments[game].engine, seat,
                    &mut units[u..u+ACTION_MAX_OWN_UNITS], &mut market[m..m+ACTION_MARKET_SLOTS],
                    &mut unit_valid[u..u+ACTION_MAX_OWN_UNITS], &mut market_valid[m..m+ACTION_MARKET_SLOTS],
                ).map_err(PyValueError::new_err)?;
            }
            Ok(())
        })
    }

    #[pyo3(signature = (masks, context, players=None))]
    fn write_patch_inputs(
        &self,
        py: Python<'_>,
        mut masks: PyReadwriteArray2<'_, bool>,
        mut context: PyReadwriteArray2<'_, i32>,
        players: Option<PyReadonlyArray1<'_, i32>>,
    ) -> PyResult<()> {
        use crate::patch_masks::{MASK_SIZE, CONTEXT_SIZE, write_patch_inputs};
        let seats = if players.is_some() { 1 } else { 2 };
        let rows = self.environments.len() * seats;
        if masks.shape() != [rows, MASK_SIZE] || context.shape() != [rows, CONTEXT_SIZE] {
            return Err(PyValueError::new_err("patch buffers must be [rows,902] and [rows,45]"));
        }
        let players = players.as_ref().map(|p| p.as_slice()).transpose()?;
        if players.is_some_and(|p| p.len() != self.environments.len() || p.iter().any(|&v| !(0..2).contains(&v))) {
            return Err(PyValueError::new_err("invalid patch player indices"));
        }
        let masks = masks.as_slice_mut()?;
        let context = context.as_slice_mut()?;
        py.allow_threads(|| {
            for (row, (mask, context)) in masks.chunks_exact_mut(MASK_SIZE).zip(context.chunks_exact_mut(CONTEXT_SIZE)).enumerate() {
                let game = row / seats;
                let seat = players.map_or(row % seats, |p| p[game] as usize);
                write_patch_inputs(&self.environments[game].engine, seat, mask, context).map_err(PyValueError::new_err)?;
            }
            Ok(())
        })
    }

    /// Snapshot-independent masks, ordered game -> seat, with one market row per seat.
    #[pyo3(signature = (unit_masks, market_masks, *, parallel=false))]
    fn write_action_masks_both(
        &self,
        py: Python<'_>,
        unit_masks: PyReadwriteArray3<'_, bool>,
        market_masks: PyReadwriteArray2<'_, bool>,
        parallel: bool,
    ) -> PyResult<()> {
        write_batch_action_masks(
            py,
            &self.environments,
            None,
            unit_masks,
            market_masks,
            parallel,
        )
    }

    /// One selected observer per game; masks never read the other player's private state.
    #[pyo3(signature = (players, unit_masks, market_masks, *, parallel=false))]
    fn write_action_masks_players(
        &self,
        py: Python<'_>,
        players: PyReadonlyArray1<'_, i32>,
        unit_masks: PyReadwriteArray3<'_, bool>,
        market_masks: PyReadwriteArray2<'_, bool>,
        parallel: bool,
    ) -> PyResult<()> {
        write_batch_action_masks(
            py,
            &self.environments,
            Some(players),
            unit_masks,
            market_masks,
            parallel,
        )
    }

    #[new]
    #[pyo3(signature = (seeds, config_overrides=None))]
    fn new(seeds: Vec<i64>, config_overrides: Option<&Bound<'_, PyDict>>) -> PyResult<Self> {
        let cfg = parse_config(config_overrides)?;
        let environments = seeds
            .into_iter()
            .map(|seed| RustEnv {
                engine: Engine::new(seed, cfg.clone()),
                trackers: [OpponentTracker::new(0), OpponentTracker::new(1)],
            })
            .collect();
        Ok(Self { environments })
    }

    fn observe(&self, py: Python<'_>) -> PyResult<Py<PyList>> {
        let observations = PyList::empty(py);
        for environment in &self.environments {
            observations.append(build_obs(py, &environment.engine)?)?;
        }
        Ok(observations.unbind())
    }

    fn observe_players(
        &self,
        py: Python<'_>,
        players: PyReadonlyArray1<'_, i32>,
    ) -> PyResult<Py<PyList>> {
        let games = self.environments.len();
        if players.shape() != [games] {
            return Err(PyValueError::new_err(
                "players must have one entry per environment",
            ));
        }
        let players = players
            .as_slice()
            .map_err(|_| PyValueError::new_err("players must be C-contiguous"))?;
        let observations = PyList::empty(py);
        for (environment, &player) in self.environments.iter().zip(players) {
            let player = usize::try_from(player)
                .ok()
                .filter(|player| *player < 2)
                .ok_or_else(|| PyValueError::new_err("player must be 0 or 1"))?;
            observations.append(build_player_obs(py, &environment.engine, player)?)?;
        }
        Ok(observations.unbind())
    }

    fn rewards(&self) -> Vec<[f64; 2]> {
        self.environments
            .iter()
            .map(|environment| environment.engine.rewards())
            .collect()
    }

    fn partitioned_features_players(
        &self,
        py: Python<'_>,
        players: PyReadonlyArray1<'_, i32>,
    ) -> PyResult<PartitionedBatchOutput> {
        let games = self.environments.len();
        if players.shape() != [games] {
            return Err(PyValueError::new_err(
                "players must have one entry per environment",
            ));
        }
        let players = players
            .as_slice()
            .map_err(|_| PyValueError::new_err("players must be C-contiguous"))?;
        let agent_stride = FIXED_TOKENS * FEATURE_DIM;
        let mut features = vec![0.0; games * agent_stride];
        let counts: Result<Vec<_>, _> = features
            .par_chunks_mut(agent_stride)
            .zip(self.environments.par_iter())
            .zip(players.par_iter())
            .map(|((output, environment), &player)| {
                let player = usize::try_from(player)
                    .ok()
                    .filter(|player| *player < 2)
                    .ok_or("player must be 0 or 1")?;
                encode_partitioned_features_into(
                    &environment.engine,
                    player,
                    &environment.trackers[player].encoded_memory(),
                    output,
                )
            })
            .collect();
        let counts = counts.map_err(PyValueError::new_err)?;
        let features = Array3::from_shape_vec((games, FIXED_TOKENS, FEATURE_DIM), features)
            .unwrap()
            .into_pyarray(py)
            .unbind();
        let actual_own_units = counts.iter().map(|count| count.actual_own_units).collect();
        let encoded_own_units = counts.iter().map(|count| count.encoded_own_units).collect();
        Ok((features, actual_own_units, encoded_own_units))
    }

    fn write_partitioned_features_players(
        &self,
        players: PyReadonlyArray1<'_, i32>,
        mut output: PyReadwriteArray3<'_, f32>,
    ) -> PyResult<(Vec<usize>, Vec<usize>)> {
        let games = self.environments.len();
        if players.shape() != [games] || output.shape() != [games, FIXED_TOKENS, FEATURE_DIM] {
            return Err(PyValueError::new_err(
                "players/output shape does not match the environment batch",
            ));
        }
        let players = players
            .as_slice()
            .map_err(|_| PyValueError::new_err("players must be C-contiguous"))?;
        let mut output_view = output.as_array_mut();
        let output = output_view
            .as_slice_mut()
            .ok_or_else(|| PyValueError::new_err("output must be C-contiguous"))?;
        let agent_stride = FIXED_TOKENS * FEATURE_DIM;
        let counts: Result<Vec<_>, _> = output
            .par_chunks_mut(agent_stride)
            .zip(self.environments.par_iter())
            .zip(players.par_iter())
            .map(|((features, environment), &player)| {
                let player = usize::try_from(player)
                    .ok()
                    .filter(|player| *player < 2)
                    .ok_or("player must be 0 or 1")?;
                encode_partitioned_features_into(
                    &environment.engine,
                    player,
                    &environment.trackers[player].encoded_memory(),
                    features,
                )
            })
            .collect();
        let counts = counts.map_err(PyValueError::new_err)?;
        Ok((
            counts.iter().map(|count| count.actual_own_units).collect(),
            counts.iter().map(|count| count.encoded_own_units).collect(),
        ))
    }

    fn write_partitioned_features_both(
        &self,
        mut output: PyReadwriteArray3<'_, f32>,
    ) -> PyResult<(Vec<usize>, Vec<usize>)> {
        let games = self.environments.len();
        if output.shape() != [games * 2, FIXED_TOKENS, FEATURE_DIM] {
            return Err(PyValueError::new_err(
                "output shape does not match both seats of the environment batch",
            ));
        }
        let mut output_view = output.as_array_mut();
        let output = output_view
            .as_slice_mut()
            .ok_or_else(|| PyValueError::new_err("output must be C-contiguous"))?;
        let agent_stride = FIXED_TOKENS * FEATURE_DIM;
        let game_stride = 2 * agent_stride;
        let counts: Result<Vec<_>, _> = output
            .par_chunks_mut(game_stride)
            .zip(self.environments.par_iter())
            .map(|(game_features, environment)| {
                let (first, second) = game_features.split_at_mut(agent_stride);
                Ok::<_, &'static str>([
                    encode_partitioned_features_into(
                        &environment.engine,
                        0,
                        &environment.trackers[0].encoded_memory(),
                        first,
                    )?,
                    encode_partitioned_features_into(
                        &environment.engine,
                        1,
                        &environment.trackers[1].encoded_memory(),
                        second,
                    )?,
                ])
            })
            .collect();
        let counts: Vec<_> = counts
            .map_err(PyValueError::new_err)?
            .into_iter()
            .flatten()
            .collect();
        Ok((
            counts.iter().map(|count| count.actual_own_units).collect(),
            counts.iter().map(|count| count.encoded_own_units).collect(),
        ))
    }

    fn write_exp28_features_players(
        &self,
        players: PyReadonlyArray1<'_, i32>,
        mut features: PyReadwriteArray3<'_, f32>,
        mut memory: PyReadwriteArray2<'_, f32>,
    ) -> PyResult<(Vec<usize>, Vec<usize>)> {
        let games = self.environments.len();
        if players.shape() != [games]
            || features.shape() != [games, EXP28_FIXED_TOKENS, EXP28_FEATURE_DIM]
            || memory.shape() != [games, EXP28_MEMORY_DIM]
        {
            return Err(PyValueError::new_err(
                "players/exp28 output shape does not match the environment batch",
            ));
        }
        let players = players
            .as_slice()
            .map_err(|_| PyValueError::new_err("players must be C-contiguous"))?;
        let mut feature_view = features.as_array_mut();
        let features = feature_view
            .as_slice_mut()
            .ok_or_else(|| PyValueError::new_err("exp28 features must be C-contiguous"))?;
        let mut memory_view = memory.as_array_mut();
        let memory = memory_view
            .as_slice_mut()
            .ok_or_else(|| PyValueError::new_err("exp28 memory must be C-contiguous"))?;
        let feature_stride = EXP28_FIXED_TOKENS * EXP28_FEATURE_DIM;
        let counts: Result<Vec<_>, &'static str> = features
            .par_chunks_mut(feature_stride)
            .zip(memory.par_chunks_mut(EXP28_MEMORY_DIM))
            .zip(self.environments.par_iter())
            .zip(players.par_iter())
            .map(|(((features, memory), environment), &player)| {
                let player = usize::try_from(player)
                    .ok()
                    .filter(|player| *player < 2)
                    .ok_or("player must be 0 or 1")?;
                let counts = encode_exp28_features_into(&environment.engine, player, features)?;
                memory.copy_from_slice(&environment.trackers[player].encoded_memory());
                Ok(counts)
            })
            .collect();
        let counts = counts.map_err(PyValueError::new_err)?;
        Ok((
            counts.iter().map(|count| count.actual_own_units).collect(),
            counts.iter().map(|count| count.encoded_own_units).collect(),
        ))
    }

    #[allow(clippy::too_many_arguments)]
    fn fixed_tape_features(
        &self,
        py: Python<'_>,
        players: PyReadonlyArray1<'_, i32>,
        previous_unit_types: PyReadonlyArray2<'_, i32>,
        previous_unit_counts: PyReadonlyArray1<'_, i32>,
        previous_market_qty: PyReadonlyArray2<'_, i32>,
        previous_hire: PyReadonlyArray1<'_, i32>,
        previous_buy_land: PyReadonlyArray1<'_, i32>,
        had_previous: PyReadonlyArray1<'_, bool>,
    ) -> PyResult<Py<PyDict>> {
        let games = self.environments.len();
        if players.shape() != [games]
            || previous_unit_types.shape() != [games, TAPE_UNIT_TOKENS]
            || previous_unit_counts.shape() != [games]
            || previous_market_qty.shape() != [games, MARKET_LABEL_SLOTS]
            || previous_hire.shape() != [games]
            || previous_buy_land.shape() != [games]
            || had_previous.shape() != [games]
        {
            return Err(PyValueError::new_err(
                "invalid fixed tape batch input shape",
            ));
        }
        let players = players
            .as_slice()
            .map_err(|_| PyValueError::new_err("players must be C-contiguous"))?;
        let unit_types = previous_unit_types
            .as_slice()
            .map_err(|_| PyValueError::new_err("previous Unit types must be C-contiguous"))?;
        let unit_counts = previous_unit_counts
            .as_slice()
            .map_err(|_| PyValueError::new_err("previous Unit counts must be C-contiguous"))?;
        let market_qty = previous_market_qty.as_slice().map_err(|_| {
            PyValueError::new_err("previous market quantities must be C-contiguous")
        })?;
        let hire = previous_hire
            .as_slice()
            .map_err(|_| PyValueError::new_err("previous hire must be C-contiguous"))?;
        let buy_land = previous_buy_land
            .as_slice()
            .map_err(|_| PyValueError::new_err("previous buy land must be C-contiguous"))?;
        let had_previous = had_previous
            .as_slice()
            .map_err(|_| PyValueError::new_err("previous flags must be C-contiguous"))?;
        let encoded: Result<Vec<_>, _> = self
            .environments
            .par_iter()
            .enumerate()
            .map(|(game, environment)| {
                let player = usize::try_from(players[game])
                    .ok()
                    .filter(|player| *player < 2)
                    .ok_or("player must be 0 or 1")?;
                let unit_start = game * TAPE_UNIT_TOKENS;
                let market_start = game * MARKET_LABEL_SLOTS;
                encode_fixed_tape(
                    &environment.engine,
                    player,
                    &TapeHistory {
                        unit_types: &unit_types[unit_start..unit_start + TAPE_UNIT_TOKENS],
                        unit_count: unit_counts[game],
                        market_qty: &market_qty[market_start..market_start + MARKET_LABEL_SLOTS],
                        hire: hire[game],
                        buy_land: buy_land[game],
                        available: had_previous[game],
                    },
                )
            })
            .collect();
        fixed_tape_batch_to_python(py, encoded.map_err(PyValueError::new_err)?)
    }

    #[allow(clippy::too_many_arguments)]
    fn write_fixed_tape_features(
        &self,
        players: PyReadonlyArray1<'_, i32>,
        previous_unit_types: PyReadonlyArray2<'_, i32>,
        previous_unit_counts: PyReadonlyArray1<'_, i32>,
        previous_market_qty: PyReadonlyArray2<'_, i32>,
        previous_hire: PyReadonlyArray1<'_, i32>,
        previous_buy_land: PyReadonlyArray1<'_, i32>,
        had_previous: PyReadonlyArray1<'_, bool>,
        mut global: PyReadwriteArray3<'_, f32>,
        mut resource: PyReadwriteArray3<'_, f32>,
        mut shop: PyReadwriteArray3<'_, f32>,
        mut tile: PyReadwriteArray3<'_, f32>,
        mut unit: PyReadwriteArray3<'_, f32>,
        mut unit_pad_mask: PyReadwriteArray2<'_, bool>,
    ) -> PyResult<()> {
        let games = self.environments.len();
        if players.shape() != [games]
            || previous_unit_types.shape() != [games, TAPE_UNIT_TOKENS]
            || previous_unit_counts.shape() != [games]
            || previous_market_qty.shape() != [games, MARKET_LABEL_SLOTS]
            || previous_hire.shape() != [games]
            || previous_buy_land.shape() != [games]
            || had_previous.shape() != [games]
            || global.shape() != [games, 1, TAPE_GLOBAL_DIM]
            || resource.shape() != [games, TAPE_RESOURCE_TOKENS, TAPE_RESOURCE_DIM]
            || shop.shape() != [games, TAPE_SHOP_TOKENS, TAPE_SHOP_DIM]
            || tile.shape() != [games, TAPE_TILE_TOKENS, TAPE_TILE_DIM]
            || unit.shape() != [games, TAPE_UNIT_TOKENS, TAPE_UNIT_DIM]
            || unit_pad_mask.shape() != [games, TAPE_UNIT_TOKENS]
        {
            return Err(PyValueError::new_err(
                "invalid fixed tape input/output shape",
            ));
        }
        let players = players
            .as_slice()
            .map_err(|_| PyValueError::new_err("players must be C-contiguous"))?;
        let unit_types = previous_unit_types
            .as_slice()
            .map_err(|_| PyValueError::new_err("previous Unit types must be C-contiguous"))?;
        let unit_counts = previous_unit_counts
            .as_slice()
            .map_err(|_| PyValueError::new_err("previous Unit counts must be C-contiguous"))?;
        let market_qty = previous_market_qty.as_slice().map_err(|_| {
            PyValueError::new_err("previous market quantities must be C-contiguous")
        })?;
        let hire = previous_hire
            .as_slice()
            .map_err(|_| PyValueError::new_err("previous hire must be C-contiguous"))?;
        let buy_land = previous_buy_land
            .as_slice()
            .map_err(|_| PyValueError::new_err("previous buy land must be C-contiguous"))?;
        let had_previous = had_previous
            .as_slice()
            .map_err(|_| PyValueError::new_err("previous flags must be C-contiguous"))?;
        let mut global_view = global.as_array_mut();
        let mut resource_view = resource.as_array_mut();
        let mut shop_view = shop.as_array_mut();
        let mut tile_view = tile.as_array_mut();
        let mut unit_view = unit.as_array_mut();
        let mut pad_view = unit_pad_mask.as_array_mut();
        let global = global_view
            .as_slice_mut()
            .ok_or_else(|| PyValueError::new_err("global output must be C-contiguous"))?;
        let resource = resource_view
            .as_slice_mut()
            .ok_or_else(|| PyValueError::new_err("resource output must be C-contiguous"))?;
        let shop = shop_view
            .as_slice_mut()
            .ok_or_else(|| PyValueError::new_err("shop output must be C-contiguous"))?;
        let tile = tile_view
            .as_slice_mut()
            .ok_or_else(|| PyValueError::new_err("tile output must be C-contiguous"))?;
        let unit = unit_view
            .as_slice_mut()
            .ok_or_else(|| PyValueError::new_err("Unit output must be C-contiguous"))?;
        let pad_mask = pad_view
            .as_slice_mut()
            .ok_or_else(|| PyValueError::new_err("Unit mask output must be C-contiguous"))?;
        let results: Result<Vec<_>, _> = global
            .par_chunks_mut(TAPE_GLOBAL_DIM)
            .zip(resource.par_chunks_mut(TAPE_RESOURCE_TOKENS * TAPE_RESOURCE_DIM))
            .zip(shop.par_chunks_mut(TAPE_SHOP_TOKENS * TAPE_SHOP_DIM))
            .zip(tile.par_chunks_mut(TAPE_TILE_TOKENS * TAPE_TILE_DIM))
            .zip(unit.par_chunks_mut(TAPE_UNIT_TOKENS * TAPE_UNIT_DIM))
            .zip(pad_mask.par_chunks_mut(TAPE_UNIT_TOKENS))
            .zip(self.environments.par_iter())
            .enumerate()
            .map(
                |(game, ((((((global, resource), shop), tile), unit), pad_mask), environment))| {
                    let player = usize::try_from(players[game])
                        .ok()
                        .filter(|player| *player < 2)
                        .ok_or("player must be 0 or 1")?;
                    let unit_start = game * TAPE_UNIT_TOKENS;
                    let market_start = game * MARKET_LABEL_SLOTS;
                    encode_fixed_tape_into(
                        &environment.engine,
                        player,
                        &TapeHistory {
                            unit_types: &unit_types[unit_start..unit_start + TAPE_UNIT_TOKENS],
                            unit_count: unit_counts[game],
                            market_qty: &market_qty
                                [market_start..market_start + MARKET_LABEL_SLOTS],
                            hire: hire[game],
                            buy_land: buy_land[game],
                            available: had_previous[game],
                        },
                        TapeFeatureOutput {
                            global,
                            resource,
                            shop,
                            tile,
                            unit,
                            unit_pad_mask: pad_mask,
                        },
                    )
                },
            )
            .collect();
        results.map_err(PyValueError::new_err)?;
        Ok(())
    }

    #[allow(clippy::too_many_arguments)]
    fn step_mixed(
        &mut self,
        py: Python<'_>,
        candidate_players: PyReadonlyArray1<'_, i32>,
        candidate_unit_ids: PyReadonlyArray2<'_, i32>,
        candidate_market_ids: PyReadonlyArray2<'_, i32>,
        fixed_unit_types: PyReadonlyArray2<'_, i32>,
        fixed_unit_crops: PyReadonlyArray2<'_, i32>,
        fixed_unit_items: PyReadonlyArray2<'_, i32>,
        fixed_unit_counts: PyReadonlyArray2<'_, i32>,
        fixed_market: &Bound<'_, PyAny>,
    ) -> PyResult<MixedStepOutput> {
        let games = self.environments.len();
        validate_mixed_shapes(
            games,
            &candidate_players,
            &candidate_unit_ids,
            &candidate_market_ids,
            &fixed_unit_types,
            &fixed_unit_crops,
            &fixed_unit_items,
            &fixed_unit_counts,
        )?;
        let players = contiguous_1d(&candidate_players, "candidate players")?;
        let candidate_units = contiguous_2d(&candidate_unit_ids, "candidate Unit ids")?;
        let candidate_market = contiguous_2d(&candidate_market_ids, "candidate market ids")?;
        let fixed_types = contiguous_2d(&fixed_unit_types, "fixed Unit types")?;
        let fixed_crops = contiguous_2d(&fixed_unit_crops, "fixed Unit crops")?;
        let fixed_items = contiguous_2d(&fixed_unit_items, "fixed Unit items")?;
        let fixed_counts = contiguous_2d(&fixed_unit_counts, "fixed Unit counts")?;
        let fixed_market = parse_market_list(fixed_market);
        let outcomes: Result<Vec<_>, String> = self
            .environments
            .par_iter_mut()
            .enumerate()
            .map(|(game, environment)| {
                mixed_step_one(
                    environment,
                    game,
                    players,
                    candidate_units,
                    candidate_market,
                    fixed_types,
                    fixed_crops,
                    fixed_items,
                    fixed_counts,
                    &fixed_market,
                )
            })
            .collect();
        mixed_outcomes_to_python(py, outcomes.map_err(PyRuntimeError::new_err)?)
    }

    #[allow(clippy::too_many_arguments)]
    fn step_ids_batch(
        &mut self,
        candidate_players: PyReadonlyArray1<'_, i32>,
        candidate_unit_ids: PyReadonlyArray2<'_, i32>,
        candidate_market_ids: PyReadonlyArray2<'_, i32>,
        opponent_unit_ids: PyReadonlyArray2<'_, i32>,
        opponent_market_ids: PyReadonlyArray2<'_, i32>,
    ) -> PyResult<Vec<bool>> {
        let games = self.environments.len();
        if candidate_players.shape() != [games]
            || candidate_unit_ids.shape() != [games, ACTION_MAX_OWN_UNITS]
            || candidate_market_ids.shape() != [games, ACTION_MARKET_SLOTS]
            || opponent_unit_ids.shape() != [games, ACTION_MAX_OWN_UNITS]
            || opponent_market_ids.shape() != [games, ACTION_MARKET_SLOTS]
        {
            return Err(PyValueError::new_err("invalid batched action-id shape"));
        }
        let players = contiguous_1d(&candidate_players, "candidate players")?;
        let candidate_units = contiguous_2d(&candidate_unit_ids, "candidate Unit ids")?;
        let candidate_market = contiguous_2d(&candidate_market_ids, "candidate market ids")?;
        let opponent_units = contiguous_2d(&opponent_unit_ids, "opponent Unit ids")?;
        let opponent_market = contiguous_2d(&opponent_market_ids, "opponent market ids")?;
        let outcomes: Result<Vec<_>, String> = self
            .environments
            .par_iter_mut()
            .enumerate()
            .map(|(game, environment)| {
                let candidate_player = usize::try_from(players[game])
                    .ok()
                    .filter(|player| *player < 2)
                    .ok_or_else(|| "player must be 0 or 1".to_string())?;
                let unit_start = game * ACTION_MAX_OWN_UNITS;
                let market_start = game * ACTION_MARKET_SLOTS;
                let candidate = decode_player_action_ids(
                    &environment.engine,
                    candidate_player,
                    &candidate_units[unit_start..unit_start + ACTION_MAX_OWN_UNITS],
                    &candidate_market[market_start..market_start + ACTION_MARKET_SLOTS],
                )
                .map_err(str::to_string)?;
                let opponent_player = 1 - candidate_player;
                let opponent = decode_player_action_ids(
                    &environment.engine,
                    opponent_player,
                    &opponent_units[unit_start..unit_start + ACTION_MAX_OWN_UNITS],
                    &opponent_market[market_start..market_start + ACTION_MARKET_SLOTS],
                )
                .map_err(str::to_string)?;
                let parsed = if candidate_player == 0 {
                    [candidate, opponent]
                } else {
                    [opponent, candidate]
                };
                environment.advance_parsed(&parsed)
            })
            .collect();
        outcomes.map_err(PyRuntimeError::new_err)
    }

    fn step_ids_python_batch(
        &mut self,
        candidate_players: PyReadonlyArray1<'_, i32>,
        candidate_unit_ids: PyReadonlyArray2<'_, i32>,
        candidate_market_ids: PyReadonlyArray2<'_, i32>,
        opponent_actions: &Bound<'_, PyList>,
    ) -> PyResult<Vec<bool>> {
        let games = self.environments.len();
        if candidate_players.shape() != [games]
            || candidate_unit_ids.shape() != [games, ACTION_MAX_OWN_UNITS]
            || candidate_market_ids.shape() != [games, ACTION_MARKET_SLOTS]
            || opponent_actions.len() != games
        {
            return Err(PyValueError::new_err(
                "invalid mixed action-id/Python action batch shape",
            ));
        }
        let players = contiguous_1d(&candidate_players, "candidate players")?;
        let candidate_units = contiguous_2d(&candidate_unit_ids, "candidate Unit ids")?;
        let candidate_market = contiguous_2d(&candidate_market_ids, "candidate market ids")?;
        let opponent_actions = opponent_actions
            .iter()
            .map(|action| to_player_action(&action))
            .collect::<PyResult<Vec<_>>>()?;
        let outcomes: Result<Vec<_>, String> = self
            .environments
            .par_iter_mut()
            .enumerate()
            .map(|(game, environment)| {
                let candidate_player = usize::try_from(players[game])
                    .ok()
                    .filter(|player| *player < 2)
                    .ok_or_else(|| "player must be 0 or 1".to_string())?;
                let unit_start = game * ACTION_MAX_OWN_UNITS;
                let market_start = game * ACTION_MARKET_SLOTS;
                let candidate = decode_player_action_ids(
                    &environment.engine,
                    candidate_player,
                    &candidate_units[unit_start..unit_start + ACTION_MAX_OWN_UNITS],
                    &candidate_market[market_start..market_start + ACTION_MARKET_SLOTS],
                )
                .map_err(str::to_string)?;
                let opponent = opponent_actions[game].clone();
                let parsed = if candidate_player == 0 {
                    [candidate, opponent]
                } else {
                    [opponent, candidate]
                };
                environment.advance_parsed(&parsed)
            })
            .collect();
        outcomes.map_err(PyRuntimeError::new_err)
    }

    fn write_sell_stocks_both(
        &self,
        unit_ids: PyReadonlyArray2<'_, i32>,
        market_ids: PyReadonlyArray2<'_, i32>,
        mut output: PyReadwriteArray3<'_, i32>,
    ) -> PyResult<()> {
        let rows = self.environments.len() * 2;
        if unit_ids.shape() != [rows, ACTION_MAX_OWN_UNITS]
            || market_ids.shape() != [rows, ACTION_MARKET_SLOTS]
            || output.shape() != [rows, ACTION_MARKET_SLOTS, N_PRODUCTS]
        {
            return Err(PyValueError::new_err(
                "SELL stock arrays require paired rows, 20 units, 10 slots and 9 products",
            ));
        }
        let units = contiguous_2d(&unit_ids, "SELL stock Unit ids")?;
        let market = contiguous_2d(&market_ids, "SELL stock market ids")?;
        let mut view = output.as_array_mut();
        let output = view
            .as_slice_mut()
            .ok_or_else(|| PyValueError::new_err("SELL stock output must be C-contiguous"))?;
        output
            .par_chunks_mut(ACTION_MARKET_SLOTS * N_PRODUCTS)
            .enumerate()
            .try_for_each(|(row, destination)| {
                sell_stocks_before_slots(
                    &self.environments[row / 2].engine,
                    row % 2,
                    &units[row * ACTION_MAX_OWN_UNITS..(row + 1) * ACTION_MAX_OWN_UNITS],
                    &market[row * ACTION_MARKET_SLOTS..(row + 1) * ACTION_MARKET_SLOTS],
                    destination,
                )
                .map_err(PyValueError::new_err)
            })
    }

    fn step_self_play_ids_batch(
        &mut self,
        unit_action_ids: PyReadonlyArray2<'_, i32>,
        market_action_ids: PyReadonlyArray2<'_, i32>,
    ) -> PyResult<Vec<bool>> {
        let games = self.environments.len();
        if unit_action_ids.shape() != [games * 2, ACTION_MAX_OWN_UNITS]
            || market_action_ids.shape() != [games * 2, ACTION_MARKET_SLOTS]
        {
            return Err(PyValueError::new_err(
                "self-play action ids must contain both seats of every game",
            ));
        }
        let unit_values = contiguous_2d(&unit_action_ids, "self-play Unit ids")?;
        let market_values = contiguous_2d(&market_action_ids, "self-play market ids")?;
        let outcomes: Result<Vec<_>, String> = self
            .environments
            .par_iter_mut()
            .enumerate()
            .map(|(game, environment)| {
                let unit_start = game * 2 * ACTION_MAX_OWN_UNITS;
                let market_start = game * 2 * ACTION_MARKET_SLOTS;
                let parsed = [
                    decode_player_action_ids(
                        &environment.engine,
                        0,
                        &unit_values[unit_start..unit_start + ACTION_MAX_OWN_UNITS],
                        &market_values[market_start..market_start + ACTION_MARKET_SLOTS],
                    )
                    .map_err(str::to_string)?,
                    decode_player_action_ids(
                        &environment.engine,
                        1,
                        &unit_values[unit_start + ACTION_MAX_OWN_UNITS
                            ..unit_start + 2 * ACTION_MAX_OWN_UNITS],
                        &market_values[market_start + ACTION_MARKET_SLOTS
                            ..market_start + 2 * ACTION_MARKET_SLOTS],
                    )
                    .map_err(str::to_string)?,
                ];
                environment.advance_parsed(&parsed)
            })
            .collect();
        outcomes.map_err(PyRuntimeError::new_err)
    }
}

#[pymethods]
impl RustEnv {
    /// Same game -> seat layout as RustBatchEnv; caller owns both reusable bool buffers.
    fn write_action_masks_both(
        &self,
        py: Python<'_>,
        unit_masks: PyReadwriteArray3<'_, bool>,
        market_masks: PyReadwriteArray2<'_, bool>,
    ) -> PyResult<()> {
        write_batch_action_masks(
            py,
            std::slice::from_ref(self),
            None,
            unit_masks,
            market_masks,
            false,
        )
    }

    #[new]
    #[pyo3(signature = (seed, config_overrides=None))]
    fn new(seed: i64, config_overrides: Option<&Bound<'_, PyDict>>) -> PyResult<Self> {
        let cfg = parse_config(config_overrides)?;
        Ok(RustEnv {
            engine: Engine::new(seed, cfg),
            trackers: [OpponentTracker::new(0), OpponentTracker::new(1)],
        })
    }

    fn reset(&mut self, py: Python<'_>, seed: i64) -> PyResult<Py<PyList>> {
        self.engine.reset(seed);
        self.trackers = [OpponentTracker::new(0), OpponentTracker::new(1)];
        build_obs(py, &self.engine)
    }

    fn step(&mut self, py: Python<'_>, actions: &Bound<'_, PyAny>) -> PyResult<(Py<PyList>, bool)> {
        let parsed = [
            to_player_action(&actions.get_item(0)?)?,
            to_player_action(&actions.get_item(1)?)?,
        ];
        let done = self
            .advance_parsed(&parsed)
            .map_err(PyRuntimeError::new_err)?;
        Ok((build_obs(py, &self.engine)?, done))
    }

    fn step_ids(
        &mut self,
        unit_action_ids: PyReadonlyArray2<'_, i32>,
        market_action_ids: PyReadonlyArray2<'_, i32>,
    ) -> PyResult<bool> {
        let unit_rows = unit_action_ids.as_array();
        let market_rows = market_action_ids.as_array();
        if unit_rows.shape() != [2, ACTION_MAX_OWN_UNITS]
            || market_rows.shape() != [2, ACTION_MARKET_SLOTS]
        {
            return Err(PyValueError::new_err(
                "step_ids requires unit ids [2,20] and market ids [2,10]",
            ));
        }
        let unit_values = unit_action_ids
            .as_slice()
            .map_err(|_| PyValueError::new_err("unit ids must be C-contiguous"))?;
        let market_values = market_action_ids
            .as_slice()
            .map_err(|_| PyValueError::new_err("market ids must be C-contiguous"))?;
        let parsed = [
            decode_player_action_ids(
                &self.engine,
                0,
                &unit_values[..ACTION_MAX_OWN_UNITS],
                &market_values[..ACTION_MARKET_SLOTS],
            )
            .map_err(PyValueError::new_err)?,
            decode_player_action_ids(
                &self.engine,
                1,
                &unit_values[ACTION_MAX_OWN_UNITS..],
                &market_values[ACTION_MARKET_SLOTS..],
            )
            .map_err(PyValueError::new_err)?,
        ];
        self.advance_parsed(&parsed)
            .map_err(PyRuntimeError::new_err)
    }

    fn observe(&self, py: Python<'_>) -> PyResult<Py<PyList>> {
        build_obs(py, &self.engine)
    }

    fn tracked_memory(&self, player: usize) -> PyResult<Vec<f32>> {
        let tracker = self
            .trackers
            .get(player)
            .ok_or_else(|| PyValueError::new_err("player must be 0 or 1"))?;
        Ok(tracker.encoded_memory().to_vec())
    }

    fn tracked_exp28_features(
        &self,
        py: Python<'_>,
        player: usize,
    ) -> PyResult<(Py<PyDict>, usize, usize)> {
        let tracker = self
            .trackers
            .get(player)
            .ok_or_else(|| PyValueError::new_err("player must be 0 or 1"))?;
        let mut features = vec![0.0; EXP28_FIXED_TOKENS * EXP28_FEATURE_DIM];
        let counts = encode_exp28_features_into(&self.engine, player, &mut features)
            .map_err(PyValueError::new_err)?;
        let arrays = PyDict::new(py);
        arrays.set_item(
            "features",
            Array3::from_shape_vec((1, EXP28_FIXED_TOKENS, EXP28_FEATURE_DIM), features)
                .unwrap()
                .into_pyarray(py),
        )?;
        arrays.set_item(
            "memory_features",
            Array2::from_shape_vec((1, EXP28_MEMORY_DIM), tracker.encoded_memory().to_vec())
                .unwrap()
                .into_pyarray(py),
        )?;
        Ok((
            arrays.unbind(),
            counts.actual_own_units,
            counts.encoded_own_units,
        ))
    }

    fn tracked_fixed_features(
        &self,
        py: Python<'_>,
        player: usize,
    ) -> PyResult<(Py<PyDict>, usize, usize)> {
        let memory = self
            .trackers
            .get(player)
            .ok_or_else(|| PyValueError::new_err("player must be 0 or 1"))?
            .encoded_memory();
        let encoded = encode_fixed(&self.engine, player, &memory).map_err(PyValueError::new_err)?;
        let arrays = PyDict::new(py);
        arrays.set_item(
            "features",
            Array3::from_shape_vec((1, FIXED_TOKENS, FEATURE_DIM), encoded.features)
                .unwrap()
                .into_pyarray(py),
        )?;
        arrays.set_item(
            "memory_features",
            Array2::from_shape_vec((1, MEMORY_DIM), encoded.memory_features)
                .unwrap()
                .into_pyarray(py),
        )?;
        arrays.set_item(
            "coordinates",
            Array3::from_shape_vec((1, FIXED_TOKENS, 2), encoded.coordinates)
                .unwrap()
                .into_pyarray(py),
        )?;
        arrays.set_item(
            "spatial_mask",
            Array2::from_shape_vec((1, FIXED_TOKENS), encoded.spatial_mask)
                .unwrap()
                .into_pyarray(py),
        )?;
        arrays.set_item(
            "rope_groups",
            Array2::from_shape_vec((1, FIXED_TOKENS), encoded.rope_groups)
                .unwrap()
                .into_pyarray(py),
        )?;
        arrays.set_item(
            "token_mask",
            Array2::from_shape_vec((1, FIXED_TOKENS), encoded.token_mask)
                .unwrap()
                .into_pyarray(py),
        )?;
        arrays.set_item(
            "unit_indices",
            Array2::from_shape_vec((1, MAX_OWN_UNITS), encoded.unit_indices)
                .unwrap()
                .into_pyarray(py),
        )?;
        arrays.set_item(
            "market_indices",
            Array2::from_shape_vec((1, MARKET_SLOTS), encoded.market_indices)
                .unwrap()
                .into_pyarray(py),
        )?;
        Ok((
            arrays.unbind(),
            encoded.actual_own_units,
            encoded.encoded_own_units,
        ))
    }

    fn fixed_features(
        &self,
        py: Python<'_>,
        player: usize,
        memory_features: Vec<f32>,
    ) -> PyResult<(Py<PyDict>, usize, usize)> {
        let encoded =
            encode_fixed(&self.engine, player, &memory_features).map_err(PyValueError::new_err)?;
        let arrays = PyDict::new(py);
        arrays.set_item(
            "features",
            Array3::from_shape_vec((1, FIXED_TOKENS, FEATURE_DIM), encoded.features)
                .unwrap()
                .into_pyarray(py),
        )?;
        arrays.set_item(
            "memory_features",
            Array2::from_shape_vec((1, MEMORY_DIM), encoded.memory_features)
                .unwrap()
                .into_pyarray(py),
        )?;
        arrays.set_item(
            "coordinates",
            Array3::from_shape_vec((1, FIXED_TOKENS, 2), encoded.coordinates)
                .unwrap()
                .into_pyarray(py),
        )?;
        arrays.set_item(
            "spatial_mask",
            Array2::from_shape_vec((1, FIXED_TOKENS), encoded.spatial_mask)
                .unwrap()
                .into_pyarray(py),
        )?;
        arrays.set_item(
            "rope_groups",
            Array2::from_shape_vec((1, FIXED_TOKENS), encoded.rope_groups)
                .unwrap()
                .into_pyarray(py),
        )?;
        arrays.set_item(
            "token_mask",
            Array2::from_shape_vec((1, FIXED_TOKENS), encoded.token_mask)
                .unwrap()
                .into_pyarray(py),
        )?;
        arrays.set_item(
            "unit_indices",
            Array2::from_shape_vec((1, MAX_OWN_UNITS), encoded.unit_indices)
                .unwrap()
                .into_pyarray(py),
        )?;
        arrays.set_item(
            "market_indices",
            Array2::from_shape_vec((1, MARKET_SLOTS), encoded.market_indices)
                .unwrap()
                .into_pyarray(py),
        )?;
        Ok((
            arrays.unbind(),
            encoded.actual_own_units,
            encoded.encoded_own_units,
        ))
    }

    #[getter]
    fn rewards(&self) -> [f64; 2] {
        self.engine.rewards()
    }
}

#[pyfunction]
fn step_ids_batch(
    environments: &Bound<'_, PyList>,
    unit_action_ids: PyReadonlyArray2<'_, i32>,
    market_action_ids: PyReadonlyArray2<'_, i32>,
) -> PyResult<Vec<bool>> {
    let unit_rows = unit_action_ids.as_array();
    let market_rows = market_action_ids.as_array();
    let agents = environments.len() * 2;
    if unit_rows.shape() != [agents, ACTION_MAX_OWN_UNITS]
        || market_rows.shape() != [agents, ACTION_MARKET_SLOTS]
    {
        return Err(PyValueError::new_err(format!(
            "step_ids_batch requires unit ids [{agents},20] and market ids [{agents},10]"
        )));
    }
    let unit_values = unit_action_ids
        .as_slice()
        .map_err(|_| PyValueError::new_err("unit ids must be C-contiguous"))?;
    let market_values = market_action_ids
        .as_slice()
        .map_err(|_| PyValueError::new_err("market ids must be C-contiguous"))?;

    let mut done = Vec::with_capacity(environments.len());
    for (game, item) in environments.iter().enumerate() {
        let mut environment = item.extract::<PyRefMut<'_, RustEnv>>()?;
        let first_unit = game * 2 * ACTION_MAX_OWN_UNITS;
        let first_market = game * 2 * ACTION_MARKET_SLOTS;
        let parsed = [
            decode_player_action_ids(
                &environment.engine,
                0,
                &unit_values[first_unit..first_unit + ACTION_MAX_OWN_UNITS],
                &market_values[first_market..first_market + ACTION_MARKET_SLOTS],
            )
            .map_err(PyValueError::new_err)?,
            decode_player_action_ids(
                &environment.engine,
                1,
                &unit_values
                    [first_unit + ACTION_MAX_OWN_UNITS..first_unit + 2 * ACTION_MAX_OWN_UNITS],
                &market_values
                    [first_market + ACTION_MARKET_SLOTS..first_market + 2 * ACTION_MARKET_SLOTS],
            )
            .map_err(PyValueError::new_err)?,
        ];
        done.push(
            environment
                .advance_parsed(&parsed)
                .map_err(PyRuntimeError::new_err)?,
        );
    }
    Ok(done)
}

#[allow(clippy::too_many_arguments)]
fn validate_mixed_shapes(
    games: usize,
    candidate_players: &PyReadonlyArray1<'_, i32>,
    candidate_unit_ids: &PyReadonlyArray2<'_, i32>,
    candidate_market_ids: &PyReadonlyArray2<'_, i32>,
    fixed_unit_types: &PyReadonlyArray2<'_, i32>,
    fixed_unit_crops: &PyReadonlyArray2<'_, i32>,
    fixed_unit_items: &PyReadonlyArray2<'_, i32>,
    fixed_unit_counts: &PyReadonlyArray2<'_, i32>,
) -> PyResult<()> {
    if candidate_players.shape() != [games]
        || candidate_unit_ids.shape() != [games, ACTION_MAX_OWN_UNITS]
        || candidate_market_ids.shape() != [games, ACTION_MARKET_SLOTS]
        || fixed_unit_types.shape() != [games, ACTION_MAX_OWN_UNITS]
        || fixed_unit_crops.shape() != [games, ACTION_MAX_OWN_UNITS]
        || fixed_unit_items.shape() != [games, ACTION_MAX_OWN_UNITS]
        || fixed_unit_counts.shape() != [games, ACTION_MAX_OWN_UNITS]
    {
        return Err(PyValueError::new_err(
            "invalid mixed-step batch input shape",
        ));
    }
    Ok(())
}

fn contiguous_1d<'a>(values: &'a PyReadonlyArray1<'_, i32>, name: &str) -> PyResult<&'a [i32]> {
    values
        .as_slice()
        .map_err(|_| PyValueError::new_err(format!("{name} must be C-contiguous")))
}

fn contiguous_2d<'a>(values: &'a PyReadonlyArray2<'_, i32>, name: &str) -> PyResult<&'a [i32]> {
    values
        .as_slice()
        .map_err(|_| PyValueError::new_err(format!("{name} must be C-contiguous")))
}

fn parse_market_list(value: &Bound<'_, PyAny>) -> Vec<TokenList> {
    value
        .downcast::<PyList>()
        .map(|orders| {
            orders
                .iter()
                .map(|order| to_token_list(&order))
                .collect::<Vec<_>>()
        })
        .unwrap_or_default()
}

#[allow(clippy::too_many_arguments)]
fn mixed_step_one(
    environment: &mut RustEnv,
    game: usize,
    players: &[i32],
    candidate_units: &[i32],
    candidate_market: &[i32],
    fixed_types: &[i32],
    fixed_crops: &[i32],
    fixed_items: &[i32],
    fixed_counts: &[i32],
    fixed_market: &[TokenList],
) -> Result<(bool, FixedMarketHistory), String> {
    let candidate_player = usize::try_from(players[game])
        .ok()
        .filter(|player| *player < 2)
        .ok_or_else(|| "candidate player must be 0 or 1".to_string())?;
    let fixed_player = 1 - candidate_player;
    let market_history =
        encode_fixed_market_history(&environment.engine, fixed_player, fixed_market)
            .map_err(str::to_string)?;
    let unit_start = game * ACTION_MAX_OWN_UNITS;
    let market_start = game * ACTION_MARKET_SLOTS;
    let candidate = decode_player_action_ids(
        &environment.engine,
        candidate_player,
        &candidate_units[unit_start..unit_start + ACTION_MAX_OWN_UNITS],
        &candidate_market[market_start..market_start + ACTION_MARKET_SLOTS],
    )
    .map_err(str::to_string)?;
    let fixed = decode_fixed_unit_labels(
        &environment.engine,
        fixed_player,
        &fixed_types[unit_start..unit_start + ACTION_MAX_OWN_UNITS],
        &fixed_crops[unit_start..unit_start + ACTION_MAX_OWN_UNITS],
        &fixed_items[unit_start..unit_start + ACTION_MAX_OWN_UNITS],
        &fixed_counts[unit_start..unit_start + ACTION_MAX_OWN_UNITS],
        fixed_market.to_vec(),
    )
    .map_err(str::to_string)?;
    let parsed = if candidate_player == 0 {
        [candidate, fixed]
    } else {
        [fixed, candidate]
    };
    let done = environment.advance_parsed(&parsed)?;
    Ok((done, market_history))
}

fn mixed_outcomes_to_python(
    py: Python<'_>,
    outcomes: Vec<(bool, FixedMarketHistory)>,
) -> PyResult<MixedStepOutput> {
    let games = outcomes.len();
    let mut done = Vec::with_capacity(games);
    let mut quantities = Vec::with_capacity(games * MARKET_LABEL_SLOTS);
    let mut hire = Vec::with_capacity(games);
    let mut buy_land = Vec::with_capacity(games);
    for (game_done, history) in outcomes {
        done.push(game_done);
        quantities.extend(history.quantities);
        hire.push(history.hire);
        buy_land.push(history.buy_land);
    }
    Ok((
        done,
        Array2::from_shape_vec((games, MARKET_LABEL_SLOTS), quantities)
            .unwrap()
            .into_pyarray(py)
            .unbind(),
        hire,
        buy_land,
    ))
}

fn fixed_tape_batch_to_python(py: Python<'_>, encoded: Vec<TapeFeatures>) -> PyResult<Py<PyDict>> {
    let games = encoded.len();
    let mut global = Vec::with_capacity(games * TAPE_GLOBAL_DIM);
    let mut resource = Vec::with_capacity(games * TAPE_RESOURCE_TOKENS * TAPE_RESOURCE_DIM);
    let mut shop = Vec::with_capacity(games * TAPE_SHOP_TOKENS * TAPE_SHOP_DIM);
    let mut tile = Vec::with_capacity(games * TAPE_TILE_TOKENS * TAPE_TILE_DIM);
    let mut unit = Vec::with_capacity(games * TAPE_UNIT_TOKENS * TAPE_UNIT_DIM);
    let mut unit_pad_mask = Vec::with_capacity(games * TAPE_UNIT_TOKENS);
    for mut values in encoded {
        global.append(&mut values.global);
        resource.append(&mut values.resource);
        shop.append(&mut values.shop);
        tile.append(&mut values.tile);
        unit.append(&mut values.unit);
        unit_pad_mask.append(&mut values.unit_pad_mask);
    }
    let arrays = PyDict::new(py);
    arrays.set_item(
        "global",
        Array3::from_shape_vec((games, 1, TAPE_GLOBAL_DIM), global)
            .unwrap()
            .into_pyarray(py),
    )?;
    arrays.set_item(
        "resource",
        Array3::from_shape_vec((games, TAPE_RESOURCE_TOKENS, TAPE_RESOURCE_DIM), resource)
            .unwrap()
            .into_pyarray(py),
    )?;
    arrays.set_item(
        "shop",
        Array3::from_shape_vec((games, TAPE_SHOP_TOKENS, TAPE_SHOP_DIM), shop)
            .unwrap()
            .into_pyarray(py),
    )?;
    arrays.set_item(
        "tile",
        Array3::from_shape_vec((games, TAPE_TILE_TOKENS, TAPE_TILE_DIM), tile)
            .unwrap()
            .into_pyarray(py),
    )?;
    arrays.set_item(
        "unit",
        Array3::from_shape_vec((games, TAPE_UNIT_TOKENS, TAPE_UNIT_DIM), unit)
            .unwrap()
            .into_pyarray(py),
    )?;
    arrays.set_item(
        "unit_pad_mask",
        Array2::from_shape_vec((games, TAPE_UNIT_TOKENS), unit_pad_mask)
            .unwrap()
            .into_pyarray(py),
    )?;
    Ok(arrays.unbind())
}

#[pyfunction]
#[allow(clippy::too_many_arguments)]
fn step_mixed_batch(
    py: Python<'_>,
    environments: &Bound<'_, PyList>,
    candidate_players: PyReadonlyArray1<'_, i32>,
    candidate_unit_ids: PyReadonlyArray2<'_, i32>,
    candidate_market_ids: PyReadonlyArray2<'_, i32>,
    fixed_unit_types: PyReadonlyArray2<'_, i32>,
    fixed_unit_crops: PyReadonlyArray2<'_, i32>,
    fixed_unit_items: PyReadonlyArray2<'_, i32>,
    fixed_unit_counts: PyReadonlyArray2<'_, i32>,
    fixed_market: &Bound<'_, PyAny>,
) -> PyResult<MixedStepOutput> {
    let games = environments.len();
    if candidate_players.shape() != [games]
        || candidate_unit_ids.shape() != [games, ACTION_MAX_OWN_UNITS]
        || candidate_market_ids.shape() != [games, ACTION_MARKET_SLOTS]
        || fixed_unit_types.shape() != [games, ACTION_MAX_OWN_UNITS]
        || fixed_unit_crops.shape() != [games, ACTION_MAX_OWN_UNITS]
        || fixed_unit_items.shape() != [games, ACTION_MAX_OWN_UNITS]
        || fixed_unit_counts.shape() != [games, ACTION_MAX_OWN_UNITS]
    {
        return Err(PyValueError::new_err(
            "invalid mixed-step batch input shape",
        ));
    }
    let players = candidate_players
        .as_slice()
        .map_err(|_| PyValueError::new_err("candidate players must be C-contiguous"))?;
    let candidate_units = candidate_unit_ids
        .as_slice()
        .map_err(|_| PyValueError::new_err("candidate Unit ids must be C-contiguous"))?;
    let candidate_market = candidate_market_ids
        .as_slice()
        .map_err(|_| PyValueError::new_err("candidate market ids must be C-contiguous"))?;
    let fixed_types = fixed_unit_types
        .as_slice()
        .map_err(|_| PyValueError::new_err("fixed Unit types must be C-contiguous"))?;
    let fixed_crops = fixed_unit_crops
        .as_slice()
        .map_err(|_| PyValueError::new_err("fixed Unit crops must be C-contiguous"))?;
    let fixed_items = fixed_unit_items
        .as_slice()
        .map_err(|_| PyValueError::new_err("fixed Unit items must be C-contiguous"))?;
    let fixed_counts = fixed_unit_counts
        .as_slice()
        .map_err(|_| PyValueError::new_err("fixed Unit counts must be C-contiguous"))?;
    let fixed_market = fixed_market
        .downcast::<PyList>()
        .map(|orders| {
            orders
                .iter()
                .map(|order| to_token_list(&order))
                .collect::<Vec<_>>()
        })
        .unwrap_or_default();

    let mut done = Vec::with_capacity(games);
    let mut history_quantities = Vec::with_capacity(games * MARKET_LABEL_SLOTS);
    let mut history_hire = Vec::with_capacity(games);
    let mut history_buy_land = Vec::with_capacity(games);
    for (game, item) in environments.iter().enumerate() {
        let mut environment = item.extract::<PyRefMut<'_, RustEnv>>()?;
        let candidate_player = usize::try_from(players[game])
            .ok()
            .filter(|player| *player < 2)
            .ok_or_else(|| PyValueError::new_err("candidate player must be 0 or 1"))?;
        let fixed_player = 1 - candidate_player;
        let market_history =
            encode_fixed_market_history(&environment.engine, fixed_player, &fixed_market)
                .map_err(PyValueError::new_err)?;
        history_quantities.extend(market_history.quantities);
        history_hire.push(market_history.hire);
        history_buy_land.push(market_history.buy_land);
        let unit_start = game * ACTION_MAX_OWN_UNITS;
        let market_start = game * ACTION_MARKET_SLOTS;
        let candidate = decode_player_action_ids(
            &environment.engine,
            candidate_player,
            &candidate_units[unit_start..unit_start + ACTION_MAX_OWN_UNITS],
            &candidate_market[market_start..market_start + ACTION_MARKET_SLOTS],
        )
        .map_err(PyValueError::new_err)?;
        let fixed = decode_fixed_unit_labels(
            &environment.engine,
            fixed_player,
            &fixed_types[unit_start..unit_start + ACTION_MAX_OWN_UNITS],
            &fixed_crops[unit_start..unit_start + ACTION_MAX_OWN_UNITS],
            &fixed_items[unit_start..unit_start + ACTION_MAX_OWN_UNITS],
            &fixed_counts[unit_start..unit_start + ACTION_MAX_OWN_UNITS],
            fixed_market.clone(),
        )
        .map_err(PyValueError::new_err)?;
        let parsed = if candidate_player == 0 {
            [candidate, fixed]
        } else {
            [fixed, candidate]
        };
        done.push(
            environment
                .advance_parsed(&parsed)
                .map_err(PyRuntimeError::new_err)?,
        );
    }
    Ok((
        done,
        Array2::from_shape_vec((games, MARKET_LABEL_SLOTS), history_quantities)
            .unwrap()
            .into_pyarray(py)
            .unbind(),
        history_hire,
        history_buy_land,
    ))
}

#[pyfunction]
#[pyo3(signature = (environments, memory_features=None))]
fn fixed_features_batch(
    py: Python<'_>,
    environments: &Bound<'_, PyList>,
    memory_features: Option<PyReadonlyArray2<'_, f32>>,
) -> PyResult<(Py<PyDict>, Vec<usize>, Vec<usize>)> {
    let batch_size = environments.len() * 2;
    let memory_rows = memory_features.as_ref().map(|memory| memory.as_array());
    if let Some(rows) = &memory_rows
        && rows.shape() != [batch_size, MEMORY_DIM]
    {
        return Err(PyValueError::new_err(format!(
            "memory_features must have shape ({batch_size}, {MEMORY_DIM})"
        )));
    }

    let mut snapshots = Vec::with_capacity(environments.len());
    for (game, item) in environments.iter().enumerate() {
        let environment = item.extract::<PyRef<'_, RustEnv>>()?;
        let memories: [Vec<f32>; 2] = std::array::from_fn(|player| {
            if let Some(rows) = &memory_rows {
                rows.row(game * 2 + player).to_vec()
            } else {
                environment.trackers[player].encoded_memory().to_vec()
            }
        });
        snapshots.push((environment.engine.clone(), memories));
    }
    let encoded_by_game: Result<Vec<_>, _> = snapshots
        .par_iter()
        .map(|(engine, memories)| {
            Ok::<_, &'static str>([
                encode_fixed(engine, 0, &memories[0])?,
                encode_fixed(engine, 1, &memories[1])?,
            ])
        })
        .collect();
    let encoded_batch = encoded_by_game
        .map_err(PyValueError::new_err)?
        .into_iter()
        .flatten();

    let mut features = Vec::with_capacity(batch_size * FIXED_TOKENS * FEATURE_DIM);
    let mut memories = Vec::with_capacity(batch_size * MEMORY_DIM);
    let mut coordinates = Vec::with_capacity(batch_size * FIXED_TOKENS * 2);
    let mut spatial_mask = Vec::with_capacity(batch_size * FIXED_TOKENS);
    let mut rope_groups = Vec::with_capacity(batch_size * FIXED_TOKENS);
    let mut token_mask = Vec::with_capacity(batch_size * FIXED_TOKENS);
    let mut unit_indices = Vec::with_capacity(batch_size * MAX_OWN_UNITS);
    let mut market_indices = Vec::with_capacity(batch_size * MARKET_SLOTS);
    let mut actual_own_units = Vec::with_capacity(batch_size);
    let mut encoded_own_units = Vec::with_capacity(batch_size);
    for mut encoded in encoded_batch {
        features.append(&mut encoded.features);
        memories.append(&mut encoded.memory_features);
        coordinates.append(&mut encoded.coordinates);
        spatial_mask.append(&mut encoded.spatial_mask);
        rope_groups.append(&mut encoded.rope_groups);
        token_mask.append(&mut encoded.token_mask);
        unit_indices.append(&mut encoded.unit_indices);
        market_indices.append(&mut encoded.market_indices);
        actual_own_units.push(encoded.actual_own_units);
        encoded_own_units.push(encoded.encoded_own_units);
    }

    let arrays = PyDict::new(py);
    arrays.set_item(
        "features",
        Array3::from_shape_vec((batch_size, FIXED_TOKENS, FEATURE_DIM), features)
            .unwrap()
            .into_pyarray(py),
    )?;
    arrays.set_item(
        "memory_features",
        Array2::from_shape_vec((batch_size, MEMORY_DIM), memories)
            .unwrap()
            .into_pyarray(py),
    )?;
    arrays.set_item(
        "coordinates",
        Array3::from_shape_vec((batch_size, FIXED_TOKENS, 2), coordinates)
            .unwrap()
            .into_pyarray(py),
    )?;
    arrays.set_item(
        "spatial_mask",
        Array2::from_shape_vec((batch_size, FIXED_TOKENS), spatial_mask)
            .unwrap()
            .into_pyarray(py),
    )?;
    arrays.set_item(
        "rope_groups",
        Array2::from_shape_vec((batch_size, FIXED_TOKENS), rope_groups)
            .unwrap()
            .into_pyarray(py),
    )?;
    arrays.set_item(
        "token_mask",
        Array2::from_shape_vec((batch_size, FIXED_TOKENS), token_mask)
            .unwrap()
            .into_pyarray(py),
    )?;
    arrays.set_item(
        "unit_indices",
        Array2::from_shape_vec((batch_size, MAX_OWN_UNITS), unit_indices)
            .unwrap()
            .into_pyarray(py),
    )?;
    arrays.set_item(
        "market_indices",
        Array2::from_shape_vec((batch_size, MARKET_SLOTS), market_indices)
            .unwrap()
            .into_pyarray(py),
    )?;
    Ok((arrays.unbind(), actual_own_units, encoded_own_units))
}

fn tracked_snapshots(
    environments: &Bound<'_, PyList>,
) -> PyResult<Vec<(Engine, [[f32; MEMORY_DIM]; 2])>> {
    let mut snapshots = Vec::with_capacity(environments.len());
    for item in environments.iter() {
        let environment = item.extract::<PyRef<'_, RustEnv>>()?;
        snapshots.push((
            environment.engine.clone(),
            [
                environment.trackers[0].encoded_memory(),
                environment.trackers[1].encoded_memory(),
            ],
        ));
    }
    Ok(snapshots)
}

fn encode_partitioned_snapshots(
    snapshots: &[(Engine, [[f32; MEMORY_DIM]; 2])],
) -> Result<(Vec<f32>, Vec<FeatureCounts>), &'static str> {
    let agent_stride = FIXED_TOKENS * FEATURE_DIM;
    let game_stride = 2 * agent_stride;
    let mut features = vec![0.0; snapshots.len() * game_stride];
    let counts_by_game: Result<Vec<_>, _> = features
        .par_chunks_mut(game_stride)
        .zip(snapshots.par_iter())
        .map(|(game_features, (engine, memories))| {
            let (first, second) = game_features.split_at_mut(agent_stride);
            Ok::<_, &'static str>([
                encode_partitioned_features_into(engine, 0, &memories[0], first)?,
                encode_partitioned_features_into(engine, 1, &memories[1], second)?,
            ])
        })
        .collect();
    Ok((features, counts_by_game?.into_iter().flatten().collect()))
}

#[pyfunction]
fn partitioned_features_batch(
    py: Python<'_>,
    environments: &Bound<'_, PyList>,
) -> PyResult<PartitionedBatchOutput> {
    let snapshots = tracked_snapshots(environments)?;
    let (features, counts) =
        encode_partitioned_snapshots(&snapshots).map_err(PyValueError::new_err)?;
    let batch_size = environments.len() * 2;
    let features = Array3::from_shape_vec((batch_size, FIXED_TOKENS, FEATURE_DIM), features)
        .unwrap()
        .into_pyarray(py)
        .unbind();
    let actual_own_units = counts.iter().map(|count| count.actual_own_units).collect();
    let encoded_own_units = counts.iter().map(|count| count.encoded_own_units).collect();
    Ok((features, actual_own_units, encoded_own_units))
}

#[pyfunction]
fn partitioned_features_players_batch(
    py: Python<'_>,
    environments: &Bound<'_, PyList>,
    players: PyReadonlyArray1<'_, i32>,
) -> PyResult<PartitionedBatchOutput> {
    let games = environments.len();
    if players.shape() != [games] {
        return Err(PyValueError::new_err(
            "players must have one entry per environment",
        ));
    }
    let players = players
        .as_slice()
        .map_err(|_| PyValueError::new_err("players must be C-contiguous"))?;
    let mut snapshots = Vec::with_capacity(games);
    for (game, item) in environments.iter().enumerate() {
        let player = usize::try_from(players[game])
            .ok()
            .filter(|player| *player < 2)
            .ok_or_else(|| PyValueError::new_err("player must be 0 or 1"))?;
        let environment = item.extract::<PyRef<'_, RustEnv>>()?;
        snapshots.push((
            environment.engine.clone(),
            player,
            environment.trackers[player].encoded_memory(),
        ));
    }

    let agent_stride = FIXED_TOKENS * FEATURE_DIM;
    let mut features = vec![0.0; games * agent_stride];
    let counts: Result<Vec<_>, _> = features
        .par_chunks_mut(agent_stride)
        .zip(snapshots.par_iter())
        .map(|(output, (engine, player, memory))| {
            encode_partitioned_features_into(engine, *player, memory, output)
        })
        .collect();
    let counts = counts.map_err(PyValueError::new_err)?;
    let features = Array3::from_shape_vec((games, FIXED_TOKENS, FEATURE_DIM), features)
        .unwrap()
        .into_pyarray(py)
        .unbind();
    let actual_own_units = counts.iter().map(|count| count.actual_own_units).collect();
    let encoded_own_units = counts.iter().map(|count| count.encoded_own_units).collect();
    Ok((features, actual_own_units, encoded_own_units))
}

#[pyfunction]
#[allow(clippy::too_many_arguments)]
fn fixed_tape_features_batch(
    py: Python<'_>,
    environments: &Bound<'_, PyList>,
    players: PyReadonlyArray1<'_, i32>,
    previous_unit_types: PyReadonlyArray2<'_, i32>,
    previous_unit_counts: PyReadonlyArray1<'_, i32>,
    previous_market_qty: PyReadonlyArray2<'_, i32>,
    previous_hire: PyReadonlyArray1<'_, i32>,
    previous_buy_land: PyReadonlyArray1<'_, i32>,
    had_previous: PyReadonlyArray1<'_, bool>,
) -> PyResult<Py<PyDict>> {
    let games = environments.len();
    if players.shape() != [games]
        || previous_unit_types.shape() != [games, TAPE_UNIT_TOKENS]
        || previous_unit_counts.shape() != [games]
        || previous_market_qty.shape() != [games, MARKET_LABEL_SLOTS]
        || previous_hire.shape() != [games]
        || previous_buy_land.shape() != [games]
        || had_previous.shape() != [games]
    {
        return Err(PyValueError::new_err(
            "invalid fixed tape batch input shape",
        ));
    }
    let players = players
        .as_slice()
        .map_err(|_| PyValueError::new_err("players must be C-contiguous"))?;
    let unit_types = previous_unit_types
        .as_slice()
        .map_err(|_| PyValueError::new_err("previous unit types must be C-contiguous"))?;
    let unit_counts = previous_unit_counts
        .as_slice()
        .map_err(|_| PyValueError::new_err("previous unit counts must be C-contiguous"))?;
    let market_qty = previous_market_qty
        .as_slice()
        .map_err(|_| PyValueError::new_err("previous market quantities must be C-contiguous"))?;
    let hire = previous_hire
        .as_slice()
        .map_err(|_| PyValueError::new_err("previous hire must be C-contiguous"))?;
    let buy_land = previous_buy_land
        .as_slice()
        .map_err(|_| PyValueError::new_err("previous buy land must be C-contiguous"))?;
    let had_previous = had_previous
        .as_slice()
        .map_err(|_| PyValueError::new_err("previous flags must be C-contiguous"))?;

    let mut global = Vec::with_capacity(games * TAPE_GLOBAL_DIM);
    let mut resource = Vec::with_capacity(games * TAPE_RESOURCE_TOKENS * TAPE_RESOURCE_DIM);
    let mut shop = Vec::with_capacity(games * TAPE_SHOP_TOKENS * TAPE_SHOP_DIM);
    let mut tile = Vec::with_capacity(games * TAPE_TILE_TOKENS * TAPE_TILE_DIM);
    let mut unit = Vec::with_capacity(games * TAPE_UNIT_TOKENS * TAPE_UNIT_DIM);
    let mut unit_pad_mask = Vec::with_capacity(games * TAPE_UNIT_TOKENS);
    for (game, item) in environments.iter().enumerate() {
        let environment = item.extract::<PyRef<'_, RustEnv>>()?;
        let player = usize::try_from(players[game])
            .map_err(|_| PyValueError::new_err("player must be 0 or 1"))?;
        let unit_start = game * TAPE_UNIT_TOKENS;
        let market_start = game * MARKET_LABEL_SLOTS;
        let history = TapeHistory {
            unit_types: &unit_types[unit_start..unit_start + TAPE_UNIT_TOKENS],
            unit_count: unit_counts[game],
            market_qty: &market_qty[market_start..market_start + MARKET_LABEL_SLOTS],
            hire: hire[game],
            buy_land: buy_land[game],
            available: had_previous[game],
        };
        let encoded = encode_fixed_tape(&environment.engine, player, &history)
            .map_err(PyValueError::new_err)?;
        global.extend(encoded.global);
        resource.extend(encoded.resource);
        shop.extend(encoded.shop);
        tile.extend(encoded.tile);
        unit.extend(encoded.unit);
        unit_pad_mask.extend(encoded.unit_pad_mask);
    }

    let arrays = PyDict::new(py);
    arrays.set_item(
        "global",
        Array3::from_shape_vec((games, 1, TAPE_GLOBAL_DIM), global)
            .unwrap()
            .into_pyarray(py),
    )?;
    arrays.set_item(
        "resource",
        Array3::from_shape_vec((games, TAPE_RESOURCE_TOKENS, TAPE_RESOURCE_DIM), resource)
            .unwrap()
            .into_pyarray(py),
    )?;
    arrays.set_item(
        "shop",
        Array3::from_shape_vec((games, TAPE_SHOP_TOKENS, TAPE_SHOP_DIM), shop)
            .unwrap()
            .into_pyarray(py),
    )?;
    arrays.set_item(
        "tile",
        Array3::from_shape_vec((games, TAPE_TILE_TOKENS, TAPE_TILE_DIM), tile)
            .unwrap()
            .into_pyarray(py),
    )?;
    arrays.set_item(
        "unit",
        Array3::from_shape_vec((games, TAPE_UNIT_TOKENS, TAPE_UNIT_DIM), unit)
            .unwrap()
            .into_pyarray(py),
    )?;
    arrays.set_item(
        "unit_pad_mask",
        Array2::from_shape_vec((games, TAPE_UNIT_TOKENS), unit_pad_mask)
            .unwrap()
            .into_pyarray(py),
    )?;
    Ok(arrays.unbind())
}

#[pyfunction]
fn copy_f32_to_bf16_bits(
    source: PyReadonlyArray3<'_, f32>,
    mut destination: PyReadwriteArray3<'_, u16>,
) -> PyResult<()> {
    if source.shape() != destination.shape() {
        return Err(PyValueError::new_err(
            "source and BF16 destination shapes must match",
        ));
    }
    let source = source
        .as_slice()
        .map_err(|_| PyValueError::new_err("source must be C-contiguous"))?;
    let mut destination_view = destination.as_array_mut();
    let destination = destination_view
        .as_slice_mut()
        .ok_or_else(|| PyValueError::new_err("destination must be C-contiguous"))?;
    destination
        .par_iter_mut()
        .zip(source.par_iter())
        .for_each(|(output, &value)| *output = half::bf16::from_f32(value).to_bits());
    Ok(())
}

#[pymodule]
fn route_rl_action_engine(m: &Bound<'_, PyModule>) -> PyResult<()> {
    m.add_class::<NativePrefixResolver>()?;
    m.add("OWNED_EXECUTION_CONTRACT", "official-prefix-full-action-ppo-v1")?;
    m.add("OWNED_SOURCE_SHA256", option_env!("ROUTE_RL_NATIVE_SOURCE_SHA256").unwrap_or("unrecorded"))?;
    m.add_class::<RustEnv>()?;
    m.add_class::<RustBatchEnv>()?;
    m.add_function(wrap_pyfunction!(fixed_features_batch, m)?)?;
    m.add_function(wrap_pyfunction!(partitioned_features_batch, m)?)?;
    m.add_function(wrap_pyfunction!(partitioned_features_players_batch, m)?)?;
    m.add_function(wrap_pyfunction!(fixed_tape_features_batch, m)?)?;
    m.add_function(wrap_pyfunction!(step_ids_batch, m)?)?;
    m.add_function(wrap_pyfunction!(step_mixed_batch, m)?)?;
    m.add_function(wrap_pyfunction!(copy_f32_to_bf16_bits, m)?)?;
    Ok(())
}

/// A private-own/public-other snapshot for the project's exact conditional supports.
#[pyclass]
struct NativePrefixResolver { state: crate::owned_prefix::PrefixState }

#[pymethods]
impl NativePrefixResolver {
    #[getter]
    fn unit_count(&self) -> usize { self.state.unit_count }
    #[getter]
    fn player(&self) -> usize { self.state.player }
    fn observation(&self, py: Python<'_>) -> PyResult<Py<PyDict>> {
        build_player_obs(py, &self.state.engine, self.state.player)
    }
    fn unit_support(&self, py: Python<'_>, index: usize) -> PyResult<Vec<bool>> {
        py.allow_threads(|| self.state.unit_support(index)).map_err(PyValueError::new_err)
    }
    fn apply_unit(&mut self, index: usize, identity: i32) -> PyResult<()> {
        self.state.apply_unit(index, identity).map_err(PyValueError::new_err)
    }
    fn market_support(&self, py: Python<'_>) -> PyResult<Vec<bool>> {
        py.allow_threads(|| self.state.market_support()).map_err(PyValueError::new_err)
    }
    fn apply_market(&mut self, identity: usize) -> PyResult<()> {
        self.state.apply_market(identity).map_err(PyValueError::new_err)
    }
}

