//! Observation-only opponent inventory reconstruction used by the RL feature path.
//!
//! This is a typed port of `inventory_tracker.py`. It deliberately uses an
//! observer's own private delta plus public farm/market transitions; it never
//! reads the opponent's `Private` state.

#![cfg_attr(not(feature = "python"), allow(dead_code))]

use std::cell::RefCell;
use std::collections::HashMap;

use crate::engine::{Engine, market_price};
use crate::state::*;

const SHED_CAPACITY: f64 = 100.0;
const FIXED_COSTS: [i64; 8] = [10, 20, 50, 100, 80, 300, 400, 500];
const LAND_PRICES: [i64; 3] = [1_000, 2_000, 4_000];

#[derive(Clone, Debug)]
pub(crate) struct PublicFlows {
    seeds: [f64; N_CROPS],
    items: [f64; N_ITEMS],
    produced: Vec<(Option<usize>, usize, f64)>,
    consumed: Vec<(Option<usize>, usize, f64)>,
    ambiguous_fertilizer: f64,
    before_positions: Vec<(i64, i64)>,
    after_positions: Vec<(i64, i64)>,
}

impl PublicFlows {
    fn new(before: Vec<(i64, i64)>, after: Vec<(i64, i64)>) -> Self {
        Self {
            seeds: [0.0; N_CROPS],
            items: [0.0; N_ITEMS],
            produced: Vec::new(),
            consumed: Vec::new(),
            ambiguous_fertilizer: 0.0,
            before_positions: before,
            after_positions: after,
        }
    }
}

#[derive(Clone, Debug)]
struct FixedPurchaseSummary {
    lower: [f64; 8],
    upper: [f64; 8],
    explained_budget: i64,
    residual: f64,
    solution_count: u8,
}

#[derive(Clone, Copy, Debug)]
struct Bounds {
    ways: u8,
    lower: [u16; 8],
    upper: [u16; 8],
}

thread_local! {
    static FIXED_PURCHASE_CACHE: RefCell<HashMap<(i64, i64), FixedPurchaseSummary>> =
        RefCell::new(HashMap::new());
}

#[derive(Clone, Debug)]
pub struct OpponentTracker {
    observer: usize,
    seeds: [f64; N_CROPS],
    item_total: [f64; N_ITEMS],
    shed: [f64; N_ITEMS],
    carried_by_unit: Vec<[f64; N_ITEMS]>,
    seed_uncertainty: [f64; N_CROPS],
    item_uncertainty: [f64; N_ITEMS],
    shed_uncertainty: [f64; N_ITEMS],
    unresolved_cash: f64,
    floor_sale_ambiguity: f64,
}

impl OpponentTracker {
    pub fn new(observer: usize) -> Self {
        assert!(observer < 2);
        Self {
            observer,
            seeds: [0.0; N_CROPS],
            item_total: [0.0; N_ITEMS],
            shed: [0.0; N_ITEMS],
            carried_by_unit: vec![[0.0; N_ITEMS]],
            seed_uncertainty: [0.0; N_CROPS],
            item_uncertainty: [0.0; N_ITEMS],
            shed_uncertainty: [0.0; N_ITEMS],
            unresolved_cash: 0.0,
            floor_sale_ambiguity: 0.0,
        }
    }

    pub fn encoded_memory(&self) -> [f32; 50] {
        let mut carried = [0.0; N_ITEMS];
        for inventory in &self.carried_by_unit {
            for item in 0..N_ITEMS {
                carried[item] += inventory[item];
            }
        }
        let carried_uncertainty: [f64; N_ITEMS] = std::array::from_fn(|item| {
            self.item_uncertainty[item]
                + (self.item_total[item] - self.shed[item] - carried[item]).abs()
        });
        let mut output = [0.0; 50];
        for item in 0..N_ITEMS {
            output[item] = log_scaled(self.item_total[item], SHED_CAPACITY);
            output[12 + item] = log_scaled(carried[item], SHED_CAPACITY);
            output[24 + item] = log_scaled(self.item_uncertainty[item], SHED_CAPACITY);
            output[36 + item] = log_scaled(carried_uncertainty[item], SHED_CAPACITY);
        }
        output[48] = log_scaled(self.unresolved_cash, 1_000_000.0);
        output[49] = log_scaled(self.floor_sale_ambiguity, SHED_CAPACITY);
        output
    }

    pub(crate) fn advance(
        &mut self,
        before: &Engine,
        after: &Engine,
        own_action: &PlayerAction,
        public_flows: &[PublicFlows; 2],
    ) {
        if before.step_no == 0 && after.step_no <= 0 {
            return;
        }
        let opponent = 1 - self.observer;
        let opponent_flows = &public_flows[opponent];
        let end_of_day = day(before) != day(after);
        self.resize_units(opponent_flows.before_positions.len());
        self.apply_public_flows(opponent_flows);
        self.infer_shed_transfers(
            &opponent_flows.before_positions,
            &opponent_flows.after_positions,
            end_of_day,
        );
        self.apply_market(
            before,
            after,
            own_action,
            end_of_day,
            &public_flows[self.observer],
            &opponent_flows.before_positions,
            &opponent_flows.after_positions,
        );
        if end_of_day {
            self.drop_all_at_end_of_day();
        } else {
            self.resize_units(opponent_flows.after_positions.len());
        }
        self.reconcile_total();
    }

    fn resize_units(&mut self, count: usize) {
        self.carried_by_unit.resize(count, [0.0; N_ITEMS]);
    }

    fn apply_public_flows(&mut self, flows: &PublicFlows) {
        for crop in 0..N_CROPS {
            self.seeds[crop] += flows.seeds[crop];
            if self.seeds[crop] < 0.0 {
                self.seed_uncertainty[crop] -= self.seeds[crop];
                self.seeds[crop] = 0.0;
            }
        }
        for item in 0..N_ITEMS {
            self.item_total[item] += flows.items[item];
        }
        for &(unit, item, quantity) in &flows.produced {
            let Some(unit) = unit.filter(|&index| index < self.carried_by_unit.len()) else {
                self.shed_uncertainty[item] += quantity;
                continue;
            };
            self.carried_by_unit[unit][item] += quantity;
        }
        for &(unit, item, quantity) in &flows.consumed {
            self.consume_carried(unit, item, quantity);
        }
        if flows.ambiguous_fertilizer > 0.0 {
            let estimate = 0.5 * flows.ambiguous_fertilizer;
            self.item_total[FERTILIZER] += estimate;
            self.item_uncertainty[FERTILIZER] += estimate;
            self.shed_uncertainty[FERTILIZER] += estimate;
        }
    }

    fn consume_carried(&mut self, unit: Option<usize>, item: usize, quantity: f64) {
        let mut remaining = quantity;
        if let Some(unit) = unit.filter(|&index| index < self.carried_by_unit.len()) {
            let taken = remaining.min(self.carried_by_unit[unit][item]);
            self.carried_by_unit[unit][item] -= taken;
            remaining -= taken;
        }
        if remaining > 0.0 {
            let taken = remaining.min(self.shed[item]);
            self.shed[item] -= taken;
            remaining -= taken;
        }
        if remaining > 0.0 {
            self.item_uncertainty[item] += remaining;
        }
        self.item_total[item] = self.item_total[item].max(0.0);
    }

    fn infer_shed_transfers(
        &mut self,
        before_positions: &[(i64, i64)],
        after_positions: &[(i64, i64)],
        end_of_day: bool,
    ) {
        if end_of_day {
            return;
        }
        for (index, &position) in before_positions.iter().enumerate() {
            if after_positions.get(index) != Some(&position) || !is_shed_access(position) {
                continue;
            }
            for item in 0..N_ITEMS {
                self.shed_uncertainty[item] +=
                    1.0_f64.min(self.shed[item]) + self.carried_by_unit[index][item];
            }
        }
    }

    #[allow(clippy::too_many_arguments)]
    fn apply_market(
        &mut self,
        before: &Engine,
        after: &Engine,
        own_action: &PlayerAction,
        end_of_day: bool,
        observer_flows: &PublicFlows,
        before_opponent_positions: &[(i64, i64)],
        after_opponent_positions: &[(i64, i64)],
    ) {
        let own_effect =
            infer_own_market_effect(before, after, own_action, self.observer, observer_flows);
        let drained = town_drain(before);
        let opponent = 1 - self.observer;
        let mut sale_revenue = 0.0;
        let mut product_buy_cost = 0.0;
        let stationary_shed_units = stationary_shed_units(
            before_opponent_positions,
            after_opponent_positions,
            end_of_day,
        );

        for item in 0..N_PRODUCTS {
            let before_inventory = before.market.inventory[item];
            let after_inventory = after.market.inventory[item];
            let joint_effect = after_inventory + drained[item] - before_inventory;
            let opponent_effect = joint_effect - own_effect[item];
            let buys = if (item == WHEAT || item == FERTILIZER) && opponent_effect < 0 {
                -opponent_effect
            } else {
                0
            };
            let sells = opponent_effect.max(0);

            if buys > 0 {
                for offset in 0..buys {
                    product_buy_cost +=
                        market_price(item, before_inventory - offset - 1, &before.market.params)
                            as f64;
                }
                self.item_total[item] += buys as f64;
                self.shed[item] += buys as f64;
            }
            if sells > 0 {
                for offset in 0..sells {
                    sale_revenue +=
                        market_price(item, before_inventory + offset, &before.market.params) as f64;
                }
                self.item_total[item] = (self.item_total[item] - sells as f64).max(0.0);
                let removed = self.shed[item].min(sells as f64);
                self.shed[item] -= removed;
                let mut remaining = sells as f64 - removed;
                for &unit in &stationary_shed_units {
                    if remaining <= 0.0 {
                        break;
                    }
                    let quantity = remaining.min(self.carried_by_unit[unit][item]);
                    self.carried_by_unit[unit][item] -= quantity;
                    remaining -= quantity;
                }
                if remaining > 0.0 {
                    self.shed_uncertainty[item] += remaining;
                }
            }
        }

        let fixed_cost =
            public_fixed_cost(&before.farms[opponent], &after.farms[opponent], end_of_day);
        let money_delta = after.farms[opponent].money - before.farms[opponent].money;
        let purchase_budget = sale_revenue - product_buy_cost - fixed_cost as f64 - money_delta;
        let animal_room = (SHED_CAPACITY as i64 - py_round(self.shed.iter().sum())).max(0);
        let summary = fixed_purchase_distribution(purchase_budget, animal_room);
        let ambiguous_budget = if summary.solution_count > 1 {
            summary.explained_budget
        } else {
            0
        };
        self.unresolved_cash = summary.residual.abs() + ambiguous_budget as f64;

        for crop in 0..N_CROPS {
            self.seeds[crop] += summary.lower[crop];
            self.seed_uncertainty[crop] += 0.5 * (summary.upper[crop] - summary.lower[crop]);
        }
        for animal in 0..N_ANIMALS {
            let summary_index = N_CROPS + animal;
            let item = FIRST_ANIMAL + animal;
            let quantity = summary.lower[summary_index];
            let uncertainty = 0.5 * (summary.upper[summary_index] - quantity);
            self.item_total[item] += quantity;
            self.shed[item] += quantity;
            self.item_uncertainty[item] += uncertainty;
            self.shed_uncertainty[item] += uncertainty;
        }

        let floor_items: Vec<usize> = (0..N_PRODUCTS)
            .filter(|&item| {
                before.market.prices[item]
                    .f()
                    .min(after.market.prices[item].f())
                    <= 1.0
            })
            .collect();
        let unexplained_gain =
            (money_delta + fixed_cost as f64 + product_buy_cost - sale_revenue).max(0.0);
        let available_floor_stock: f64 =
            floor_items.iter().map(|&item| self.item_total[item]).sum();
        let guaranteed_floor_sales =
            py_round(unexplained_gain).min(py_round(available_floor_stock));
        self.remove_floor_sales(&floor_items, guaranteed_floor_sales, &stationary_shed_units);
        let remaining_floor_stock: f64 =
            floor_items.iter().map(|&item| self.item_total[item]).sum();
        self.floor_sale_ambiguity = remaining_floor_stock.min((-purchase_budget).max(0.0));
    }

    fn remove_floor_sales(
        &mut self,
        floor_items: &[usize],
        quantity: i64,
        stationary_shed_units: &[usize],
    ) {
        let mut ordered = floor_items.to_vec();
        ordered.sort_by(|&left, &right| {
            self.shed[right]
                .total_cmp(&self.shed[left])
                .then_with(|| self.item_total[right].total_cmp(&self.item_total[left]))
                .then_with(|| left.cmp(&right))
        });
        let mut remaining = quantity as f64;
        for item in ordered {
            if remaining <= 0.0 {
                break;
            }
            let sold = remaining.min(self.shed[item]);
            self.shed[item] -= sold;
            self.item_total[item] = (self.item_total[item] - sold).max(0.0);
            remaining -= sold;
            for &unit in stationary_shed_units {
                if remaining <= 0.0 {
                    break;
                }
                let sold = remaining.min(self.carried_by_unit[unit][item]);
                self.carried_by_unit[unit][item] -= sold;
                self.item_total[item] = (self.item_total[item] - sold).max(0.0);
                remaining -= sold;
            }
            if remaining > 0.0 {
                let sold = remaining.min(self.item_total[item]);
                self.item_total[item] -= sold;
                self.item_uncertainty[item] += sold;
                remaining -= sold;
            }
        }
    }

    fn drop_all_at_end_of_day(&mut self) {
        for inventory in &mut self.carried_by_unit {
            for (item, carried_quantity) in inventory.iter_mut().enumerate() {
                let room = (SHED_CAPACITY - self.shed.iter().sum::<f64>()).max(0.0);
                let quantity = *carried_quantity;
                let moved = quantity.min(room);
                let discarded = quantity - moved;
                self.shed[item] += moved;
                *carried_quantity = 0.0;
                if discarded > 0.0 {
                    self.item_total[item] = (self.item_total[item] - discarded).max(0.0);
                    self.item_uncertainty[item] += discarded;
                }
            }
        }
        self.carried_by_unit = vec![[0.0; N_ITEMS]];
        if self.item_total.iter().sum::<f64>() <= SHED_CAPACITY {
            self.shed = self.item_total;
        }
    }

    fn reconcile_total(&mut self) {
        let mut carried = [0.0; N_ITEMS];
        for inventory in &self.carried_by_unit {
            for item in 0..N_ITEMS {
                carried[item] += inventory[item];
            }
        }
        for item in 0..N_ITEMS {
            let represented = self.shed[item] + carried[item];
            if represented < self.item_total[item] {
                self.shed_uncertainty[item] += self.item_total[item] - represented;
                continue;
            }
            if represented <= self.item_total[item] {
                continue;
            }
            let mut excess = represented - self.item_total[item];
            let removed = excess.min(self.shed[item]);
            self.shed[item] -= removed;
            excess -= removed;
            for inventory in &mut self.carried_by_unit {
                if excess <= 0.0 {
                    break;
                }
                let removed = excess.min(inventory[item]);
                inventory[item] -= removed;
                excess -= removed;
            }
            self.item_uncertainty[item] += represented - self.item_total[item];
        }
    }
}

pub(crate) fn infer_all_public_flows(before: &Engine, after: &Engine) -> [PublicFlows; 2] {
    [
        infer_public_flows(before, after, 0),
        infer_public_flows(before, after, 1),
    ]
}

fn infer_public_flows(before: &Engine, after: &Engine, player: usize) -> PublicFlows {
    let before_farm = &before.farms[player];
    let after_farm = &after.farms[player];
    let before_positions = farm_positions(before_farm);
    let after_positions = farm_positions(after_farm);
    let mut flows = PublicFlows::new(before_positions, after_positions);
    let end_of_day = day(before) != day(after);

    for y in 0..before.cfg.board_size as usize {
        for x in 0..before.cfg.board_size as usize {
            let before_tile = &before_farm.tiles[y][x];
            let after_tile = &after_farm.tiles[y][x];
            if matches!(
                (before_tile, after_tile),
                (Tile::Empty, Tile::Empty) | (Tile::Locked, Tile::Locked)
            ) {
                continue;
            }
            let unit = stationary_unit(
                &flows.before_positions,
                &flows.after_positions,
                (x as i64, y as i64),
                end_of_day,
            );

            if let Tile::Plant(after_plant) = after_tile {
                let new_plant = !matches!(
                    before_tile,
                    Tile::Plant(before_plant)
                        if before_plant.crop == after_plant.crop
                            && before_plant.planted_day == after_plant.planted_day
                );
                if new_plant {
                    flows.seeds[after_plant.crop] -= 1.0;
                }
            }

            if let Tile::Plant(before_plant) = before_tile {
                let harvested =
                    harvested_plant_units(before_plant, after_tile, before.step_no, end_of_day);
                if harvested > 0.0 {
                    flows.items[before_plant.crop] += harvested;
                    flows.produced.push((unit, before_plant.crop, harvested));
                }
                if let Tile::Plant(after_plant) = after_tile
                    && after_plant.fertilized_until_day > before_plant.fertilized_until_day
                {
                    flows.items[FERTILIZER] -= 1.0;
                    flows.consumed.push((unit, FERTILIZER, 1.0));
                }
            }

            let before_animal = match before_tile {
                Tile::Animal(animal) => Some(animal),
                _ => None,
            };
            let after_animal = match after_tile {
                Tile::Animal(animal) => Some(animal),
                _ => None,
            };
            if before_animal.is_none()
                && let Some(animal) = after_animal
            {
                let item = FIRST_ANIMAL + animal.animal;
                flows.items[item] -= 1.0;
                flows.consumed.push((unit, item, 1.0));
            }
            if let Some(before_animal) = before_animal {
                let harvested =
                    harvested_animal_units(before_animal, after_tile, before.step_no, end_of_day);
                if harvested > 0.0 {
                    let product = ANIMALS[before_animal.animal].product;
                    flows.items[product] += harvested;
                    flows.produced.push((unit, product, harvested));
                }
                let same_animal =
                    after_animal.is_some_and(|animal| animal.animal == before_animal.animal);
                if same_animal && !end_of_day {
                    let after_animal = after_animal.unwrap();
                    if before_animal.fertilizer_available && !after_animal.fertilizer_available {
                        flows.items[FERTILIZER] += 1.0;
                        flows.produced.push((unit, FERTILIZER, 1.0));
                    }
                    if !before_animal.fed_today && after_animal.fed_today {
                        flows.items[WHEAT] -= 1.0;
                        flows.consumed.push((unit, WHEAT, 1.0));
                    }
                } else if same_animal && end_of_day {
                    let after_animal = after_animal.unwrap();
                    if !before_animal.fed_today && after_animal.consecutive_unfed == 0 {
                        flows.items[WHEAT] -= 1.0;
                        flows.consumed.push((unit, WHEAT, 1.0));
                    }
                    if before_animal.fertilizer_available && unit.is_some() {
                        flows.ambiguous_fertilizer += 1.0;
                    }
                }
            }
        }
    }
    flows
}

fn harvested_plant_units(before: &Plant, after: &Tile, step: i64, end_of_day: bool) -> f64 {
    let before_yield = before.yield_units;
    if before_yield <= 0 {
        return 0.0;
    }
    let crop = before.crop;
    if matches!(after, Tile::Empty) && !CROPS[crop].ongoing {
        return before_yield as f64;
    }
    if matches!(after, Tile::Weed) {
        return if before_yield > 1 || !decay_due(before, step) {
            before_yield as f64
        } else {
            0.0
        };
    }
    let Tile::Plant(after) = after else {
        return 0.0;
    };
    if after.crop != crop {
        return 0.0;
    }
    let no_action_yield = before_yield - i64::from(decay_due(before, step));
    if !end_of_day {
        return if after.yield_units < no_action_yield.max(0) {
            before_yield as f64
        } else {
            0.0
        };
    }
    if !CROPS[crop].ongoing {
        return 0.0;
    }
    let next_day = step / 24 + 1;
    let first = match crop {
        2 => 8,
        3 => 10,
        _ => 0,
    };
    let days_since_first = next_day - before.planted_day - first;
    let interval = CROPS[crop].interval;
    let scheduled = days_since_first >= 0 && days_since_first % interval == 0;
    let production = if scheduled {
        let was_watered = before.watered_today || after.consecutive_unwatered == 0;
        let fertilized = before.fertilized_until_day >= step / 24;
        if was_watered && fertilized { 2 } else { 1 }
    } else {
        0
    };
    let no_harvest = CROPS[crop]
        .max_yield
        .min(no_action_yield.max(0) + production);
    let after_harvest = CROPS[crop].max_yield.min(production);
    if after.yield_units == after_harvest && no_harvest != after_harvest {
        before_yield as f64
    } else {
        0.0
    }
}

fn harvested_animal_units(before: &AnimalTile, after: &Tile, step: i64, end_of_day: bool) -> f64 {
    if before.yield_units <= 0 {
        return 0.0;
    }
    let Tile::Animal(after) = after else {
        return 0.0;
    };
    if after.animal != before.animal {
        return 0.0;
    }
    if !end_of_day {
        return if after.yield_units < before.yield_units {
            before.yield_units as f64
        } else {
            0.0
        };
    }
    let animal = &ANIMALS[before.animal];
    let next_day = step / 24 + 1;
    let days_since_first = next_day - before.placed_day - animal.first_yield_day;
    let scheduled = days_since_first >= 0 && days_since_first % animal.interval == 0;
    let production = if scheduled {
        let was_fed = before.fed_today || after.consecutive_unfed == 0;
        if was_fed {
            1 + before.pending_care_bonus
        } else {
            1
        }
    } else {
        0
    };
    let no_harvest = animal.max_held.min(before.yield_units + production);
    let after_harvest = animal.max_held.min(production);
    if after.yield_units == after_harvest && no_harvest != after_harvest {
        before.yield_units as f64
    } else {
        0.0
    }
}

fn infer_own_market_effect(
    before: &Engine,
    after: &Engine,
    action: &PlayerAction,
    player: usize,
    field_flows: &PublicFlows,
) -> [i64; N_PRODUCTS] {
    let before_total = private_item_totals(&before.privates[player]);
    let after_total = private_item_totals(&after.privates[player]);
    let mut buy_requested = [false; N_PRODUCTS];
    let mut sell_requested = [false; N_PRODUCTS];
    let mut buy_quantity = [0; N_PRODUCTS];
    let mut sell_quantity = [0; N_PRODUCTS];
    for order in action.market.iter().take(10) {
        let Some(tokens) = order.as_ref().filter(|tokens| tokens.len() >= 3) else {
            continue;
        };
        let Some(op) = token_str(&tokens[0]) else {
            continue;
        };
        let Some(item) = token_str(&tokens[1])
            .and_then(item_id)
            .filter(|&item| item < N_PRODUCTS)
        else {
            continue;
        };
        let Some(quantity) = token_int(&tokens[2]).map(|value| value.max(0)) else {
            continue;
        };
        match op {
            "BUY_PRODUCT" => {
                buy_requested[item] = true;
                buy_quantity[item] += quantity;
            }
            "SELL" => {
                sell_requested[item] = true;
                sell_quantity[item] += quantity;
            }
            _ => {}
        }
    }

    let mut visible_effect = [0; N_PRODUCTS];
    for item in 0..N_PRODUCTS {
        let net_buy = after_total[item] - before_total[item] - field_flows.items[item];
        let mut buys = 0;
        let mut sells = 0;
        match (buy_requested[item], sell_requested[item]) {
            (true, false) => buys = buy_quantity[item].min(py_round(net_buy).max(0)),
            (false, true) => sells = sell_quantity[item].min(py_round(-net_buy).max(0)),
            (true, true) if net_buy >= 0.0 => {
                buys = buy_quantity[item].min(py_round(net_buy));
            }
            (true, true) => sells = sell_quantity[item].min(py_round(-net_buy)),
            _ => {}
        }
        let visible_sells = if before.market.prices[item]
            .f()
            .min(after.market.prices[item].f())
            > 1.0
        {
            sells
        } else {
            0
        };
        visible_effect[item] = visible_sells - buys;
    }
    visible_effect
}

fn fixed_purchase_distribution(budget: f64, animal_room: i64) -> FixedPurchaseSummary {
    let rounded_budget = py_round(budget).max(0);
    let explained_budget = py_round(rounded_budget as f64 / 10.0) * 10;
    let room = animal_room.max(0);
    let mut cached = FIXED_PURCHASE_CACHE.with(|cache| {
        let mut cache = cache.borrow_mut();
        if let Some(summary) = cache.get(&(explained_budget, room)) {
            return summary.clone();
        }
        let summary = fixed_purchase_dp(explained_budget, room);
        if cache.len() >= 4_096 {
            cache.clear();
        }
        cache.insert((explained_budget, room), summary.clone());
        summary
    });
    cached.residual = budget - explained_budget as f64;
    cached
}

fn fixed_purchase_dp(explained_budget: i64, animal_room: i64) -> FixedPurchaseSummary {
    let scaled_budget = (explained_budget / 10).max(0) as usize;
    let costs: [usize; 8] = FIXED_COSTS.map(|cost| (cost / 10) as usize);
    let max_animals = animal_room.min((scaled_budget / 30) as i64).max(0) as usize;
    let width = max_animals + 1;
    let mut states = vec![None; (scaled_budget + 1) * width];
    states[0] = Some(Bounds {
        ways: 1,
        lower: [0; 8],
        upper: [0; 8],
    });

    for (item, &cost) in costs.iter().enumerate() {
        let animal_increment = usize::from(item >= N_CROPS);
        let mut next = states.clone();
        for spent in cost..=scaled_budget {
            for animals in animal_increment..=max_animals {
                let previous_index = (spent - cost) * width + animals - animal_increment;
                let Some(previous) = next[previous_index] else {
                    continue;
                };
                let index = spent * width + animals;
                let mut candidate = previous;
                candidate.lower[item] += 1;
                candidate.upper[item] += 1;
                if let Some(existing) = &mut next[index] {
                    existing.ways = existing.ways.saturating_add(candidate.ways).min(2);
                    for field in 0..8 {
                        existing.lower[field] = existing.lower[field].min(candidate.lower[field]);
                        existing.upper[field] = existing.upper[field].max(candidate.upper[field]);
                    }
                } else {
                    next[index] = Some(candidate);
                }
            }
        }
        states = next;
    }

    let mut combined: Option<Bounds> = None;
    for animals in 0..=max_animals {
        let Some(bounds) = states[scaled_budget * width + animals] else {
            continue;
        };
        if let Some(current) = &mut combined {
            current.ways = current.ways.saturating_add(bounds.ways).min(2);
            for item in 0..8 {
                current.lower[item] = current.lower[item].min(bounds.lower[item]);
                current.upper[item] = current.upper[item].max(bounds.upper[item]);
            }
        } else {
            combined = Some(bounds);
        }
    }
    let combined = combined.unwrap_or(Bounds {
        ways: 0,
        lower: [0; 8],
        upper: [0; 8],
    });
    FixedPurchaseSummary {
        lower: combined.lower.map(f64::from),
        upper: combined.upper.map(f64::from),
        explained_budget,
        residual: 0.0,
        solution_count: combined.ways,
    }
}

fn private_item_totals(private: &Private) -> [f64; N_ITEMS] {
    let mut totals = private.shed.map(|quantity| quantity as f64);
    for inventory in &private.inventories {
        for &(item, quantity) in &inventory.0 {
            totals[item] += quantity as f64;
        }
    }
    totals
}

fn town_drain(engine: &Engine) -> [i64; N_PRODUCTS] {
    let mut drained = [0; N_PRODUCTS];
    if engine.step_no % 4 == 0 {
        for &shop in &engine.town.unlocked_shops {
            let products = SHOP_PRODUCTS[shop];
            let multiplier = if products.len() == 1 { 2 } else { 1 };
            for &item in products {
                drained[item] += multiplier;
            }
        }
    }
    if engine.step_no % 24 == 0 {
        for quantity in &mut drained[..FERTILIZER] {
            *quantity += 1;
        }
    }
    drained
}

fn public_fixed_cost(before: &Farm, after: &Farm, end_of_day: bool) -> i64 {
    let before_land = before.unlocked_quadrants.len();
    let after_land = after.unlocked_quadrants.len();
    let extra_before = before_land.saturating_sub(1);
    let land_count = after_land.saturating_sub(before_land);
    let land_cost: i64 = LAND_PRICES[extra_before..extra_before + land_count]
        .iter()
        .sum();
    if end_of_day {
        return land_cost;
    }
    land_cost
        + hire_cost(
            before.hires_today,
            (after.hires_today - before.hires_today).max(0),
        )
}

fn hire_cost(hires_before: i64, count: i64) -> i64 {
    let (mut a, mut b) = (1, 1);
    let mut result = 0;
    for index in 0..hires_before + count {
        if index >= hires_before {
            result += a;
        }
        (a, b) = (b, a + b);
    }
    result
}

fn farm_positions(farm: &Farm) -> Vec<(i64, i64)> {
    std::iter::once(farm.farmer)
        .chain(farm.hands.iter().copied())
        .collect()
}

fn stationary_unit(
    before: &[(i64, i64)],
    after: &[(i64, i64)],
    position: (i64, i64),
    end_of_day: bool,
) -> Option<usize> {
    before.iter().enumerate().find_map(|(index, &candidate)| {
        if candidate == position && (end_of_day || after.get(index) == Some(&candidate)) {
            Some(index)
        } else {
            None
        }
    })
}

fn stationary_shed_units(
    before: &[(i64, i64)],
    after: &[(i64, i64)],
    end_of_day: bool,
) -> Vec<usize> {
    if end_of_day {
        return Vec::new();
    }
    before
        .iter()
        .enumerate()
        .filter_map(|(index, &position)| {
            (after.get(index) == Some(&position) && is_shed_access(position)).then_some(index)
        })
        .collect()
}

fn day(engine: &Engine) -> i64 {
    engine.step_no / 24
}

fn decay_due(plant: &Plant, step: i64) -> bool {
    plant.max_lifespan_step >= 0
        && step >= plant.max_lifespan_step
        && (step - plant.max_lifespan_step) % 2 == 0
}

fn is_shed_access(position: (i64, i64)) -> bool {
    matches!(position, (4, 4) | (5, 4) | (4, 5) | (5, 5))
}

fn py_round(value: f64) -> i64 {
    value.round_ties_even() as i64
}

fn log_scaled(value: f64, reference: f64) -> f32 {
    (value.max(0.0).ln_1p() / reference.ln_1p()) as f32
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn fresh_tracker_memory_is_zero() {
        assert_eq!(OpponentTracker::new(0).encoded_memory(), [0.0; 50]);
    }

    #[test]
    fn fixed_purchase_bounds_cover_ambiguous_seed_bundle() {
        let summary = fixed_purchase_distribution(20.0, 100);
        assert_eq!(summary.solution_count, 2);
        assert_eq!(summary.lower[0], 0.0);
        assert_eq!(summary.upper[0], 2.0);
        assert_eq!(summary.lower[1], 0.0);
        assert_eq!(summary.upper[1], 1.0);
    }
}
