//! Decode the frozen exp43 categorical action catalogs without Python objects.

use std::sync::{Arc, LazyLock};

use crate::engine::Engine;
use crate::state::*;

pub const UNIT_ACTION_COUNT: i32 = 500;
pub const MARKET_ACTION_COUNT: i32 = 1_075;
pub const MAX_OWN_UNITS: usize = 20;
pub const MARKET_SLOTS: usize = 10;
pub const ABSOLUTE_SELL_START: i32 = 10_000;
pub const ABSOLUTE_SELL_END: i32 = ABSOLUTE_SELL_START + N_PRODUCTS as i32 * 100;

pub(crate) const PICKUP_START: i32 = 5;
pub(crate) const DROP_ID: i32 = 245;
pub(crate) const PLACE_START: i32 = 246;
pub(crate) const PLANT_START: i32 = 486;
pub(crate) const SELL_START: i32 = 1_001;
type DecodedMarketAction = (Arc<[Token]>, Option<(usize, i64)>);

pub struct FixedMarketHistory {
    pub quantities: [i32; 19],
    pub hire: i32,
    pub buy_land: i32,
}

static UNIT_ACTIONS: LazyLock<Vec<Arc<[Token]>>> = LazyLock::new(|| {
    (0..UNIT_ACTION_COUNT)
        .map(build_unit_action)
        .collect::<Vec<_>>()
});

pub fn decode_player_action_ids(
    engine: &Engine,
    player: usize,
    unit_ids: &[i32],
    market_ids: &[i32],
) -> Result<PlayerAction, &'static str> {
    if player > 1 || unit_ids.len() != MAX_OWN_UNITS || market_ids.len() != MARKET_SLOTS {
        return Err("action id rows require player 0/1, 20 units, and 10 market slots");
    }
    if unit_ids
        .iter()
        .any(|&action| !(0..UNIT_ACTION_COUNT).contains(&action))
        || market_ids.iter().any(|&action| {
            !(0..MARKET_ACTION_COUNT).contains(&action)
                && !(ABSOLUTE_SELL_START..ABSOLUTE_SELL_END).contains(&action)
        })
    {
        return Err("action id is outside the legacy or absolute SELL catalog");
    }

    let actual_units = 1 + engine.farms[player].hands.len();
    let mut units: Vec<TokenList> = unit_ids
        .iter()
        .take(actual_units.min(MAX_OWN_UNITS))
        .map(|&action| Some(decode_unit_action(action)))
        .collect();
    units.resize_with(actual_units, || Some(tokens(&["PASS"])));
    let sellable = shed_after_unit_actions(engine, player, &units);
    let mut remaining_shed = sellable;
    let mut market = Vec::with_capacity(MARKET_SLOTS);
    for &action in market_ids {
        let Some((tokens, sold)) = decode_market_action(action, &remaining_shed) else {
            continue;
        };
        if let Some((item, quantity)) = sold {
            remaining_shed[item] = (remaining_shed[item] - quantity).max(0);
        }
        market.push(Some(tokens));
    }
    let farmer = units.remove(0);
    Ok(PlayerAction {
        farmer,
        hands: units,
        market,
    })
}

/// Decode the factorized v2 Unit heads used by submission 55548303.
pub fn decode_fixed_unit_labels(
    engine: &Engine,
    player: usize,
    unit_types: &[i32],
    unit_crops: &[i32],
    unit_items: &[i32],
    unit_counts: &[i32],
    market: Vec<TokenList>,
) -> Result<PlayerAction, &'static str> {
    if player > 1
        || unit_types.len() != MAX_OWN_UNITS
        || unit_crops.len() != MAX_OWN_UNITS
        || unit_items.len() != MAX_OWN_UNITS
        || unit_counts.len() != MAX_OWN_UNITS
    {
        return Err("fixed label rows require player 0/1 and 20 Unit labels");
    }
    let actual_units = 1 + engine.farms[player].hands.len();
    let mut units = Vec::with_capacity(actual_units);
    for unit in 0..actual_units.min(MAX_OWN_UNITS) {
        units.push(Some(decode_fixed_unit_label(
            engine,
            player,
            unit,
            unit_types[unit],
            unit_crops[unit],
            unit_items[unit],
            unit_counts[unit],
        )?));
    }
    units.resize_with(actual_units, || Some(tokens(&["PASS"])));
    let farmer = units.remove(0);
    Ok(PlayerAction {
        farmer,
        hands: units,
        market,
    })
}

/// Reproduce submission 55548303's v2 `encode_action` market summary.
pub fn encode_fixed_market_history(
    engine: &Engine,
    player: usize,
    market: &[TokenList],
) -> Result<FixedMarketHistory, &'static str> {
    if player > 1 {
        return Err("fixed history player must be 0 or 1");
    }
    let mut summed = [0_i64; 19];
    let mut hire = 0_i32;
    let mut buy_land = 0_i32;
    for order in market.iter().take(10) {
        let Some(tokens) = order.as_ref().filter(|tokens| !tokens.is_empty()) else {
            continue;
        };
        let Some(operation) = token_str(&tokens[0]) else {
            continue;
        };
        if operation == "HIRE" {
            hire += 1;
            continue;
        }
        if operation == "BUY_LAND" {
            buy_land = 1;
            continue;
        }
        let Some(item) = tokens.get(1).and_then(token_str).and_then(item_id) else {
            continue;
        };
        let Some(quantity) = tokens
            .get(2)
            .and_then(token_int)
            .filter(|quantity| *quantity > 0)
        else {
            continue;
        };
        let slot = match operation {
            "SELL" if item < N_PRODUCTS => Some(item),
            "BUY_SEED" if item < N_CROPS => Some(N_PRODUCTS + item),
            "BUY_ANIMAL" if item >= FIRST_ANIMAL => {
                Some(N_PRODUCTS + N_CROPS + item - FIRST_ANIMAL)
            }
            "BUY_PRODUCT" if item == WHEAT => Some(17),
            "BUY_PRODUCT" if item == FERTILIZER => Some(18),
            _ => None,
        };
        if let Some(slot) = slot {
            summed[slot] += quantity;
        }
    }
    let mut quantities = [0_i32; 19];
    for (slot, quantity) in summed
        .into_iter()
        .enumerate()
        .filter(|(_, quantity)| *quantity > 0)
    {
        let sell_stock = if slot < N_PRODUCTS {
            Some(engine.privates[player].shed[slot])
        } else {
            None
        };
        quantities[slot] = encode_fixed_market_quantity(quantity, sell_stock);
    }
    Ok(FixedMarketHistory {
        quantities,
        hire: hire.min(10),
        buy_land,
    })
}

fn encode_fixed_market_quantity(quantity: i64, sell_stock: Option<i64>) -> i32 {
    if sell_stock.is_some_and(|stock| stock > 0 && quantity >= stock)
        || (sell_stock.is_none() && quantity > 100)
    {
        return 67;
    }
    match quantity.min(100) {
        1..=64 => quantity as i32,
        65..=79 => 64,
        80..=99 => 65,
        _ => 66,
    }
}

fn decode_fixed_unit_label(
    engine: &Engine,
    player: usize,
    unit: usize,
    operation: i32,
    crop: i32,
    item: i32,
    count_class: i32,
) -> Result<Arc<[Token]>, &'static str> {
    const SIMPLE_ACTION_IDS: [i32; 18] = [
        4, 0, 1, 2, 3, 491, 492, 493, 497, 499, 498, -1, -1, -1, 245, 494, 495, 496,
    ];
    if !(0..18).contains(&operation)
        || !(0..N_CROPS as i32).contains(&crop)
        || !(0..N_ITEMS as i32).contains(&item)
        || !(0..=12).contains(&count_class)
    {
        return Err("fixed Unit label is outside the v2 label space");
    }
    if operation == 11 {
        return Ok(decode_unit_action(PLANT_START + crop));
    }
    if operation != 12 && operation != 13 {
        return Ok(decode_unit_action(SIMPLE_ACTION_IDS[operation as usize]));
    }

    let item = item as usize;
    let quantity = if count_class == 12 {
        let private = &engine.privates[player];
        let available = if operation == 12 {
            private.shed[item]
        } else {
            private
                .inventories
                .get(unit)
                .and_then(|inventory| inventory.0.iter().find(|(held, _)| *held == item))
                .map_or(0, |(_, quantity)| *quantity)
        };
        available.max(1)
    } else {
        i64::from(count_class + 1)
    };
    if quantity <= 20 {
        let start = if operation == 12 {
            PICKUP_START
        } else {
            PLACE_START
        };
        return Ok(decode_unit_action(
            start + item as i32 * 20 + quantity as i32 - 1,
        ));
    }
    let operation_name = if operation == 12 { "PICKUP" } else { "PLACE" };
    Ok(vec![
        Token::Str(operation_name.to_string()),
        Token::Str(ITEM_NAMES[item].to_string()),
        Token::Int(quantity),
    ]
    .into())
}

pub(crate) fn decode_unit_action(action: i32) -> Arc<[Token]> {
    UNIT_ACTIONS[action as usize].clone()
}

fn build_unit_action(action: i32) -> Arc<[Token]> {
    const SIMPLE: [&str; 5] = ["NORTH", "SOUTH", "EAST", "WEST", "PASS"];
    const TAIL: [&str; 9] = [
        "WATER",
        "HARVEST",
        "FERTILIZE",
        "BUILD_COOP",
        "BUILD_PASTURE",
        "DIG",
        "FEED",
        "COLLECT_FERTILIZER",
        "CARE",
    ];
    match action {
        0..=4 => tokens(&[SIMPLE[action as usize]]),
        PICKUP_START..DROP_ID => {
            let offset = action - PICKUP_START;
            item_quantity_tokens("PICKUP", offset / 20, offset % 20 + 1)
        }
        DROP_ID => tokens(&["DROP"]),
        PLACE_START..PLANT_START => {
            let offset = action - PLACE_START;
            item_quantity_tokens("PLACE", offset / 20, offset % 20 + 1)
        }
        PLANT_START..=490 => vec![
            Token::Str("PLANT".to_string()),
            Token::Str(ITEM_NAMES[(action - PLANT_START) as usize].to_string()),
        ]
        .into(),
        491..UNIT_ACTION_COUNT => tokens(&[TAIL[(action - 491) as usize]]),
        _ => unreachable!(),
    }
}

fn decode_market_action(
    action: i32,
    sellable_shed: &[i64; N_ITEMS],
) -> Option<DecodedMarketAction> {
    match action {
        0 => None,
        1..=500 => {
            let offset = action - 1;
            Some((
                item_quantity_tokens("BUY_SEED", offset / 100, offset % 100 + 1),
                None,
            ))
        }
        501..=700 => {
            let offset = action - 501;
            let item = if offset / 100 == 0 { WHEAT } else { FERTILIZER };
            Some((
                vec![
                    Token::Str("BUY_PRODUCT".to_string()),
                    Token::Str(ITEM_NAMES[item].to_string()),
                    Token::Int(offset as i64 % 100 + 1),
                ]
                .into(),
                None,
            ))
        }
        701..=1_000 => {
            let offset = action - 701;
            let item = FIRST_ANIMAL + (offset / 100) as usize;
            Some((
                vec![
                    Token::Str("BUY_ANIMAL".to_string()),
                    Token::Str(ITEM_NAMES[item].to_string()),
                    Token::Int(offset as i64 % 100 + 1),
                ]
                .into(),
                None,
            ))
        }
        SELL_START..=1_072 => {
            let offset = action - SELL_START;
            let item = (offset / 8) as usize;
            let quantity_class = offset % 8;
            let sellable = sellable_shed[item];
            let quantity = if quantity_class == 7 {
                sellable
            } else {
                decode_sell_quantity(quantity_class as i64 + 1, sellable)
            };
            (quantity > 0).then(|| {
                (
                    vec![
                        Token::Str("SELL".to_string()),
                        Token::Str(ITEM_NAMES[item].to_string()),
                        Token::Int(quantity),
                    ]
                    .into(),
                    Some((item, quantity)),
                )
            })
        }
        1_073 => Some((tokens(&["HIRE"]), None)),
        1_074 => Some((tokens(&["BUY_LAND"]), None)),
        ABSOLUTE_SELL_START..ABSOLUTE_SELL_END => {
            let offset = action - ABSOLUTE_SELL_START;
            let item = (offset / 100) as usize;
            let quantity = (offset as i64 % 100 + 1).min(sellable_shed[item]).max(0);
            (quantity > 0).then(|| {
                (
                    vec![
                        Token::Str("SELL".to_string()),
                        Token::Str(ITEM_NAMES[item].to_string()),
                        Token::Int(quantity),
                    ]
                    .into(),
                    Some((item, quantity)),
                )
            })
        }
        _ => unreachable!(),
    }
}

pub fn sell_stocks_before_slots(
    engine: &Engine,
    player: usize,
    unit_ids: &[i32],
    market_ids: &[i32],
    output: &mut [i32],
) -> Result<(), &'static str> {
    if output.len() != MARKET_SLOTS * N_PRODUCTS {
        return Err("SELL stock output must contain ten slots and nine products");
    }
    let decoded = decode_player_action_ids(engine, player, unit_ids, market_ids)?;
    let units: Vec<TokenList> = std::iter::once(decoded.farmer)
        .chain(decoded.hands)
        .collect();
    let mut remaining = shed_after_unit_actions(engine, player, &units);
    for (slot, &action) in market_ids.iter().enumerate() {
        for item in 0..N_PRODUCTS {
            output[slot * N_PRODUCTS + item] =
                i32::try_from(remaining[item]).map_err(|_| "SELL stock exceeds i32")?;
        }
        if let Some((_, Some((item, quantity)))) = decode_market_action(action, &remaining) {
            remaining[item] = (remaining[item] - quantity).max(0);
        }
    }
    Ok(())
}

fn shed_after_unit_actions(
    engine: &Engine,
    player: usize,
    unit_actions: &[TokenList],
) -> [i64; N_ITEMS] {
    let farm = &engine.farms[player];
    let private = &engine.privates[player];
    let positions: Vec<(i64, i64)> = std::iter::once(farm.farmer)
        .chain(farm.hands.iter().copied())
        .collect();
    let mut inventories: Vec<[i64; N_ITEMS]> = private
        .inventories
        .iter()
        .map(|inventory| {
            let mut dense = [0; N_ITEMS];
            for &(item, quantity) in &inventory.0 {
                dense[item] = quantity;
            }
            dense
        })
        .collect();
    let mut shed = private.shed;
    let mut shed_total: i64 = shed.iter().sum();

    for (index, action) in unit_actions.iter().take(positions.len()).enumerate() {
        let Some(action) = action.as_ref().filter(|tokens| !tokens.is_empty()) else {
            continue;
        };
        let position = positions[index];
        let inventory = &mut inventories[index];
        let Some(operation) = token_str(&action[0]) else {
            continue;
        };
        if operation == "DROP" && is_shed_access(position) {
            for item in 0..N_ITEMS {
                let room = (engine.cfg.shed_capacity - shed_total).max(0);
                let moved = inventory[item].min(room);
                shed[item] += moved;
                shed_total += moved;
                inventory[item] = 0;
            }
            continue;
        }
        let Some(item) = action.get(1).and_then(token_str).and_then(item_id) else {
            continue;
        };
        if operation == "PICKUP" && is_shed_access(position) {
            let requested = action.get(2).and_then(token_int).unwrap_or(1).max(0);
            let moved = requested.min(shed[item]);
            shed[item] -= moved;
            shed_total -= moved;
            inventory[item] += moved;
            continue;
        }
        if operation != "PLACE" {
            continue;
        }
        let tile = &farm.tiles[position.1 as usize][position.0 as usize];
        let animal_pen = item >= FIRST_ANIMAL
            && matches!(
                tile,
                Tile::Structure(structure)
                    if *structure == ANIMALS[item - FIRST_ANIMAL].structure
            );
        if animal_pen {
            inventory[item] = (inventory[item] - 1).max(0);
            continue;
        }
        if !is_shed_access(position) {
            continue;
        }
        let requested = action.get(2).and_then(token_int).unwrap_or(1).max(0);
        let room = (engine.cfg.shed_capacity - shed_total).max(0);
        let moved = requested.min(inventory[item]).min(room);
        shed[item] += moved;
        shed_total += moved;
        inventory[item] -= moved;
    }
    shed
}

fn decode_sell_quantity(numerator: i64, sellable: i64) -> i64 {
    if sellable <= 0 {
        return 0;
    }
    ((sellable * numerator + 4) / 8).clamp(1, sellable)
}

fn item_quantity_tokens(operation: &str, item: i32, quantity: i32) -> Arc<[Token]> {
    vec![
        Token::Str(operation.to_string()),
        Token::Str(ITEM_NAMES[item as usize].to_string()),
        Token::Int(quantity as i64),
    ]
    .into()
}

fn tokens(values: &[&str]) -> Arc<[Token]> {
    values
        .iter()
        .map(|value| Token::Str((*value).to_string()))
        .collect::<Vec<_>>()
        .into()
}

fn is_shed_access(position: (i64, i64)) -> bool {
    matches!(position, (4, 4) | (5, 4) | (4, 5) | (5, 5))
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn absolute_sell_all_products_quantities_and_stock_boundaries() {
        for item in 0..N_PRODUCTS {
            for stock in [0, 1, 8, 99, 100] {
                let mut shed = [0; N_ITEMS];
                shed[item] = stock;
                for quantity in 1..=100 {
                    let id = ABSOLUTE_SELL_START + item as i32 * 100 + quantity - 1;
                    let decoded = decode_market_action(id, &shed);
                    if stock == 0 {
                        assert!(decoded.is_none());
                    } else {
                        let (tokens, sold) = decoded.unwrap();
                        assert_eq!(token_str(&tokens[0]), Some("SELL"));
                        assert_eq!(token_str(&tokens[1]), Some(ITEM_NAMES[item]));
                        assert_eq!(token_int(&tokens[2]), Some((quantity as i64).min(stock)));
                        assert_eq!(sold, Some((item, (quantity as i64).min(stock))));
                    }
                }
            }
        }
    }

    #[test]
    fn stock_writer_tracks_unit_pickup_and_prior_fraction_sells() {
        let mut engine = Engine::new(0, Config::default());
        engine.privates[0].shed[WHEAT] = 37;
        let mut units = [4; MAX_OWN_UNITS];
        units[0] = PICKUP_START + 9;
        let mut markets = [0; MARKET_SLOTS];
        markets[0] = SELL_START + 3;
        markets[1] = SELL_START + 7;
        let mut stocks = [0; MARKET_SLOTS * N_PRODUCTS];
        sell_stocks_before_slots(&engine, 0, &units, &markets, &mut stocks).unwrap();
        assert_eq!(stocks[WHEAT], 27);
        assert_eq!(stocks[N_PRODUCTS + WHEAT], 13);
        assert_eq!(stocks[2 * N_PRODUCTS + WHEAT], 0);
        assert_eq!(engine.privates[0].shed[WHEAT], 37);
        let mut after_units = engine.clone();
        let unit_only = decode_player_action_ids(&engine, 0, &units, &[0; MARKET_SLOTS]).unwrap();
        let passive =
            decode_player_action_ids(&engine, 1, &[4; MAX_OWN_UNITS], &[0; MARKET_SLOTS]).unwrap();
        after_units.step(&[unit_only, passive]).unwrap();
        assert_eq!(after_units.privates[0].shed[WHEAT], stocks[WHEAT] as i64);
    }

    #[test]
    fn catalog_boundaries_decode_to_expected_operations() {
        assert_eq!(token_str(&decode_unit_action(4)[0]), Some("PASS"));
        assert_eq!(token_str(&decode_unit_action(5)[0]), Some("PICKUP"));
        assert_eq!(token_str(&decode_unit_action(245)[0]), Some("DROP"));
        assert_eq!(token_str(&decode_unit_action(246)[0]), Some("PLACE"));
        assert_eq!(token_str(&decode_unit_action(486)[0]), Some("PLANT"));
        assert_eq!(token_str(&decode_unit_action(499)[0]), Some("CARE"));
    }

    #[test]
    fn fixed_v2_factor_labels_decode_quantities_and_all() {
        let mut engine = Engine::new(0, Config::default());
        engine.privates[0].shed[WHEAT] = 37;
        let mut types = [0; MAX_OWN_UNITS];
        let crops = [0; MAX_OWN_UNITS];
        let items = [0; MAX_OWN_UNITS];
        let mut counts = [0; MAX_OWN_UNITS];
        types[0] = 12;
        counts[0] = 12;
        let action =
            decode_fixed_unit_labels(&engine, 0, &types, &crops, &items, &counts, Vec::new())
                .unwrap();
        let farmer = action.farmer.unwrap();
        assert_eq!(token_str(&farmer[0]), Some("PICKUP"));
        assert_eq!(token_int(&farmer[2]), Some(37));

        counts[0] = 11;
        let action =
            decode_fixed_unit_labels(&engine, 0, &types, &crops, &items, &counts, Vec::new())
                .unwrap();
        assert_eq!(token_int(&action.farmer.unwrap()[2]), Some(12));
    }

    #[test]
    fn fixed_v2_market_history_matches_bucket_and_all_rules() {
        let mut engine = Engine::new(0, Config::default());
        engine.privates[0].shed[WHEAT] = 9;
        let market = vec![
            Some(item_quantity_tokens("SELL", WHEAT as i32, 9)),
            Some(item_quantity_tokens("BUY_SEED", 1, 79)),
            Some(tokens(&["HIRE"])),
            Some(tokens(&["BUY_LAND"])),
        ];
        let history = encode_fixed_market_history(&engine, 0, &market).unwrap();
        assert_eq!(history.quantities[0], 67);
        assert_eq!(history.quantities[N_PRODUCTS + 1], 64);
        assert_eq!(history.hire, 1);
        assert_eq!(history.buy_land, 1);
    }
}
