//! Project-owned official-prefix semantics; no heuristic rewriting or SELL clamping.
use crate::catalog::decode_unit_action;
use crate::engine::{Engine, apply_unit_action};
use crate::state::*;

pub const OWNED_MARKET_ACTIONS: usize = 1903;

pub fn market_tokens(identity: usize) -> Result<TokenList, &'static str> {
    let tokens = match identity {
        0 => vec![Token::Str("NOOP".into())],
        1..=500 => quantity("BUY_SEED", (identity - 1) / 100, (identity - 1) % 100 + 1),
        501..=700 => quantity("BUY_PRODUCT", if identity < 601 { WHEAT } else { FERTILIZER }, (identity - 501) % 100 + 1),
        701..=1000 => quantity("BUY_ANIMAL", FIRST_ANIMAL + (identity - 701) / 100, (identity - 701) % 100 + 1),
        1001 => vec![Token::Str("HIRE".into())],
        1002 => vec![Token::Str("BUY_LAND".into())],
        1003..=1902 => quantity("SELL", (identity - 1003) / 100, (identity - 1003) % 100 + 1),
        _ => return Err("owned absolute market action id is out of range"),
    };
    Ok(Some(tokens.into()))
}

fn quantity(op: &str, item: usize, quantity: usize) -> Vec<Token> {
    vec![Token::Str(op.into()), Token::Str(ITEM_NAMES[item].into()), Token::Int(quantity as i64)]
}

pub fn exact_action(engine: &Engine, player: usize, units: &[i32], markets: &[i32]) -> Result<PlayerAction, &'static str> {
    let count = 1 + engine.farms[player].hands.len();
    if count > 20 || units.len() != 20 || markets.len() != 10 || units.iter().any(|&id| !(0..500).contains(&id)) {
        return Err("exact actions require represented units and catalog ids");
    }
    let mut decoded: Vec<TokenList> = units.iter().take(count).map(|&id| Some(decode_unit_action(id))).collect();
    let farmer = decoded.remove(0);
    let market = markets.iter().map(|&id| usize::try_from(id).map_err(|_| "negative market id").and_then(market_tokens)).collect::<Result<_, _>>()?;
    Ok(PlayerAction { farmer, hands: decoded, market })
}

#[derive(Clone)]
pub struct PrefixState {
    pub engine: Engine,
    pub player: usize,
    pub unit_count: usize,
}

impl PrefixState {
    pub fn new(engine: &Engine, player: usize) -> Result<Self, &'static str> {
        let unit_count = 1 + engine.farms[player].hands.len();
        let total = 2 + engine.farms[0].hands.len() + engine.farms[1].hands.len();
        if unit_count > 20 || total > 40 { return Err("PPO observation exceeds fixed unit capacity; truncation is forbidden"); }
        let mut engine = engine.clone();
        // Conditional supports never consult the other seat's hidden state.
        engine.privates[1-player] = Private::new();
        Ok(Self { engine, player, unit_count })
    }

    pub fn apply_unit(&mut self, index: usize, identity: i32) -> Result<(), &'static str> {
        if index >= self.unit_count || !(0..500).contains(&identity) { return Err("invalid prefix unit action"); }
        let day = self.engine.day_hour().0;
        apply_unit_action(&mut self.engine.farms[self.player], &mut self.engine.privates[self.player], index,
            &Some(decode_unit_action(identity)), &[false; N_CROPS], self.engine.cfg.board_size,
            day, self.engine.cfg.turns_per_day, self.engine.cfg.shed_capacity);
        Ok(())
    }

    pub fn unit_support(&self, index: usize) -> Result<Vec<bool>, &'static str> {
        let mut mask = vec![false; 500];
        mask[4] = true;
        if index >= self.unit_count { return Ok(mask); }
        // Quantity stems have the same effect/no-effect support under official limiting.
        for identity in 0..500 {
            let representative = match identity { 5..=244 => 5 + ((identity-5)/20)*20, 246..=485 => 246 + ((identity-246)/20)*20, _ => identity };
            if representative < identity { mask[identity as usize] = mask[representative as usize]; continue; }
            if identity == 4 { continue; }
            let mut trial = self.clone();
            trial.apply_unit(index, identity)?;
            mask[identity as usize] = trial.engine.farms[self.player] != self.engine.farms[self.player]
                || trial.engine.privates[self.player] != self.engine.privates[self.player];
        }
        Ok(mask)
    }

    pub fn apply_market(&mut self, identity: usize) -> Result<(), &'static str> {
        if identity == 0 { return Ok(()); }
        let mut actions = [PlayerAction::empty(), PlayerAction::empty()];
        actions[self.player].market = vec![market_tokens(identity)?];
        self.engine.process_market(&actions);
        Ok(())
    }

    pub fn market_support(&self) -> Result<Vec<bool>, &'static str> {
        let mut mask = vec![false; OWNED_MARKET_ACTIONS];
        mask[0] = true;
        for identity in 1..OWNED_MARKET_ACTIONS {
            let representative = match identity { 1..=1000 => 1 + ((identity-1)/100)*100, 1003..=1902 => 1003 + ((identity-1003)/100)*100, _ => identity };
            if representative < identity { mask[identity] = mask[representative]; }
            else if identity != 1001 || 1 + self.engine.farms[self.player].hands.len() < 20 {
                let mut trial = self.clone();
                trial.apply_market(identity)?;
                mask[identity] = trial.engine.farms[self.player] != self.engine.farms[self.player]
                    || trial.engine.privates[self.player] != self.engine.privates[self.player];
            }
            if identity >= 1003 {
                let item = (identity-1003)/100;
                let amount = (identity-1003)%100+1;
                mask[identity] &= amount as i64 <= self.engine.privates[self.player].shed[item];
            }
        }
        Ok(mask)
    }
}
