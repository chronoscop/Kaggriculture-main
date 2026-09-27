//! Diagnostic settlement accounting only; never used as private/future policy input.
use kagg_engine::{
    engine::{self, PlayerAction, UnitAction},
    json::Json,
    state::{State, MAX_MARKET_ORDERS, PRODUCTS},
};
#[derive(Clone, Default, Debug)]
pub struct TradeStats {
    pub revenue: [f64; 9],
    pub units: [i64; 9],
    pub spending: f64,
    pub early_spending: f64,
}
/// Replay just the unit and market phases on a copy to measure actual filled sales.
/// The real simulator still performs the sole transition of the live state.
pub fn observe_market(state: &State, actions: &[PlayerAction; 2], stats: &mut [TradeStats; 2]) {
    if actions.iter().all(|a| a.market.is_empty()) {
        return;
    }
    let mut s = state.clone();
    for p in 0..2 {
        let us: Vec<_> = std::iter::once(&actions[p].farmer)
            .chain(actions[p].hands.iter())
            .collect();
        let mut demand = std::collections::BTreeMap::<String, i64>::new();
        for a in &us {
            if a.op == "PLANT" && !a.item.is_empty() {
                *demand.entry(a.item.clone()).or_default() += 1;
            }
        }
        let blocked: Vec<_> = demand
            .iter()
            .filter(|(c, n)| **n > s.private[p].seeds.get(c))
            .map(|(c, _)| c.clone())
            .collect();
        for (i, a) in us.into_iter().enumerate() {
            let pass = UnitAction {
                op: "PASS".into(),
                ..Default::default()
            };
            let a = if a.op == "PLANT" && blocked.contains(&a.item) {
                &pass
            } else {
                a
            };
            engine::apply_unit_action(&mut s.farms[p], &mut s.private[p], i, a, state.step / 24);
        }
    }
    let n = actions
        .iter()
        .map(|a| a.market.len().min(MAX_MARKET_ORDERS))
        .max()
        .unwrap_or(0);
    for i in 0..n {
        let one = std::array::from_fn(|p| PlayerAction {
            market: actions[p].market.get(i).cloned().into_iter().collect(),
            ..Default::default()
        });
        let cash = [s.farms[0].money, s.farms[1].money];
        let inventory: [Vec<i64>; 2] =
            std::array::from_fn(|p| PRODUCTS.iter().map(|x| s.private[p].shed.get(x)).collect());
        engine::process_market(&mut s, &one);
        for p in 0..2 {
            if let Some(o) = one[p].market.first() {
                let delta = s.farms[p].money - cash[p];
                if o.first().is_some_and(|x| x == "SELL") {
                    if let Some(j) = o.get(1).and_then(|x| PRODUCTS.iter().position(|y| x == y)) {
                        stats[p].revenue[j] += delta;
                        stats[p].units[j] += inventory[p][j] - s.private[p].shed.get(PRODUCTS[j]);
                    }
                } else {
                    stats[p].spending -= delta;
                    if state.step < 12 * 24 {
                        stats[p].early_spending -= delta;
                    }
                }
            }
        }
    }
}
#[derive(Clone, Debug)]
pub struct Profile {
    pub values: Vec<f64>,
}
impl Profile {
    pub fn from_collection(c: &super::rollout::Collection) -> Self {
        let mut v = vec![0.; 11];
        for g in &c.games {
            let t = &g.trade_stats[g.learner];
            let total = t.revenue.iter().sum::<f64>().max(1.);
            for j in 0..9 {
                v[j] += t.revenue[j] / total;
            }
            v[9] += t.early_spending / t.spending.max(1.);
            let s = &g.agents[g.learner].stats;
            v[10] += s.mixed_routes as f64 / s.routes.max(1) as f64;
        }
        for x in &mut v {
            *x /= c.games.len().max(1) as f64;
        }
        Self { values: v }
    }
    pub fn distance(&self, other: &Self) -> f64 {
        0.35 * (0..9)
            .map(|i| (self.values[i] - other.values[i]).abs())
            .sum::<f64>()
            + 0.15 * (self.values[9] - other.values[9]).abs()
            + 0.15 * (self.values[10] - other.values[10]).abs()
    }
    pub fn json(&self) -> Json {
        Json::Arr(self.values.iter().copied().map(Json::Num).collect())
    }
    pub fn parse(j: &Json) -> Result<Option<Self>, String> {
        if j.is_null() {
            return Ok(None);
        }
        if j.arr().len() != 11
            || j.arr().iter().any(|x| {
                !x.is_num() || !x.f64().is_finite() || !(0. ..=1.000001).contains(&x.f64())
            })
        {
            return Err("invalid behavior profile".into());
        }
        Ok(Some(Self {
            values: j.arr().iter().map(Json::f64).collect(),
        }))
    }
}
