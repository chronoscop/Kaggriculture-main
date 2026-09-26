from dataclasses import dataclass, asdict
SCHEMA = "baseline-local-economy-v4"
ENGINE = "kaggle-environments==1.32.7"
CROPS = ("WHEAT", "CARROT", "TOMATO", "STRAWBERRY", "MELON")
COST = dict(zip(CROPS, (10, 20, 50, 100, 80)))
FIRST = dict(zip(CROPS, (2, 2, 8, 10, 10)))
PRODUCTS = CROPS + ("MILK", "WOOL", "EGG", "FERTILIZER")
SHOPS = ("BAKERY","PIZZA_SHOP","BRUNCH_SPOT","YARN_STORE","ICE_CREAM_SHOP","PET_CAFE","SMOOTHIE_SHOP","FARMERS_MARKET")
SAFE = ("PASS","PLANT","WATER","HARVEST","FERTILIZE","DIG")
@dataclass(frozen=True)
class Settings:
    start_day: int = 6
    max_changes: int = 1
    max_active: int = 1
    max_extra_cash: float = 300.
    initial_change_probability: float = .03
    def validate(self):
        if not 0 <= self.start_day < 30 or self.max_changes < 0 or self.max_active < 1:
            raise ValueError("invalid local decision limits")
        if self.max_extra_cash < 0 or not 0 < self.initial_change_probability < 1:
            raise ValueError("invalid exploration settings")
        return self
    def dict(self):
        return asdict(self)
