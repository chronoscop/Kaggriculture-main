"""Engine constants (kaggle-environments 1.32.7, default configuration)."""

BOARD = 10
TURNS_PER_DAY = 24
EPISODE_STEPS = 720
#: Terminal step: actions are solicited for steps 0..FINAL_STEP-1 (719
#: actions) and the final bank is read from the step-FINAL_STEP state.
FINAL_STEP = EPISODE_STEPS - 1
N_ACTIONS = FINAL_STEP
SHED_CAP = 100
MAX_MARKET_ORDERS = 10
STARTING_MONEY = 3000
TOWN_SHOP_UNLOCK_INTERVAL = 3

PRODUCTS = ["WHEAT", "CARROT", "TOMATO", "STRAWBERRY", "MELON",
            "EGG", "MILK", "WOOL", "FERTILIZER"]
CROPS = ["WHEAT", "CARROT", "TOMATO", "STRAWBERRY", "MELON"]
ANIMALS = ["GOOSE", "COW", "SHEEP"]
SEED_COST = {"WHEAT": 10, "CARROT": 20, "TOMATO": 50, "STRAWBERRY": 100,
             "MELON": 80}
ANIMAL_COST = {"GOOSE": 300, "COW": 400, "SHEEP": 500}
SHOPS_SORTED = ["BAKERY", "BRUNCH_SPOT", "FARMERS_MARKET", "ICE_CREAM_SHOP",
                "PET_CAFE", "PIZZA_SHOP", "SMOOTHIE_SHOP", "YARN_STORE"]

MOVES = ["NORTH", "SOUTH", "EAST", "WEST"]
#: Unit (farmer / hand) ops and their argument shapes.
UNIT_OPS = {
    "PASS": 0, "NORTH": 0, "SOUTH": 0, "EAST": 0, "WEST": 0, "WATER": 0,
    "HARVEST": 0, "DROP": 0, "FEED": 0, "CARE": 0, "COLLECT_FERTILIZER": 0,
    "FERTILIZE": 0, "DIG": 0, "BUILD_COOP": 0, "BUILD_PASTURE": 0,
    "PLANT": 1, "PICKUP": 2, "PLACE": (1, 2),
}
#: Market ops and their argument counts.
MARKET_OPS = {"BUY_SEED": 2, "SELL": 2, "BUY_PRODUCT": 2, "BUY_ANIMAL": 2,
              "HIRE": 0, "BUY_LAND": 0}

#: The first shop unlocks at the end of day 2 (the step 71 -> 72 transition)
#: and the second at the end of day 5 (143 -> 144). A world keyed on the
#: first shop is fixed by actions before step 72; both shops by step 144.
#: Changes to actions at or after these splits cannot change that key.
FIRST_SHOP_STEP = 3 * TURNS_PER_DAY
SECOND_SHOP_STEP = 6 * TURNS_PER_DAY
