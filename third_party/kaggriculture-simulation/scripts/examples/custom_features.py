"""An example custom feature extractor for ``kaggsim.processor``.

A feature function receives one sample record (with ``obs`` when the run
sets ``samples.include_obs``) and returns a dict merged into ``features``.
"""


def extract(sample):
    obs = sample.get("obs")
    if not obs:
        return {}
    me = obs["player"]
    farm = obs["farms"][me]
    ripe = sum(1 for row in farm["tiles"] for t in row
               if isinstance(t, dict) and t.get("kind") == "PLANT"
               and t.get("yield_units", 0) > 0)
    shed = obs["private"]["shed"]
    value = sum(shed.get(p, 0) * obs["market"]["prices"].get(p, 0)
                for p in obs["market"]["prices"])
    return {"ripe_tiles": ripe, "shed_market_value": value,
            "money_lead": farm["money"] - obs["farms"][1 - me]["money"]}
