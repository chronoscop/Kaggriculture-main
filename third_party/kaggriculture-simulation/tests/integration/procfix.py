"""Custom feature / label functions used by the processor tests."""


def feats(sample):
    obs = sample.get("obs") or {}
    return {"n_shops": len(obs.get("town", {}).get("unlocked_shops", [])),
            "custom_step": sample["step"]}


def labs(sample):
    return {"doubled": 2 * sample["labels"]["outcome"]}


def make_agent(bias=0, game_seed=None):
    from kaggsim.policies import RandomPolicy
    return RandomPolicy(int(game_seed or 0) + bias)


def exiting():
    def agent(obs):
        raise SystemExit(3)
    return agent
