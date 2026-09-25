"""PPO over autoregressive route construction and full-season cash returns."""
from __future__ import annotations

import argparse
import hashlib
import json
import random
from pathlib import Path

from .paths import BASELINE, RUNS, add_kaggsim

add_kaggsim()

from kaggsim.serve import Serve, call_agent, load_agent, obs_for

from .controller import RouteController
from .features import FEATURE_SIZE, STATE_SIZE, SCHEMA, menu_features, state_features


def _load_baseline():
    return load_agent(str(BASELINE))


class Network:
    def __init__(self, torch):
        nn = torch.nn
        self.torch = torch
        self.encoder = nn.Sequential(nn.Linear(FEATURE_SIZE, 128), nn.Tanh())
        self.context = nn.Sequential(nn.Linear(STATE_SIZE, 128), nn.Tanh(),
                                     nn.Linear(128, 128), nn.Tanh())
        self.actor = nn.Sequential(nn.Linear(256, 128), nn.Tanh(), nn.Linear(128, 1))
        self.critic = nn.Sequential(nn.Linear(128, 64), nn.Tanh(), nn.Linear(64, 1))
        self.module = nn.ModuleDict({"encoder": self.encoder, "context": self.context,
                                     "actor": self.actor, "critic": self.critic})

    def __call__(self, features, mask, state):
        z, context = self.encoder(features), self.context(state)
        joined = self.torch.cat((z, context[:, None, :].expand(-1, z.shape[1], -1)), -1)
        logits = self.actor(joined).squeeze(-1).masked_fill(~mask, -1e9)
        return logits, self.critic(context).squeeze(-1)


def episode(srv, model, torch, seed, seat, device, training=True, takeover=0,
            rng=None, horizon=24, replan_interval=6, trace=None):
    state = srv.reset(seed)
    base = _load_baseline()
    opponent = _load_baseline()
    rows = []
    rng = rng or random.Random(seed + seat)

    def choose(plan, routes):
        feat, mask = menu_features(plan, routes)
        context = state_features(plan)
        if model is None:
            choice = 0 if not training else rng.randrange(len(routes))
            return choice
        packed = torch.tensor(feat, dtype=torch.float32)
        packed_mask = torch.tensor(mask, dtype=torch.bool)
        packed_state = torch.tensor(context, dtype=torch.float32)
        x = packed.unsqueeze(0).to(device)
        m = packed_mask.unsqueeze(0).to(device)
        with torch.no_grad():
            logits, value = model(x, m, packed_state.unsqueeze(0).to(device))
            dist = torch.distributions.Categorical(logits=logits)
            idx = dist.sample() if training else logits.argmax(dim=-1)
            logp = dist.log_prob(idx)
        if training:
            rows.append({"state": packed_state, "features": packed, "mask": packed_mask, "action": int(idx.item()),
                         "logp": float(logp.item()), "value": float(value.item()),
                         "reward": 0.0})
        return int(idx.item())

    controller = RouteController(base, choose, takeover, horizon=horizon, replan_interval=replan_interval, trace=trace)
    while state["step"] < 719:
        view = obs_for(state, seat)
        other = obs_for(state, 1 - seat)
        own = call_agent(base if model is None and not training else controller.act, view)
        opposing = call_agent(opponent, other)
        before = state["farms"][seat]["money"] - state["farms"][1 - seat]["money"]
        state = srv.step2(own, opposing) if seat == 0 else srv.step2(opposing, own)
        after = state["farms"][seat]["money"] - state["farms"][1 - seat]["money"]
        if rows:
            rows[-1]["reward"] += (after - before) / 10000.0
    banks = (state["farms"][seat]["money"], state["farms"][1 - seat]["money"])
    return rows, banks, controller.stats


def _advantages(episodes, lam=1.0):
    samples = []
    for rows in episodes:
        gae = 0.0
        for i in range(len(rows) - 1, -1, -1):
            following = rows[i + 1]["value"] if i + 1 < len(rows) else 0.0
            delta = rows[i]["reward"] + following - rows[i]["value"]
            gae = delta + lam * gae
            rows[i]["advantage"] = gae
            rows[i]["return"] = gae + rows[i]["value"]
        samples.extend(rows)
    return samples


def update(model, optimizer, torch, episodes, device, epochs=2, batch_size=64):
    samples = _advantages(episodes)
    if not samples:
        return {"samples": 0}
    actions = torch.tensor([r["action"] for r in samples], device=device)
    old_logp = torch.tensor([r["logp"] for r in samples], device=device)
    advantages = torch.tensor([r["advantage"] for r in samples], device=device)
    returns = torch.tensor([r["return"] for r in samples], device=device)
    advantages = (advantages - advantages.mean()) / (advantages.std(unbiased=False) + 1e-8)
    losses = []
    for _ in range(epochs):
        for ids in torch.randperm(len(samples), device=device).split(batch_size):
            batch = [samples[i] for i in ids.tolist()]
            width = max(len(r["features"]) for r in batch)
            features = torch.zeros((len(batch), width, FEATURE_SIZE), device=device)
            masks = torch.zeros((len(batch), width), dtype=torch.bool, device=device)
            for i, row in enumerate(batch):
                size = len(row["features"])
                features[i, :size] = torch.as_tensor(row["features"], device=device)
                masks[i, :size] = torch.as_tensor(row["mask"], dtype=torch.bool, device=device)
            states = torch.stack([torch.as_tensor(r["state"], device=device) for r in batch])
            logits, values = model(features, masks, states)
            dist = torch.distributions.Categorical(logits=logits)
            ratio = (dist.log_prob(actions[ids]) - old_logp[ids]).exp()
            policy_loss = -torch.minimum(ratio * advantages[ids],
                ratio.clamp(0.8, 1.2) * advantages[ids]).mean()
            loss = policy_loss + 0.5 * (values - returns[ids]).square().mean() \
                   - 0.005 * dist.entropy().mean()
            optimizer.zero_grad(set_to_none=True)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.module.parameters(), 1.0)
            optimizer.step()
            losses.append(float(loss.item()))
    return {"samples": len(samples), "loss": sum(losses) / len(losses)}


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--episodes", type=int, default=100)
    parser.add_argument("--seed", type=int, default=1234)
    parser.add_argument("--out", type=Path, default=RUNS / "dynamic_v2")
    parser.add_argument("--eval-every", type=int, default=10)
    parser.add_argument("--takeover", type=int, default=0)
    parser.add_argument("--horizon", type=int, default=24)
    parser.add_argument("--replan-interval", type=int, default=6)
    parser.add_argument("--device", default="auto")
    parser.add_argument("--resume", type=Path)
    args = parser.parse_args()
    if args.episodes < 1 or args.eval_every < 1 or not 1 <= args.horizon <= 24 or args.replan_interval < 1:
        parser.error("episodes, eval-every and replan-interval must be positive; horizon must be 1..24")
    import torch
    torch.set_num_threads(1)
    torch.manual_seed(args.seed)
    random.seed(args.seed)
    device = "cuda" if args.device == "auto" and torch.cuda.is_available() else (
        "cpu" if args.device == "auto" else args.device)
    net = Network(torch)
    net.module.to(device)
    optimizer = torch.optim.Adam(net.module.parameters(), lr=3e-4)
    args.out.mkdir(parents=True, exist_ok=True)
    best = float("-inf")
    baseline_hash = hashlib.sha256(BASELINE.read_bytes()).hexdigest()
    if (args.out / "best.pt").exists():
        prior = torch.load(args.out / "best.pt", map_location="cpu", weights_only=False)
        if prior.get("schema") == SCHEMA and prior.get("baseline_sha256") == baseline_hash:
            best = float(prior["mean_margin"])
    first = 1
    if args.resume:
        checkpoint = torch.load(args.resume, map_location=device, weights_only=False)
        if checkpoint.get("schema") != SCHEMA:
            raise ValueError("legacy route-template checkpoint; retrain for dynamic-routes-v2")
        for key in ("takeover", "horizon", "replan_interval"):
            if checkpoint.get(key) != getattr(args, key):
                raise ValueError(f"--{key.replace('_', '-')} must match resumed run")
        if checkpoint.get("engine") != "kaggle-environments==1.32.7":
            raise ValueError("checkpoint engine pin does not match")
        if checkpoint.get("baseline_sha256") != baseline_hash:
            raise ValueError("baseline file differs from the resumed run")
        if int(checkpoint.get("seed", -1)) != args.seed:
            raise ValueError("--seed must match the resumed run")
        net.module.load_state_dict(checkpoint["network"])
        optimizer.load_state_dict(checkpoint["optimizer"])
        if "torch_rng" in checkpoint:
            torch.set_rng_state(checkpoint["torch_rng"].cpu())
        if device.startswith("cuda") and checkpoint.get("cuda_rng") is not None:
            torch.cuda.set_rng_state_all(checkpoint["cuda_rng"])
        first = int(checkpoint["iteration"]) + 1
    with Serve() as srv:
        for iteration in range(first, args.episodes + 1):
            batch, margins, diagnostics = [], [], []
            for seat in (0, 1):
                rows, banks, stats = episode(srv, net, torch, args.seed + iteration,
                                            seat, device, takeover=args.takeover,
                                            horizon=args.horizon, replan_interval=args.replan_interval)
                batch.append(rows)
                diagnostics.append(stats)
                margins.append(banks[0] - banks[1])
            result = update(net, optimizer, torch, batch, device)
            result.update(iteration=iteration, seed=args.seed + iteration,
                          margins=margins, device=device, routes=diagnostics, schema=SCHEMA)
            with (args.out / "metrics.jsonl").open("a", encoding="utf-8") as fh:
                fh.write(json.dumps(result) + "\n")
            print(json.dumps(result), flush=True)
            torch.save({"schema": SCHEMA, "takeover": args.takeover,
                        "horizon": args.horizon, "replan_interval": args.replan_interval, "network": net.module.state_dict(), "optimizer": optimizer.state_dict(),
                        "iteration": iteration, "seed": args.seed,
                        "torch_rng": torch.get_rng_state(),
                        "cuda_rng": torch.cuda.get_rng_state_all() if device.startswith("cuda") else None,
                        "baseline_sha256": baseline_hash,
                        "engine": "kaggle-environments==1.32.7"}, args.out / "latest.pt")
            if iteration % args.eval_every == 0 or iteration == args.episodes:
                scores = []
                for seed in (9001, 9002):
                    for seat in (0, 1):
                        _, banks, _ = episode(srv, net, torch, seed, seat, device,
                                              training=False, takeover=args.takeover,
                                              horizon=args.horizon, replan_interval=args.replan_interval)
                        scores.append(banks[0] - banks[1])
                mean = sum(scores) / len(scores)
                with (args.out / "eval.jsonl").open("a", encoding="utf-8") as fh:
                    fh.write(json.dumps({"iteration": iteration, "margins": scores,
                                         "mean_margin": mean}) + "\n")
                if mean > best:
                    best = mean
                    torch.save({"schema": SCHEMA, "takeover": args.takeover,
                        "horizon": args.horizon, "replan_interval": args.replan_interval, "network": net.module.state_dict(),
                                "iteration": iteration, "mean_margin": mean,
                                "baseline_sha256": baseline_hash,
                                "engine": "kaggle-environments==1.32.7"},
                               args.out / "best.pt")


if __name__ == "__main__":
    main()
