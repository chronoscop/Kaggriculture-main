"""Offline checks for teacher identity, replay alignment and BC execution contracts."""
from copy import deepcopy
import gzip
import hashlib
import json
from pathlib import Path
import tempfile
from types import SimpleNamespace as NS
import unittest
from unittest.mock import patch

from route_rl.action_bc import CONTRACT, audit_cache, inspect_index, source_identity, train, validation_episode
from route_rl.replay_download import discover, download, teacher_seats, validate_replay
from route_rl.replay_rules import replay_seed
from route_rl.paths import PROJECT_ROOT


class Game:
    """Test fixture backed by the installed official rules, never the reference checkout."""
    def __init__(self, seed):
        from kaggle_environments import make
        self.environment = make("kaggriculture", configuration={"seed": seed}, debug=False)

    @property
    def privates(self):
        return [state.observation.private for state in self.environment.state]

    @property
    def done(self):
        return self.environment.done

    def observation(self, seat):
        observation = deepcopy(dict(self.environment.state[seat].observation))
        observation.setdefault("step", observation["day"] * 24 + observation["hour"])
        return observation

    def advance(self, actions):
        self.environment.step(actions)


def episode(identity, submissions):
    return NS(id=identity, state=NS(name="COMPLETED"), agents=[
        NS(submission_id=submission, index=seat, state=NS(name="EPISODE_AGENT_STATE_COMPLETE"))
        for seat, submission in submissions])


def replay_fixture():
    return {"module_version": "1.32.7", "name": "kaggriculture", "configuration": {"seed": 42},
            "steps": [[{"observation": {"step": turn, "player": seat, "farms": [], "private": {},
                                         "market": {}, "town": {}},
                        "action": None if turn == 0 else {"farmer": ["PASS"]},
                        "status": "DONE" if turn == 719 else "ACTIVE"}
                       for seat in (0, 1)] for turn in range(720)]}


class ReplayContracts(unittest.TestCase):
    def test_teacher_seat_comes_from_index_even_when_metadata_is_shuffled(self):
        self.assertEqual(teacher_seats(episode(1, [(1, 12), (0, 99)]), 12), [1])
        self.assertEqual(teacher_seats(episode(1, [(1, 12), (0, 12)]), 12), [1, 0])
        bad = episode(1, [(1, 12), (0, 99)])
        bad.agents[1].state.name = "EPISODE_AGENT_STATE_ERROR_TIMEOUT"
        self.assertEqual(teacher_seats(bad, 12), [])

    def test_completed_episode_with_unspecified_agent_states_is_checked_via_replay(self):
        game = episode(117485467, [(0, 56711476), (1, 56722220)])
        for agent in game.agents:
            agent.state.name = "EPISODE_AGENT_STATE_UNSPECIFIED"
        self.assertEqual(teacher_seats(game, 56722220), [1])
        self.assertEqual(teacher_seats(game, 56711476), [0])
        game.state.name = "CREATED"
        self.assertEqual(teacher_seats(game, 56722220), [])
        game.state.name = "COMPLETED"
        game.agents[0].state.name = "EPISODE_AGENT_STATE_PENDING"
        self.assertEqual(teacher_seats(game, 56722220), [])

    def test_replay_must_have_exact_pre_action_alignment_and_normal_terminal_state(self):
        good = replay_fixture()
        validate_replay(good)
        for mutation in ("shifted", "missing_action", "timeout", "version"):
            with self.subTest(mutation=mutation):
                replay = deepcopy(good)
                if mutation == "shifted":
                    replay["steps"][1][0]["observation"]["step"] = 0
                elif mutation == "missing_action":
                    replay["steps"][1][0]["action"] = None
                elif mutation == "timeout":
                    replay["steps"][-1][1]["status"] = "TIMEOUT"
                else:
                    replay["module_version"] = "1.32.6"
                with self.assertRaises(ValueError):
                    validate_replay(replay)

    def test_verified_public_version_uses_actual_day_hour_and_recorded_runtime_seed(self):
        replay = replay_fixture()
        replay["module_version"] = "1.33.0"
        replay["configuration"]["seed"] = None
        replay["info"] = {"seed": 1206603275}
        for turn, states in enumerate(replay["steps"]):
            observation = states[1]["observation"]
            observation.pop("step")
            observation.update(day=turn // 24, hour=turn % 24)
        validate_replay(replay)
        self.assertEqual(replay_seed(replay), 1206603275)
        self.assertEqual(replay["module_version"], "1.33.0")
        self.assertNotIn("step", replay["steps"][1][1]["observation"])
        replay["steps"][1][1]["observation"]["hour"] = 0
        with self.assertRaisesRegex(ValueError, "alignment mismatch"):
            validate_replay(replay)

    def test_download_retains_selected_losing_seat_and_resumes_without_duplicate_rows(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            teachers = root / "selected.json"
            teachers.write_text(json.dumps({"competition": "kaggriculture",
                                          "teachers": [{"submission_id": 12}, {"submission_id": 99}]}))
            game = episode(1, [(1, 12), (0, 99)])
            for agent in game.agents:
                agent.state.name = "EPISODE_AGENT_STATE_UNSPECIFIED"
            game.agents[0].reward = -1
            downloaded = []

            def fetch(identity, path, quiet):
                downloaded.append(identity)
                (Path(path) / f"episode-{identity}-replay.json").write_text(json.dumps(replay_fixture()))

            api = NS(competition_list_episodes=lambda _: [game], competition_episode_replay=fetch)
            with patch("route_rl.replay_download.call_api", side_effect=lambda fn, *a, **kw: fn(*a, **kw)):
                download(api, teachers, root / "data", 1, 0)
                first = (root / "data/teacher-seats.jsonl").read_text()
                download(api, teachers, root / "data", 1, 0)
            rows = [json.loads(line) for line in first.splitlines()]
            self.assertEqual(downloaded, [1])
            self.assertEqual({(r["submission_id"], r["seat"]) for r in rows}, {(12, 1), (99, 0)})
            self.assertEqual(first, (root / "data/teacher-seats.jsonl").read_text())

    def test_discover_freezes_best_active_submission_for_each_distinct_team(self):
        with tempfile.TemporaryDirectory() as directory:
            output = Path(directory) / "teachers.json"
            api = NS(competition_leaderboard_view=lambda *a, **kw: [NS(team_id=1, team_name="Teacher")],
                     competition_team_submissions=lambda _: [NS(id=11, public_score="100"),
                                                            NS(id=12, public_score="200")])
            with patch("route_rl.replay_download.call_api", side_effect=lambda fn, *a, **kw: fn(*a, **kw)):
                discover(api, "kaggriculture", 1, output)
                self.assertEqual(json.loads(output.read_text())["teachers"][0]["submission_id"], 12)
                with self.assertRaisesRegex(ValueError, "snapshot already exists"):
                    discover(api, "kaggriculture", 1, output)

    def test_inspection_holds_both_seats_out_and_rejects_conflicting_teachers(self):
        self.assertFalse(validation_episode(1))
        self.assertTrue(validation_episode(7))
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            for game in (1, 7):
                replay = replay_fixture()
                replay["configuration"]["seed"] = game
                (root / f"{game}.json.gz").write_bytes(gzip.compress(json.dumps(replay).encode()))
            rows = [{"path": f"{game}.json.gz", "episode_id": game, "seat": seat, "submission_id": 12}
                    for game in (1, 7) for seat in (0, 1)]
            index = root / "index.jsonl"
            index.write_text("".join(json.dumps(row) + "\n" for row in rows))
            self.assertEqual(inspect_index(index)["splits"], {"train": 2, "validation": 2})
            collision = replay_fixture()
            collision["configuration"]["seed"] = 1
            (root / "7.json.gz").write_bytes(gzip.compress(json.dumps(collision).encode()))
            with self.assertRaisesRegex(ValueError, "crosses train/validation"):
                inspect_index(index)
            with index.open("a") as stream:
                stream.write(json.dumps({**rows[0], "submission_id": 99}) + "\n")
            with self.assertRaisesRegex(ValueError, "conflicting teacher"):
                inspect_index(index)


class FullActionImplementationChecks(unittest.TestCase):
    """Run with project BC dependencies; these are CPU invariants, not match evidence."""

    def test_explicit_step_and_day_hour_prepare_identical_arrays_without_rewriting_version(self):
        import numpy as np
        from route_rl.replay_prepare import prepare_trajectory
        from route_rl.replay_prepare import prepare_trajectory

        game = Game(0)
        steps = [[{"observation": game.observation(seat), "action": None, "status": "ACTIVE"}
                  for seat in (0, 1)]]
        for turn in range(719):
            actions = [{"farmer": ["PASS"]}, {"farmer": ["EAST" if turn == 0 else "PASS"]}]
            game.advance(actions)
            steps.append([{"observation": game.observation(seat), "action": actions[seat],
                           "status": "DONE" if game.done else "ACTIVE"} for seat in (0, 1)])
        replay = {"module_version": "1.32.7", "steps": steps}
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            old = root / "old.json.gz"
            old.write_bytes(gzip.compress(json.dumps(replay).encode()))
            row = {"episode_id": 1, "seat": 1, "submission_id": 12,
                   "replay_path": old.name, "replay_sha256": hashlib.sha256(old.read_bytes()).hexdigest()}
            expected = prepare_trajectory((row, str(old), directory))
            replay["module_version"] = "1.33.0"
            for states in replay["steps"]:
                states[1]["observation"].pop("step")
            new = root / "new.json.gz"
            new.write_bytes(gzip.compress(json.dumps(replay).encode()))
            cache = root / "adapted"
            cache.mkdir()
            adapted = prepare_trajectory(({ "episode_id": 1, "seat": 1, "submission_id": 12},
                                          str(new), str(cache)))
            self.assertEqual(adapted["replay_module_version"], "1.33.0")
            with np.load(root / f"{expected['key']}.npz") as a, np.load(cache / f"{adapted['key']}.npz") as b:
                np.testing.assert_array_equal(a["features"], b["features"])
                np.testing.assert_array_equal(a["labels"], b["labels"])

    def test_cache_audit_checks_both_seat_splits_and_action_id_bounds(self):
        import numpy as np

        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            features = np.zeros((719, 264, 124), np.float16)
            labels = np.full((719, 30), -100, np.int16)
            labels[:, 0] = 4
            labels[:, 20:] = 0
            rows = [{"key": str(game), "episode_id": game, "seat": 0, "split": split}
                    for game, split in ((1, "train"), (7, "validation"))]
            for row in rows:
                np.savez_compressed(root / f"{row['key']}.npz", features=features, labels=labels)
            index = root / "index.json"
            index.write_text(json.dumps({"episodes": rows}))
            self.assertEqual(audit_cache(root)["samples"], {"train": 719, "validation": 719})
            labels[0, 0] = 500
            np.savez_compressed(root / "1.npz", features=features, labels=labels)
            with self.assertRaisesRegex(ValueError, "invalid unit action IDs"):
                audit_cache(root)
            labels[0, 0] = 4
            np.savez_compressed(root / "1.npz", features=features, labels=labels)
            rows[1]["split"] = "train"
            index.write_text(json.dumps({"episodes": rows}))
            with self.assertRaisesRegex(ValueError, "episode-hash split differs"):
                audit_cache(root)

    def test_resume_rejects_changed_initial_settings_or_lower_epoch_target(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            source = source_identity()
            cache = root / "cache"
            cache.mkdir()
            (cache / "pipeline.json").write_text(json.dumps({"contract": CONTRACT, "source": source}))
            initial = root / "initial.pkl"
            initial.write_bytes(b"initial")
            args = NS(cache=cache, initial=initial, out=root / "run", epochs=2, batch_size=2, compute_dtype="float32")
            with patch("route_rl.action_bc.audit_cache", return_value={"decoded_bytes": 0}), \
                    patch("route_rl.action_bc.run_training") as launch:
                train(args)
                args.epochs = 3
                train(args)
                self.assertEqual(launch.call_count, 2)
                args.epochs = 1
                with self.assertRaisesRegex(ValueError, "cannot decrease"):
                    train(args)
                args.epochs = 3
                args.batch_size = 4
                with self.assertRaisesRegex(ValueError, "resume source/data/settings mismatch"):
                    train(args)
                args.batch_size = 2
                initial.write_bytes(b"different initial")
                with self.assertRaisesRegex(ValueError, "resume source/data/settings mismatch"):
                    train(args)
                self.assertEqual(launch.call_count, 2)

    def test_resolver_filters_failed_work_and_labels_actual_sell_quantity(self):
        from route_rl.full_action.legality import action_legality
        from route_rl.full_action.sell_quantity import ABSOLUTE_START
        from route_rl.full_action.labels import label_actions

        game = Game(0)
        game.privates[0]["shed"]["WHEAT"] = 3
        observations = [game.observation(seat) for seat in (0, 1)]
        actions = [{"farmer": ["WATER"], "market": [["SELL", "WHEAT", 100]]}, {"farmer": ["PASS"]}]
        legal = action_legality(observations, actions, 0)
        labels = label_actions(observations[0], actions[0], legal)
        self.assertFalse(legal.unit_mask[0])
        self.assertEqual(labels[0], -100)
        self.assertEqual(legal.market_executed[0], 3)
        self.assertEqual(labels[20], ABSOLUTE_START + 2)
        self.assertTrue((labels[1:20] == -100).all())

    def test_full_trajectory_pairs_first_observation_with_next_recorded_action(self):
        import numpy as np
        from route_rl.full_action.catalog import UNIT_ACTION_TO_ID
        from route_rl.replay_prepare import prepare_trajectory

        game = Game(0)
        steps = [[{"observation": game.observation(seat), "action": None, "status": "ACTIVE"}
                  for seat in (0, 1)]]
        for turn in range(719):
            actions = [{"farmer": ["EAST" if turn == 0 else "PASS"]}, {"farmer": ["PASS"]}]
            game.advance(actions)
            steps.append([{"observation": game.observation(seat), "action": actions[seat],
                           "status": "DONE" if game.done else "ACTIVE"} for seat in (0, 1)])
        replay = {"module_version": "1.32.7", "steps": steps}
        validate_replay(replay)
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            source = root / "replay.json.gz"
            source.write_bytes(gzip.compress(json.dumps(replay).encode()))
            row = {"episode_id": 1, "seat": 0, "submission_id": 12,
                   "replay_path": source.name, "replay_sha256": hashlib.sha256(source.read_bytes()).hexdigest()}
            receipt = prepare_trajectory((row, str(source), directory))
            with np.load(root / f"{receipt['key']}.npz") as arrays:
                self.assertEqual(arrays["features"].shape, (719, 264, 124))
                self.assertEqual(arrays["labels"][0, 0], UNIT_ACTION_TO_ID[("EAST",)])
                self.assertEqual(arrays["labels"][1, 0], UNIT_ACTION_TO_ID[("PASS",)])
                self.assertEqual(arrays["labels"][-1, 0], UNIT_ACTION_TO_ID[("PASS",)])

    def test_feature_only_training_forward_matches_inference_batch(self):
        import jax
        import numpy as np
        from route_rl.full_action.inference import prepare_fixed_batch
        from route_rl.full_action.features import encode_observation
        from route_rl.full_action.model import JaxModelConfig, initialize_params, policy_forward

        model = JaxModelConfig(d_model=32, layers=1, heads=2, ffn_dim=64)
        params = initialize_params(jax.random.PRNGKey(0), model)
        observation = Game(0).observation(0)
        fixed = prepare_fixed_batch(encode_observation(observation), observation).arrays
        full = policy_forward(params, fixed, model)
        cached = policy_forward(params, {"features": fixed["features"]}, model)
        np.testing.assert_array_equal(full["unit_action"], cached["unit_action"])
        np.testing.assert_array_equal(full["market_action"], cached["market_action"])

    def test_project_trainer_saves_resumes_and_loads_candidate_policy(self):
        import numpy as np
        from route_rl.action_bc import initialize
        from route_rl.full_action.catalog import UNIT_ACTION_TO_ID
        from route_rl.full_action.checkpoints import load_training_source, policy_hash
        from route_rl.full_action.features import encode_observation
        from route_rl.full_action.inference import load_policy, prepare_fixed_batch
        from route_rl.full_action.trainer import run_training

        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            initial, cache, output = root / "initial.pkl", root / "cache", root / "run"
            initialize("smoke", initial, 0)
            before = load_training_source(initial)
            cache.mkdir()
            observation = Game(0).observation(0)
            features = prepare_fixed_batch(encode_observation(observation), observation).arrays["features"]
            labels = np.full((3, 30), -100, np.int16)
            labels[:, 0] = UNIT_ACTION_TO_ID[("PASS",)]
            labels[:, 20:] = 0
            rows = [{"key": str(game), "episode_id": game, "seat": 0, "split": split}
                    for game, split in ((1, "train"), (7, "validation"))]
            for row in rows:
                np.savez_compressed(cache / f"{row['key']}.npz", features=np.repeat(features, 3, axis=0), labels=labels)
            (cache / "index.json").write_text(json.dumps({"episodes": rows}))
            config = {"batch_per_gpu": 2, "epochs": 1, "unit_entropy": 0.1, "market_entropy": 0.1,
                      "teacher_kl": 0.05, "compute_dtype": "float32", "learning_rate": 1e-4,
                      "seed": 51, "save_epoch_policies": True}
            run_training(initial, cache, output, config)
            policy_path = output / "final_student_jax.pkl"
            learned = load_training_source(policy_path)
            self.assertNotEqual(policy_hash(before["params"]), policy_hash(learned["params"]))
            self.assertEqual(policy_hash(before["params"]["value"]), policy_hash(learned["params"]["value"]))
            self.assertEqual(policy_hash(before["params"]), policy_hash(load_training_source(initial)["params"]))
            receipt = json.loads((output / "receipt.json").read_text())
            self.assertFalse(receipt["value_training"])
            self.assertEqual(receipt["teacher_policy_sha256"], policy_hash(before["params"]))
            # Resuming a completed epoch must not silently perform another update.
            run_training(initial, cache, output, config)
            self.assertEqual(policy_hash(learned["params"]), policy_hash(load_training_source(policy_path)["params"]))
            self.assertEqual(len((output / "metrics.jsonl").read_text().splitlines()), 1)
            action = load_policy(policy_path)(observation)
            self.assertEqual(set(action), {"farmer", "hands", "market"})
            self.assertEqual(len(action["hands"]), len(observation["farms"][0]["hands"]))
            self.assertLessEqual(len(action["market"]), 10)
            # Reference/legacy payloads need an explicit conversion, never a silent resume.
            import pickle
            old = root / "old.pkl"
            old.write_bytes(pickle.dumps({"model_config": learned["model_config"], "params": learned["params"]}))
            with self.assertRaisesRegex(ValueError, "not a project full-action policy"):
                load_training_source(old)

    def test_bc_updates_actor_preserves_value_and_ignores_padding(self):
        import jax
        import jax.numpy as jnp
        import numpy as np
        import optax
        from route_rl.full_action.inference import prepare_fixed_batch
        from route_rl.full_action.model import JaxModelConfig, initialize_params, add_zero_value_head
        from route_rl.full_action.features import encode_observation
        from route_rl.full_action.bc_objective import make_steps
        from route_rl.full_action.checkpoints import policy_hash
        from route_rl.full_action.global_update import GlobalUpdate
        from route_rl.full_action.sharding import put_replicated

        model = JaxModelConfig(d_model=32, layers=1, heads=2, ffn_dim=64, rope_dim=16,
                               attention_backend="manual", rope_correction_backend="dense", absolute_sell=True)
        params = add_zero_value_head(initialize_params(jax.random.PRNGKey(0), model), model)
        teacher_hash = policy_hash(params)
        observation = Game(0).observation(0)
        fixed = prepare_fixed_batch(encode_observation(observation), observation).arrays
        batch = {key: np.repeat(value, 2, axis=0) for key, value in fixed.items()}
        batch.update(unit_action=np.full((2, 20), -100, np.int32),
                     market_action=np.zeros((2, 10), np.int32), sample_mask=np.array([1, 0], np.float32))
        batch["unit_action"][0, 0] = 4
        optimizer = optax.chain(optax.clip_by_global_norm(5), optax.adam(1e-4, eps=1e-5))
        step, _ = make_steps(model, jnp.float32, optimizer, 0.1, 0.1, 0.05)
        update = GlobalUpdate(step, jax.devices())
        devices = jax.local_devices()
        replicated = put_replicated(params, devices)
        state = put_replicated(optimizer.init(params), devices)

        def apply(values):
            arrays = jax.tree.map(lambda value: jnp.asarray(value)[None], values)
            result, _, metrics = update(replicated, state, replicated, arrays, True)
            host = jax.tree.map(lambda value: np.asarray(value)[0], result)
            return host, metrics

        learned, metrics = apply(batch)
        altered = deepcopy(batch)
        altered["unit_action"][1] = 499
        altered["market_action"][1] = 1902
        after_padding_change, _ = apply(altered)
        self.assertNotEqual(policy_hash(learned), teacher_hash)
        self.assertEqual(policy_hash(params), teacher_hash)
        self.assertEqual(policy_hash(learned["value"]), policy_hash(params["value"]))
        self.assertEqual(policy_hash(learned), policy_hash(after_padding_change))
        self.assertTrue(np.isfinite(np.asarray(metrics["loss"])).all())
        self.assertEqual(float(np.asarray(metrics["sample_count"]).item()), 1)


if __name__ == "__main__":
    unittest.main()
