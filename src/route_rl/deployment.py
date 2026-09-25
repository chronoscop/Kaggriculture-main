"""Submission-only CPU inference. Runtime imports only the Python stdlib."""
from __future__ import annotations
from array import array
import ctypes
import hashlib
import json
import math
from pathlib import Path
import platform
import sys
import time

from .controller import RouteController
from .features import FEATURE_SIZE, STATE_SIZE, SCHEMA, menu_features, state_features

class NativePolicy:
    def __init__(self, directory):
        directory = Path(directory)
        self.metadata = json.loads((directory / "metadata.json").read_text())
        if self.metadata["schema"] != SCHEMA:
            raise ValueError("incompatible policy schema")
        if sys.platform != "linux" or platform.machine() != "x86_64" or sys.byteorder != "little":
            raise RuntimeError("this artifact requires Linux x86_64 little-endian")
        raw = (directory / "weights.bin").read_bytes()
        if hashlib.sha256(raw).hexdigest() != self.metadata["weights_sha256"]:
            raise ValueError("policy weight checksum mismatch")
        self.weights = array("f")
        self.weights.frombytes(raw)
        self.library = ctypes.CDLL(str(directory / "inference.so"))
        self.library.route_weight_count.restype = ctypes.c_uint
        if self.weights.itemsize != 4 or len(self.weights) != self.library.route_weight_count():
            raise ValueError("unexpected policy weight layout")
        pointer = ctypes.POINTER(ctypes.c_float)
        self.library.route_scores.argtypes = [pointer, pointer, pointer, ctypes.c_size_t, pointer]
        self.library.route_scores.restype = None
        self.weight_buffer = (ctypes.c_float * len(self.weights)).from_buffer(self.weights)
        self.deadline = None
        self.budget_cutoffs = 0

    def scores(self, features, state):
        if len(state) != STATE_SIZE or not features or any(len(row) != FEATURE_SIZE for row in features):
            raise ValueError("invalid inference input shape")
        x = array("f", (v for row in features for v in row))
        s = array("f", state)
        out = (ctypes.c_float * len(features))()
        self.library.route_scores(self.weight_buffer,
            (ctypes.c_float * len(s)).from_buffer(s),
            (ctypes.c_float * len(x)).from_buffer(x), len(features), out)
        result = list(out)
        if not all(math.isfinite(v) for v in result):
            raise ValueError("non-finite policy output")
        return result

    def choose(self, plan, jobs):
        # Stop extending routes before the per-action budget is exhausted.
        # Already committed work remains executable; never invoke a baseline.
        if self.deadline is not None and time.perf_counter() >= self.deadline:
            self.budget_cutoffs += 1
            return 0  # END is always the first legal candidate.
        features, _ = menu_features(plan, jobs)
        scores = self.scores(features, state_features(plan))
        return max(range(len(scores)), key=scores.__getitem__)

class SubmissionAgent:
    def __init__(self, directory=None):
        self.policy = NativePolicy(directory or Path(__file__).resolve().parent)
        self.controllers = {}
        self.last_steps = {}

    def act(self, observation, configuration=None):
        seat, step = observation["player"], observation["step"]
        config = self.policy.metadata
        if seat not in self.controllers or step <= self.last_steps[seat]:
            self.controllers[seat] = RouteController(choose=self.policy.choose,
                horizon=config["horizon"], replan_interval=config["replan_interval"])
        self.last_steps[seat] = step
        seconds = config["planning_seconds"]
        self.policy.deadline = time.perf_counter() + seconds if seconds > 0 else None
        self.policy.budget_cutoffs = 0
        return self.controllers[seat].act(observation, configuration)
