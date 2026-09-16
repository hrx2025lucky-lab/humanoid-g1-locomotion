"""CPU regressions supplementing the untouched course tests.

The course toy clip (5 frames / 2 bins) does not expose truncated bin widths.
These checks exercise real-clip tail coverage, reset subsets and tiny clips.
"""
from __future__ import annotations

import importlib.util
from pathlib import Path
import sys
from types import SimpleNamespace
import unittest
from unittest.mock import patch

import torch


ROOT = Path(__file__).resolve().parent


def load_module(name, path):
    spec = importlib.util.spec_from_file_location(name, path)
    module = importlib.util.module_from_spec(spec)
    sys.modules[name] = module
    spec.loader.exec_module(module)
    return module


def load_routes():
    routes = {}
    for route in ("mjlab", "isaaclab"):
        helper = load_module(
            f"boundary_helper_{route}", ROOT / f"bymic_commands_{route}_test.py"
        )
        getattr(helper, f"install_{route}_stubs")()
        routes[route] = load_module(
            f"boundary_target_{route}", ROOT / f"bymic_commands_{route}_todo.py"
        )
    return routes


def make_state(total, bins):
    return SimpleNamespace(
        device="cpu",
        motion=SimpleNamespace(time_step_total=total),
        bin_count=bins,
        time_steps=torch.tensor([0, 17, 0], dtype=torch.long),
        _env=SimpleNamespace(
            termination_manager=SimpleNamespace(
                terminated=torch.tensor([False, False, False])
            )
        ),
        bin_failed_count=torch.zeros(bins),
        _current_bin_failed=torch.zeros(bins),
        cfg=SimpleNamespace(adaptive_uniform_ratio=0.1, adaptive_kernel_size=1),
        kernel=torch.ones(1),
        metrics={
            name: torch.zeros(3)
            for name in ("sampling_entropy", "sampling_top1_prob", "sampling_top1_bin")
        },
    )


class SamplingBoundaries(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.routes = load_routes()

    def sample(self, module, state, bins, offsets):
        def choose(probabilities, count, replacement):
            self.assertTrue(torch.isfinite(probabilities).all())
            self.assertTrue((probabilities >= 0).all())
            self.assertAlmostEqual(float(probabilities.sum()), 1.0, places=6)
            self.assertEqual(count, 2)
            self.assertTrue(replacement)
            return torch.tensor(bins, dtype=torch.long)

        with patch.object(torch, "multinomial", side_effect=choose), patch.object(
            module, "sample_uniform", return_value=torch.tensor(offsets)
        ):
            module.MotionCommand._adaptive_sampling(state, torch.tensor([0, 2]))
        self.assertEqual(int(state.time_steps[1]), 17, "unselected env was changed")
        self.assertEqual(state.time_steps.dtype, torch.long)
        self.assertTrue((state.time_steps[[0, 2]] >= 0).all())
        self.assertTrue((state.time_steps[[0, 2]] < state.motion.time_step_total).all())

    def test_real_clip_tail_can_be_sampled(self):
        for route, module in self.routes.items():
            with self.subTest(route=route):
                state = make_state(6574, 132)
                self.sample(module, state, [131, 131], [0.99, 0.999])
                self.assertTrue((state.time_steps[[0, 2]] >= 6572).all())

    def test_nondivisible_small_clip_reaches_final_interval(self):
        for route, module in self.routes.items():
            with self.subTest(route=route):
                state = make_state(11, 3)
                self.sample(module, state, [0, 2], [0.5, 0.99])
                self.assertLess(int(state.time_steps[0]), 3)
                self.assertGreaterEqual(int(state.time_steps[2]), 9)

    def test_single_frame_and_single_bin_are_finite(self):
        for route, module in self.routes.items():
            with self.subTest(route=route):
                state = make_state(1, 1)
                self.sample(module, state, [0, 0], [0.0, 0.999])
                self.assertEqual(state.time_steps[[0, 2]].tolist(), [0, 0])
                self.assertTrue(all(torch.isfinite(v).all() for v in state.metrics.values()))

    def test_failure_counts_use_only_selected_failed_environments(self):
        for route, module in self.routes.items():
            with self.subTest(route=route):
                state = make_state(6574, 132)
                state.time_steps[:] = torch.tensor([6573, 17, 0])
                state._env.termination_manager.terminated[:] = torch.tensor([True, True, False])
                self.sample(module, state, [0, 131], [0.1, 0.9])
                self.assertEqual(float(state._current_bin_failed.sum()), 1.0)
                self.assertEqual(float(state._current_bin_failed[-1]), 1.0)
                state._env.termination_manager.terminated[:] = False
                self.sample(module, state, [0, 131], [0.1, 0.9])
                self.assertEqual(float(state._current_bin_failed.sum()), 0.0)


if __name__ == "__main__":
    unittest.main(verbosity=2)
