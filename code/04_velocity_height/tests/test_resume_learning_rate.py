"""CPU PPO update equivalence across the actual runner checkpoint boundary."""

from types import SimpleNamespace

import pytest
import torch
from rsl_rl.algorithms import PPO
from rsl_rl.models import MLPModel
from rsl_rl.storage import RolloutStorage
from tensordict import TensorDict

from mjlab.rl.runner import MjlabOnPolicyRunner


def make_runner():
  obs = TensorDict({"policy": torch.zeros(4, 3)})
  groups = {"actor": ["policy"], "critic": ["policy"]}
  actor = MLPModel(
    obs, groups, "actor", 2, hidden_dims=[8], activation="elu",
    distribution_cfg={"class_name": "GaussianDistribution", "init_std": 1.0},
  )
  critic = MLPModel(obs, groups, "critic", 1, hidden_dims=[8], activation="elu")
  storage = RolloutStorage("rl", 4, 4, obs, [2])
  runner = MjlabOnPolicyRunner.__new__(MjlabOnPolicyRunner)
  runner.alg = PPO(actor, critic, storage, num_learning_epochs=1,
                   num_mini_batches=1, schedule="adaptive", learning_rate=1e-3)
  runner.env = SimpleNamespace(unwrapped=SimpleNamespace(common_step_counter=0))
  runner.cfg = {"upload_model": False}
  runner.current_learning_iteration = 0
  return runner


def update(alg):
  torch.manual_seed(731)
  obs = TensorDict({"policy": torch.randn(4, 3)})
  for _ in range(4):
    alg.act(obs)
    alg.process_env_step(obs, torch.arange(4, dtype=torch.float32),
                         torch.zeros(4, dtype=torch.bool), {})
  alg.compute_returns(obs)
  # A small, known positive KL exercises the scheduler rather than its KL=0
  # no-op path. This represents a slightly older behavior policy mean.
  alg.storage.distribution_params[0].add_(0.02)
  return alg.update()


@pytest.mark.parametrize("load_cfg", [None, {
  "actor": True, "critic": True, "optimizer": True, "iteration": True,
}])
def test_full_resume_matches_next_adaptive_update(tmp_path, load_cfg):
  original = make_runner()
  update(original.alg)  # Populate real Adam moments.
  original.alg.learning_rate = 4e-5
  original.alg.optimizer.param_groups[0]["lr"] = 4e-5
  original.current_learning_iteration = 2999
  original.env.unwrapped.common_step_counter = 72000
  checkpoint = str(tmp_path / "parent.pt")
  original.save(checkpoint)

  resumed = make_runner()
  resumed.load(checkpoint, load_cfg=load_cfg, map_location="cpu")
  assert resumed.current_learning_iteration == 2999
  assert resumed.env.unwrapped.common_step_counter == 72000
  update(original.alg)
  update(resumed.alg)
  assert resumed.alg.learning_rate == original.alg.learning_rate
  assert resumed.alg.learning_rate == pytest.approx(6e-5)
  for key, value in original.alg.actor.state_dict().items():
    torch.testing.assert_close(resumed.alg.actor.state_dict()[key], value, rtol=0, atol=0)
  for key, value in original.alg.critic.state_dict().items():
    torch.testing.assert_close(resumed.alg.critic.state_dict()[key], value, rtol=0, atol=0)


def test_weights_only_keeps_new_learning_rate_and_curriculum(tmp_path):
  original = make_runner()
  original.alg.optimizer.param_groups[0]["lr"] = 4e-5
  original.env.unwrapped.common_step_counter = 72000
  original.current_learning_iteration = 2999
  checkpoint = str(tmp_path / "parent.pt")
  original.save(checkpoint)
  fresh = make_runner()
  fresh.load(checkpoint, {"actor": True, "critic": True,
                         "optimizer": False, "iteration": False})
  assert fresh.alg.learning_rate == 1e-3
  assert fresh.alg.optimizer.param_groups[0]["lr"] == 1e-3
  assert fresh.current_learning_iteration == 0
  assert fresh.env.unwrapped.common_step_counter == 0


def test_legacy_checkpoint_without_environment_state(tmp_path):
  original = make_runner()
  saved = original.alg.save()
  saved.update(iter=12, infos=None)
  checkpoint = str(tmp_path / "legacy.pt")
  torch.save(saved, checkpoint)
  fresh = make_runner()
  fresh.env.unwrapped.common_step_counter = 7
  fresh.load(checkpoint)
  assert fresh.env.unwrapped.common_step_counter == 7
  assert fresh.current_learning_iteration == 12
