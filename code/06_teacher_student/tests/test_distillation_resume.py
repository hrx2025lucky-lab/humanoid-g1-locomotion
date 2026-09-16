"""Exercise actual PPO save/load with CPU models and nonempty Adam state."""
import copy

import pytest
import torch

from humanoid_hw6.rl.algorithms.action_matching import ActionMatchingPPO
from humanoid_hw6.rl.algorithms.kl_matching import KlMatchingPPO


def make_algorithm(cls, updates=0):
  # Physics/rollout collection is irrelevant to the checkpoint contract.
  alg = cls.__new__(cls)
  alg.actor = torch.nn.Linear(2, 1)
  alg.critic = torch.nn.Linear(2, 1)
  alg.rnd = None
  alg.optimizer = torch.optim.Adam(
    list(alg.actor.parameters()) + list(alg.critic.parameters()), lr=3e-4
  )
  alg.learning_rate = 3e-4
  setattr(alg, alg.distillation_counter_name, updates)
  alg.bc_coef_start, alg.bc_coef_end, alg.bc_anneal_iters = 1., .05, 20_000
  alg.kl_coef_start, alg.kl_coef_min, alg.kl_coef_anneal_iters = .1, .01, 60_000
  return alg


@pytest.mark.parametrize('cls,coef,next_coef', [
  (ActionMatchingPPO, .8575, .8574525),
  (KlMatchingPPO, .0955, .0954985),
])
def test_resume_preserves_next_coefficient_optimizer_and_adaptive_lr(cls, coef, next_coef):
  before = make_algorithm(cls, 3000)
  loss = before.actor(torch.ones(3, 2)).square().mean()
  loss.backward()
  before.optimizer.step()
  before.learning_rate = 4e-5
  before.optimizer.param_groups[0]['lr'] = 4e-5
  saved = copy.deepcopy(before.save())
  saved['iter'] = 2999
  resumed = make_algorithm(cls)
  assert resumed.load(saved, None, True)
  assert resumed._current_distillation_coef() == pytest.approx(coef)
  assert resumed.learning_rate == 4e-5
  assert len(resumed.optimizer.state) > 0
  for actual, expected in zip(resumed.actor.parameters(), before.actor.parameters()):
    torch.testing.assert_close(actual, expected)
  for key, state in saved['optimizer_state_dict']['state'].items():
    torch.testing.assert_close(resumed.optimizer.state_dict()['state'][key]['exp_avg'], state['exp_avg'])
  resumed._on_distillation_update_end()
  assert resumed._current_distillation_coef() == pytest.approx(next_coef)


def test_weights_only_does_not_restore_schedule_or_optimizer():
  saved = make_algorithm(ActionMatchingPPO, 3000).save()
  fresh = make_algorithm(ActionMatchingPPO)
  assert not fresh.load(saved, {'actor': True, 'iteration': False, 'optimizer': False}, True)
  assert fresh.num_bc_updates == 0
  assert fresh.learning_rate == 3e-4


def test_legacy_fresh_run_compatibility_is_explicit():
  saved = make_algorithm(KlMatchingPPO, 3000).save()
  saved.pop('distillation_state')
  saved['iter'] = 2999
  resumed = make_algorithm(KlMatchingPPO)
  with pytest.warns(RuntimeWarning, match='Legacy checkpoint'):
    resumed.load(saved, None, True)
  assert resumed.num_kl_updates == 3000


def test_wrong_method_state_is_rejected():
  saved = make_algorithm(ActionMatchingPPO, 3000).save()
  with pytest.raises(ValueError, match='method does not match'):
    make_algorithm(KlMatchingPPO).load(saved, None, True)
