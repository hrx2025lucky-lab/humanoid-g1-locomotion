"""Configuration changes used by the G1 AMP yaw-tracking experiment.

Apply after constructing G1AMPWalkToRunEnvCfg from unitree_lab_amp.
This component is a configuration fragment, not a standalone training entry.
The existing corrected reference-reset path and expert data remain required.
"""


def apply_tracking_tuning(env_cfg):
    """Prioritize yaw-rate tracking while preserving the existing AMP mixture.

    The comparison used weight 0.5 versus 2.0 at std=0.5 rad/s.
    Both comparison arms enabled external forces on every PhysX iteration.
    The physics flag is therefore a shared condition, not the reward ablation.
    """
    env_cfg.sim.physx.enable_external_forces_every_iteration = True
    env_cfg.rewards.track_ang_vel_z.weight = 2.0
    return env_cfg
