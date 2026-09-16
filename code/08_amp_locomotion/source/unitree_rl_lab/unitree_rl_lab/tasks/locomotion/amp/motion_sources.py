"""Compatibility exports for the relocated G1 motion selections."""

from .config.g1.motion_cfg import (  # noqa: F401
    G1DanceMotionCfg,
    G1DanceMotionSourceCfg,
    G1MixedMotionCfg,
    G1MixedMotionSourceCfg,
    G1MotionSourceCfg,
    G1OmniRunMotionCfg,
    G1OmniRunMotionSourceCfg,
    G1RunMotionCfg,
    G1RunMotionSourceCfg,
    G1WalkMotionCfg,
    G1WalkMotionSourceCfg,
    G1WalkToRunMotionCfg,
    G1WalkToRunMotionSourceCfg,
    G1_Run_Cfg,
    G1_dance_Cfg,
    G1_walk_Cfg,
    MOTION_CONFIGS,
    MotionSourceCfg,
    make_motion_source,
)

MOTION_SOURCE_TYPES = MOTION_CONFIGS
get_default_motion_source = make_motion_source
