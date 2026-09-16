from __future__ import annotations

import os
from dataclasses import dataclass
from pathlib import Path


ROOT_DIR = Path(os.getenv("PYHOI_ROOT_DIR", "/home/sustech/unitree_project")).resolve()
G1_MUJOCO_DIR = ROOT_DIR / "unitree_mujoco" / "unitree_robots" / "g1"
DEFAULT_POLICY_DIR = (
    ROOT_DIR
    / "logs"
    / "rsl_rl"
    / "unitree_g1_29dof_mimic_hoi_terrain"
    / "2026-05-07_13-55-25"
)


def _env_bool(name: str, default: bool) -> bool:
    value = os.getenv(name)
    if value is None:
        return default
    return value.strip().lower() in {"1", "true", "yes", "on"}


def _env_path(name: str, default: Path) -> Path:
    return Path(os.getenv(name, str(default))).expanduser().resolve()


def _env_float(name: str, default: float) -> float:
    raw = os.getenv(name)
    if raw is None or raw.strip() == "":
        return default
    return float(raw)


def _env_vec3(name: str, default: tuple[float, float, float]) -> tuple[float, float, float]:
    raw = os.getenv(name)
    if raw is None or not raw.strip():
        return default
    tokens = raw.replace(",", " ").split()
    if len(tokens) != 3:
        raise ValueError(f"{name} must be three floats (x y z), got {raw!r}")
    return (float(tokens[0]), float(tokens[1]), float(tokens[2]))


def _env_optional_int(name: str) -> int | None:
    raw = os.getenv(name)
    if raw is None or not raw.strip():
        return None
    return int(raw)


def _env_optional_vec(name: str, expected_len: int) -> tuple[float, ...] | None:
    raw = os.getenv(name)
    if raw is None or not raw.strip():
        return None
    tokens = raw.replace(",", " ").split()
    if len(tokens) != expected_len:
        raise ValueError(f"{name} must have {expected_len} floats, got {raw!r}")
    return tuple(float(token) for token in tokens)


def _env_vec2(name: str, default: tuple[float, float]) -> tuple[float, float]:
    raw = os.getenv(name)
    if raw is None or not raw.strip():
        return default
    tokens = raw.replace(",", " ").split()
    if len(tokens) != 2:
        raise ValueError(f"{name} must be two floats (x y), got {raw!r}")
    return (float(tokens[0]), float(tokens[1]))


def _env_vec4(name: str, default: tuple[float, float, float, float]) -> tuple[float, float, float, float]:
    raw = os.getenv(name)
    if raw is None or not raw.strip():
        return default
    tokens = raw.replace(",", " ").split()
    if len(tokens) != 4:
        raise ValueError(f"{name} must be four floats (r g b a), got {raw!r}")
    return (float(tokens[0]), float(tokens[1]), float(tokens[2]), float(tokens[3]))


@dataclass(frozen=True)
class PyHOIConfig:
    root_dir: Path
    interface: str
    domain_id: int
    conda_env: str

    policy_dir: Path
    deploy_yaml: Path
    policy_onnx: Path
    motion_file: Path
    terrain_meta_file: Path
    terrain_urdf: Path

    base_scene: Path
    base_robot_xml: Path
    generated_scene: Path
    generated_robot_xml: Path
    scene_override: Path | None

    controller_launcher: Path
    use_joystick: bool
    enable_elastic_band: bool
    elastic_band_stiffness: float
    elastic_band_damping: float
    elastic_band_point: tuple[float, float, float]
    print_scene_information: bool
    simulate_dt: float
    viewer_dt: float
    startup_delay_sec: float
    start_paused: bool
    validate_cached_scene: bool
    terrain_collision_mode: str
    terrain_friction: tuple[float, ...] | None
    terrain_condim: int | None
    terrain_solref: tuple[float, ...] | None
    terrain_solimp: tuple[float, ...] | None
    terrain_priority: int | None
    terrain_box_margin: float
    floor_collision: bool
    init_pose_mode: str
    flat_init_xy_offset: tuple[float, float]
    vis_contact_points: bool
    vis_contact_forces: bool
    vis_contact_split: bool
    motion_ref_viz_enabled: bool
    motion_ref_phase_sync_enable: bool
    motion_ref_align_sim_at_mimic_start: bool
    motion_ref_phase_topic: str
    motion_ref_phase_stale_timeout: float
    motion_ref_fps: float
    motion_ref_dot_radius: float
    motion_ref_time_offset: float
    motion_ref_loop: bool
    motion_ref_log_hz: float
    motion_ref_color_ctrl: tuple[float, float, float, float]
    motion_ref_color_sim: tuple[float, float, float, float]

    @property
    def scene_path(self) -> Path:
        return self.scene_override or self.generated_scene

    @classmethod
    def from_env(cls, interface: str | None = None) -> "PyHOIConfig":
        policy_dir = _env_path("PYHOI_POLICY_DIR", DEFAULT_POLICY_DIR)
        scene_override_env = os.getenv("PYHOI_SCENE", "").strip()
        scene_override = Path(scene_override_env).expanduser().resolve() if scene_override_env else None
        _tcm = os.getenv("PYHOI_TERRAIN_COLLISION_MODE", "mesh").strip().lower()
        if _tcm not in {"mesh", "box", "primitive"}:
            raise ValueError(
                "PYHOI_TERRAIN_COLLISION_MODE must be 'mesh', 'box', or 'primitive', "
                f"got {os.getenv('PYHOI_TERRAIN_COLLISION_MODE', 'mesh')!r}"
            )
        terrain_collision_mode = "box" if _tcm == "primitive" else _tcm
        return cls(
            root_dir=ROOT_DIR,
            interface=interface or os.getenv("PYHOI_INTERFACE", "wlp4s0"),
            domain_id=int(os.getenv("PYHOI_DOMAIN_ID", "0")),
            conda_env=os.getenv("PYHOI_CONDA_ENV", "env_isaaclab"),
            policy_dir=policy_dir,
            deploy_yaml=_env_path("PYHOI_DEPLOY_YAML", policy_dir / "params" / "deploy.yaml"),
            policy_onnx=_env_path("PYHOI_POLICY_ONNX", policy_dir / "exported" / "policy.onnx"),
            motion_file=_env_path(
                "PYHOI_MOTION_FILE",
                ROOT_DIR / "logs" / "hoi_mimic_data" / "climb_15_z_scale_1.0_mimic.npz",
            ),
            terrain_meta_file=_env_path(
                "PYHOI_TERRAIN_META_FILE",
                ROOT_DIR / "logs" / "hoi_mimic_data" / "climb_15_z_scale_1.0_mimic.terrain.json",
            ),
            terrain_urdf=_env_path(
                "PYHOI_TERRAIN_URDF",
                ROOT_DIR
                / "motion_dataset"
                / "HOI"
                / "models"
                / "terrain"
                / "climb_15"
                / "multi_boxes_z_scale_1.0.urdf",
            ),
            base_scene=_env_path("PYHOI_BASE_SCENE", G1_MUJOCO_DIR / "scene_29dof.xml"),
            base_robot_xml=_env_path("PYHOI_BASE_ROBOT_XML", G1_MUJOCO_DIR / "g1_29dof.xml"),
            generated_scene=_env_path("PYHOI_GENERATED_SCENE", G1_MUJOCO_DIR / "scene_29dof_pyhoi.xml"),
            generated_robot_xml=_env_path(
                "PYHOI_GENERATED_ROBOT_XML", G1_MUJOCO_DIR / "g1_29dof_pyhoi_init.xml"
            ),
            scene_override=scene_override,
            controller_launcher=_env_path(
                "PYHOI_CONTROLLER_LAUNCHER", ROOT_DIR / "run_g1_hoi_mimic_terrain_sim2sim.sh"
            ),
            use_joystick=_env_bool("PYHOI_USE_JOYSTICK", False),
            enable_elastic_band=_env_bool("PYHOI_ENABLE_ELASTIC_BAND", False),
            elastic_band_stiffness=_env_float("PYHOI_ELASTIC_BAND_STIFFNESS", 200.0),
            elastic_band_damping=_env_float("PYHOI_ELASTIC_BAND_DAMPING", 100.0),
            elastic_band_point=_env_vec3("PYHOI_ELASTIC_BAND_POINT", (0.0, 0.0, 3.0)),
            print_scene_information=_env_bool("PYHOI_PRINT_SCENE_INFORMATION", True),
            simulate_dt=float(os.getenv("PYHOI_SIMULATE_DT", "0.005")),
            viewer_dt=float(os.getenv("PYHOI_VIEWER_DT", "0.02")),
            startup_delay_sec=float(os.getenv("PYHOI_STARTUP_DELAY_SEC", "1.0")),
            start_paused=_env_bool("PYHOI_START_PAUSED", True),
            validate_cached_scene=_env_bool("PYHOI_VALIDATE_SCENE", False),
            terrain_collision_mode=terrain_collision_mode,
            terrain_friction=_env_optional_vec("PYHOI_TERRAIN_FRICTION", 3),
            terrain_condim=_env_optional_int("PYHOI_TERRAIN_CONDIM"),
            terrain_solref=_env_optional_vec("PYHOI_TERRAIN_SOLREF", 2),
            terrain_solimp=_env_optional_vec("PYHOI_TERRAIN_SOLIMP", 3),
            terrain_priority=_env_optional_int("PYHOI_TERRAIN_PRIORITY"),
            terrain_box_margin=_env_float("PYHOI_TERRAIN_BOX_MARGIN", 0.0),
            floor_collision=_env_bool("PYHOI_FLOOR_COLLISION", True),
            init_pose_mode=os.getenv("PYHOI_INIT_POSE_MODE", "motion").strip().lower(),
            flat_init_xy_offset=_env_vec2("PYHOI_FLAT_INIT_XY_OFFSET", (5.0, 5.0)),
            vis_contact_points=_env_bool("PYHOI_VIS_CONTACT_POINTS", False),
            vis_contact_forces=_env_bool("PYHOI_VIS_CONTACT_FORCES", False),
            vis_contact_split=_env_bool("PYHOI_VIS_CONTACT_SPLIT", False),
            motion_ref_viz_enabled=_env_bool("PYHOI_MOTION_REF_VIZ", False),
            motion_ref_phase_sync_enable=_env_bool("PYHOI_MOTION_REF_PHASE_SYNC", True),
            motion_ref_align_sim_at_mimic_start=_env_bool("PYHOI_MOTION_REF_ALIGN_SIM_AT_MIMIC", True),
            motion_ref_phase_topic=os.getenv("PYHOI_MOTION_REF_PHASE_TOPIC", "rt/mimic/debug_phase").strip(),
            motion_ref_phase_stale_timeout=_env_float("PYHOI_MOTION_REF_PHASE_STALE_TIMEOUT", 0.5),
            motion_ref_fps=_env_float("PYHOI_MOTION_REF_FPS", 30.0),
            motion_ref_dot_radius=_env_float("PYHOI_MOTION_REF_DOT_RADIUS", 0.02),
            motion_ref_time_offset=_env_float("PYHOI_MOTION_REF_TIME_OFFSET", 0.0),
            motion_ref_loop=_env_bool("PYHOI_MOTION_REF_LOOP", True),
            motion_ref_log_hz=_env_float("PYHOI_MOTION_REF_LOG_HZ", 2.0),
            motion_ref_color_ctrl=_env_vec4("PYHOI_MOTION_REF_COLOR_CTRL", (0.2, 0.9, 0.2, 0.9)),
            motion_ref_color_sim=_env_vec4("PYHOI_MOTION_REF_COLOR_SIM", (0.2, 0.7, 1.0, 0.9)),
        )

    def preflight(self) -> None:
        if self.terrain_collision_mode not in {"mesh", "box"}:
            raise ValueError(
                "PYHOI_TERRAIN_COLLISION_MODE must be 'mesh' or 'box' (use 'primitive' in env as alias for box); "
                f"got {self.terrain_collision_mode!r}"
            )
        if self.init_pose_mode not in {"motion", "flat_offset"}:
            raise ValueError(
                "PYHOI_INIT_POSE_MODE must be 'motion' or 'flat_offset', "
                f"got {self.init_pose_mode!r}"
            )
        required_files = {
            "deploy_yaml": self.deploy_yaml,
            "policy_onnx": self.policy_onnx,
            "motion_file": self.motion_file,
            "terrain_meta_file": self.terrain_meta_file,
            "terrain_urdf": self.terrain_urdf,
            "base_scene": self.base_scene,
            "base_robot_xml": self.base_robot_xml,
            "controller_launcher": self.controller_launcher,
        }
        if self.scene_override is not None:
            required_files["scene_override"] = self.scene_override

        missing = [f"{name}: {path}" for name, path in required_files.items() if not path.is_file()]
        if missing:
            raise FileNotFoundError("Missing required pyHOIsim2sim inputs:\n  " + "\n  ".join(missing))

