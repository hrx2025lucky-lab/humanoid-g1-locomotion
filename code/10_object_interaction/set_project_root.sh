#!/usr/bin/env bash
# Source this file from the share package root (or any cwd):
#   source scripts/set_project_root.sh
#
# Sets PROJECT_ROOT and the HOI_MIMIC_* env vars used by tracking_env_cfg.py
# so training/play work when the zip is extracted outside /home/sustech/unitree_project.

export PROJECT_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"

export HOI_ROOT="${HOI_ROOT:-${PROJECT_ROOT}/datasets}"
export HOI_MIMIC_DATA_DIR="${HOI_ROOT}/hoi_mimic_data"

export HOI_MIMIC_TERRAIN_MOTION_FILE="${HOI_MIMIC_TERRAIN_MOTION_FILE:-${HOI_MIMIC_DATA_DIR}/climb_15_z_scale_1.0_mimic.npz}"
export HOI_MIMIC_TERRAIN_META_FILE="${HOI_MIMIC_TERRAIN_META_FILE:-${HOI_MIMIC_DATA_DIR}/climb_15_z_scale_1.0_mimic.terrain.json}"

# export HOI_MIMIC_TERRAIN_URDF="${HOI_MIMIC_TERRAIN_URDF:-${HOI_ROOT}/models/terrain/climb_15/multi_boxes_z_scale_1.0.urdf}"       # bind task default URDF
export HOI_MIMIC_TERRAIN_URDF="${HOI_MIMIC_TERRAIN_URDF:-${HOI_ROOT}/models/terrain/climb_15/multi_boxes_z_scale_1.0_isaac_world.urdf}" # perceptive task default URDF

echo "[set_project_root] PROJECT_ROOT=${PROJECT_ROOT}"
echo "[set_project_root] HOI_ROOT=${HOI_ROOT}"
echo "[set_project_root] HOI_MIMIC_DATA_DIR=${HOI_MIMIC_DATA_DIR}"
echo "[set_project_root] HOI_MIMIC_TERRAIN_MOTION_FILE=${HOI_MIMIC_TERRAIN_MOTION_FILE}"
echo "[set_project_root] HOI_MIMIC_TERRAIN_META_FILE=${HOI_MIMIC_TERRAIN_META_FILE}"
echo "[set_project_root] HOI_MIMIC_TERRAIN_URDF=${HOI_MIMIC_TERRAIN_URDF}"
