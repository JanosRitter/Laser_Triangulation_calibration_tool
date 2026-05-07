# src/calibration/trajectory_debug.py

from __future__ import annotations

from src.debug.trajectory_debug import (
    build_ground_truth_transforms_from_frame_table,
    rotation_angle_deg_from_matrix,
    compare_transform_lists,
    summarize_transform_comparison,
    can_run_trajectory_debug,
    run_optional_trajectory_debug,
    print_trajectory_debug_report,
)