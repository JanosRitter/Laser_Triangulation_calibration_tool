from __future__ import annotations

from src.app.calibration_app import (
    RunOptions,
    run_calibration_app,
)


def main() -> None:
    folder_name = "20260504_082923_robot_calibration"

    options = RunOptions(
        run_trajectory_debug=True,
        run_robot_ray_debug=True,
        run_initial_pose_ray_debug=True,
        run_plane_projection_debug=True,
        run_gt_ray_debug=True,
        run_gt_solver_debug=True,
        run_optimized_pose_ray_debug=True,
        write_debug_log=True,
    )

    run_calibration_app(
        folder_name=folder_name,
        options=options,
    )


if __name__ == "__main__":
    main()