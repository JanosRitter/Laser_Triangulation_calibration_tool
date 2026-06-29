from __future__ import annotations

from src.app.calibration_app import (
    RunOptions,
    run_calibration_app,
)


def main() -> None:
    folder_name = "20260520_095934_robot_calibration"

    options = RunOptions(
        run_trajectory_debug=True,
        run_robot_ray_debug=True,
        run_initial_ray_pair_debug=True,
        run_camera_pose_optimization= True,
        run_optimized_ray_pair_debug= True,
    )

    run_calibration_app(
        folder_name=folder_name,
        options=options,
    )


if __name__ == "__main__":
    main()