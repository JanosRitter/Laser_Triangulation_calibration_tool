# Laser Triangulation Calibration Tool

This project implements a simulation-based calibration pipeline for a laser triangulation system using two 6-axis robotic arms.

## 🚀 Overview

The goal is to estimate the relative pose between:
- a **camera system**
- a **laser projection system**

based on:
- known robot motion (trajectory)
- observed laser points in camera images

---

## 📦 Project Structure
src/
calibration/
fitting/
geometry/
io/
evaluation/
visualization/

---

## 🔬 Core Idea

We combine:

- Camera rays (from pixel observations)
- Laser rays (from robot trajectory)

The calibration problem becomes:

> Find the transformation such that both ray bundles intersect consistently in 3D space.

---

## 🧪 Current Status

✅ Simulation pipeline implemented  
✅ Trajectory reconstruction verified (exact match with GT)  
✅ Camera ray model validated  
✅ Debug tools for geometry consistency  
🚧 Calibration solver (least squares) in progress  

---

## 📊 Data

The tool operates on simulation datasets containing:

- `metadata.json`
- `frame_table.csv`
- cropped laser observations (`.npy`)

---

## ⚙️ Requirements

- Python 3.10+
- numpy
- scipy

Install dependencies:

```bash
pip install -r requirements.txt
```

## Running evaluations

`main.py` can start three evaluation modes. Set `MODE` at the top of the file
and configure the matching block below it:

- `single_run`: evaluates one complete calibration run.
- `multi_run`: evaluates several complete runs and compares their optimized
  camera poses statistically.
- `intra_run`: repeatedly draws random subsets from one large run and compares
  the resulting calibrations statistically.

All modes use the same core calibration pipeline. The difference is only which
observations are passed into that pipeline and how the results are aggregated.

Start the selected mode with:

```bash
python main.py
```

## Multi-run evaluation

`multi_run` evaluates several calibration runs sequentially and compares the
optimized camera poses in the robot frame. Configure the run folders in
`MULTI_RUN_FOLDERS`.

The combined evaluation is written to a new subfolder below
`data/multirun_evaluation/`. If the run folder names start with timestamps like
`YYYYMMDD_HHMMSS`, the folder name is derived from the covered time range, for
example `multirun_20260630_10_3748_5625`. If the naming convention does not
match, a timestamped fallback folder name is used. Existing evaluation folders
are not overwritten; repeated evaluations get a numeric suffix.

- `multirun_results.csv`: one row per successful run
- `multirun_summary.json`: aggregate position, orientation and fit statistics
- `camera_position_components.png`: absolute x/y/z camera positions
- `camera_position_deviations.png`: x/y/z deviations from the mean
- `camera_position_3d.png`: spatial distribution of camera centers
- `camera_orientation_deviations.png`: rotation-vector and angular deviations
- `fit_quality_by_run.png`: ray-pair RMSE for each run

Rotations are averaged on SO(3), not by directly averaging Euler angles.

## Intra-run evaluation

`run_intrarun_evaluation` repeatedly draws random subsets from one large
calibration run and evaluates every subset with the normal calibration
pipeline. Configure `INTRA_RUN_FOLDER`, `OBSERVATIONS_PER_SUBRUN`,
`NUM_SUBRUNS`, and `RANDOM_SEED` in `main.py`.

Results are stored below the source run in:

```text
statistical_evaluation/evaluation_YYYYMMDD_HHMMSS/
```

The directory contains one folder per sub-run, aggregate pose and fit plots,
sampling coverage plots, `subrun_results.csv`, the complete observation
assignment in `subrun_observation_selections.csv`, and all configuration and
statistics in `intrarun_summary.json`.

The intra-run pipeline also creates a standardized 6D parameter-coupling
analysis from `subrun_results.csv`: Pearson/Spearman heatmaps, a pair plot,
translation-rotation scatter plots with confidence bands and covariance
ellipses, PCA plots, parallel coordinates, optional interactive Plotly 3D
views, numerical CSV/JSON exports, and an automatic Markdown interpretation.
Existing evaluations can be analyzed without repeating any calibration fit:

```python
from src.evaluation.intrarun_evaluation import (
    regenerate_intrarun_parameter_correlation_analysis,
)

regenerate_intrarun_parameter_correlation_analysis("path/to/evaluation_dir")
```
