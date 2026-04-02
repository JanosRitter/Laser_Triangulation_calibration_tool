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