# SciRobot

SciRobot is a simulation and data-generation project built on **NVIDIA Isaac Sim**, with a primary focus on **Anomaly Injection** for scientific embodied agents.  
Instead of collecting only successful trajectories, SciRobot systematically injects failure modes into laboratory tasks and produces reproducible abnormal trajectories, images, and state logs for anomaly detection, fault diagnosis, and robustness evaluation.

## Key Features

- Isaac Sim-based simulation with both GUI and headless execution
- Hydra-based configuration management for multi-task experiments
- Batch execution via shell scripts for large-scale runs
- **Anomaly-oriented task configs** (e.g., `fail_position`, `fail_angle`, `drop`, `incomplete`, `no_*`)
- Screenshot/video logging for replay, debugging, and analysis
- Optional dataset disabling via `--no-dataset`

## Anomaly Injection (Core Capability)

SciRobot is designed to generate controllable abnormal samples, not just successful demos.  
Current config patterns support multiple failure categories:

- **Pose/geometry deviations**: `fail_position`, `wrong_angle`, `fail_angle_pick`
- **Manipulation/interactions failures**: `fail_grasp`, `drop`, `no_pick`, `no_press`
- **Incomplete execution**: `incomplete_open`, `incomplete_close`, `incomplete_angle`
- **Missing procedural steps**: `no_open`, `no_close`, `no_stir`

These abnormalities can be used for:

- anomaly detection training (normal vs abnormal),
- process-level fault localization,
- policy/controller robustness benchmarking.

## System Requirements

- NVIDIA GPU with CUDA support (RTX series recommended)
- Linux (Ubuntu recommended)
- conda
- Python 3.10
- Isaac Sim-compatible runtime

##  Environment Setup

### 1. Clone the Code

```bash
git clone https://github.com/Su-ISE-2001/SciRobo.git
cd SciRobo
```

If you copied the code from another repository, make sure you are in the project root before running commands.

### 2. Install Git LFS (Strongly Recommended)

This project contains large scene/resource files managed by LFS rules:

```bash
sudo apt-get update
sudo apt-get install -y git-lfs
git lfs install
git lfs pull
```

### 3. Create and Activate a Conda Environment

```bash
conda create -n isaac_env42 python=3.10 -y
conda activate isaac_env42
```

### 4. Install Project Dependencies

```bash
pip install -r requirements.txt
```

### 5. (Optional) Initialize Isaac Sim VSCode Settings

```bash
python -m isaacsim --generate-vscode-settings
```

## Quick Start

### 1) Run a Single Task (Main Entry)

```bash
python main.py --config-name level1_pick
```

Common arguments:

- `--headless`: run without GUI
- `--backend {numpy,gpu}`: simulation backend
- `--no-video`: disable video display/saving
- `--config-name <name>`: choose task config from `config/`

Example:

```bash
python main.py --config-name level1_pick --headless --backend gpu
```

### 2) Server Entry (Supports `--no-dataset`)

```bash
python main_server.py --config-name level1_pick --headless --no-dataset
```

Run anomaly injection experiments by switching to anomaly configs (if present in `config/`):

```bash
python main_server.py --config-name level1_pick_fail_position --headless
python main_server.py --config-name level1_pour_incomplete_angle --headless
python main_server.py --config-name level4_DeviceOperation_no_open --headless
```

### 3) Batch Execution Scripts

```bash
bash main.sh
```

Run in background with:

```bash
bash main_server.sh --background
```

Logs are written into `logs/`, and PID is recorded for easy process management.

## Outputs

- `logs/`: runtime logs
- `outputs/`: run outputs
- Typical run artifacts:
  - `config.yaml` (run snapshot),
  - `screenshots/` (episode-level screenshots),
  - video files (if video saving is enabled)

## Project Structure (Brief)

```text
LabUtopia/
├── assets/              # Scene/resource files (LFS-heavy)
├── config/              # Hydra configs
├── controllers/         # Controllers
├── data_collectors/     # Data collection logic
├── factories/           # Factory methods
├── robots/              # Robot definitions
├── tasks/               # Task definitions
├── utils/               # Utilities
├── main.py              # Main entry
├── main_server.py       # Server entry
├── main.sh              # Local batch script
└── main_server.sh       # Server/background batch script
```

## Git LFS Note

If push fails with `git-lfs not found`, run:

```bash
sudo apt-get install -y git-lfs
git lfs install
```

## FAQ

### 1) Error in headless/server environments

Use `--headless`, or directly run `main_server.sh` (it auto-adds headless mode when no DISPLAY is detected).

### 2) GitHub push/authentication failure

- Confirm Git LFS is installed
- Authenticate with PAT (HTTPS) or SSH key
- Push again with `git push -u <remote> main`

## License

This project follows the license file in the repository (`LICENSE`, if present).
