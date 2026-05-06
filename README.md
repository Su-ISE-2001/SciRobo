# SciRobot

SciRobot 是一个基于 **NVIDIA Isaac Sim** 的实验室机器人仿真与数据构建项目，核心目标是进行**异常注入（Anomaly Injection）**：在标准实验任务中系统性注入失败模式，生成可复现的异常轨迹、图像与状态数据，用于异常检测、故障诊断和鲁棒策略评测。

## 主要特性

- 基于 Isaac Sim 的可视化与无头（headless）仿真运行
- 使用 Hydra 配置系统管理多任务实验
- 支持批量任务循环执行（Shell 脚本一键跑多个任务）
- 面向异常注入的任务配置（如 `fail_position`、`fail_angle`、`drop`、`incomplete`、`no_*`）
- 支持截图与视频输出，便于异常回放、定位与复现
- 支持可选关闭数据集写入（`--no-dataset`）

## 异常注入能力（核心）

SciRobot 不是只跑成功轨迹，而是重点构建“可控异常样本”。当前配置体系支持在多任务中注入不同失效类型，例如：

- **位姿偏差类**：`fail_position`、`wrong_angle`、`fail_angle_pick`
- **抓取/交互失败类**：`fail_grasp`、`drop`、`no_pick`、`no_press`
- **过程不完整类**：`incomplete_open`、`incomplete_close`、`incomplete_angle`
- **步骤缺失类**：`no_open`、`no_close`、`no_stir`

这些异常配置可用于：

- 异常检测模型训练（normal vs abnormal）
- 过程级故障定位（在哪一步失败、为何失败）
- 控制器鲁棒性评测（对扰动和执行偏差的容忍能力）

## System Requirements

- NVIDIA GPU with CUDA support (RTX series recommended)
- Linux (Ubuntu recommended)
- conda
- Python 3.10
- Isaac Sim compatible runtime

##  环境配置

### 1. 代码下载

```bash
git clone https://github.com/Su-ISE-2001/SciRobo.git
cd SciRobo
```

如果你是从其他仓库拷贝过来的代码，请先确认当前目录就是项目根目录。

### 2. 安装 Git LFS（强烈建议）

本项目包含大量场景与资源文件，建议启用 Git LFS：

```bash
sudo apt-get update
sudo apt-get install -y git-lfs
git lfs install
git lfs pull
```

### 3. 创建并激活 Conda 环境

```bash
conda create -n isaac_env42 python=3.10 -y
conda activate isaac_env42
```

### 4. 安装项目依赖

```bash
pip install -r requirements.txt
```

### 5.（可选）Isaac Sim 环境初始化

如果你通过 pip 安装 Isaac Sim，可在首次安装后生成 VSCode 配置：

```bash
python -m isaacsim --generate-vscode-settings
```

## 快速开始

### 1) 单任务运行（本地入口）

```bash
python main.py --config-name level1_pick
```

常用参数：

- `--headless`：无界面运行
- `--backend {numpy,gpu}`：仿真后端
- `--no-video`：关闭视频显示/保存
- `--config-name <name>`：选择配置（位于 `config/`）

示例：

```bash
python main.py --config-name level1_pick --headless --backend gpu
```

### 2) 服务端入口（支持关闭数据集写入）

```bash
python main_server.py --config-name level1_pick --headless --no-dataset
```

你可以将 `--config-name` 替换为异常配置名称来直接运行异常注入实验，例如（需对应 `config/` 中存在的配置文件）：

```bash
python main_server.py --config-name level1_pick_fail_position --headless
python main_server.py --config-name level1_pour_incomplete_angle --headless
python main_server.py --config-name level4_DeviceOperation_no_open --headless
```

### 3) 批量任务脚本

项目提供了脚本化批量运行方式：

```bash
bash main.sh
```

`main_server.sh` 支持后台运行参数：

```bash
bash main_server.sh --background
```

后台运行后日志会输出到 `logs/`，并保存 PID 文件，便于停止任务。

## 结果输出

- `logs/`：运行日志
- `outputs/`：任务输出
- 运行目录下通常会包含：
  - `config.yaml`（本次运行配置快照）
  - `screenshots/`（按 episode 保存截图）
  - 视频文件（若未关闭视频保存）

## 项目结构（简要）

```text
LabUtopia/
├── assets/              # 场景与资源文件（含大量 LFS 资源）
├── config/              # Hydra 配置
├── controllers/         # 控制器实现
├── data_collectors/     # 数据采集逻辑
├── factories/           # 工厂方法（robot/task/controller 创建）
├── robots/              # 机器人相关定义
├── tasks/               # 任务定义
├── utils/               # 通用工具
├── main.py              # 主入口
├── main_server.py       # 服务端入口
├── main.sh              # 批量执行脚本（本地）
└── main_server.sh       # 批量执行脚本（服务端/后台）
```

## Git LFS 提示

若推送时报错 `git-lfs not found`，请先安装并初始化：

```bash
sudo apt-get install -y git-lfs
git lfs install
```

## 常见问题

### 1) 无显示器环境运行报错

请添加 `--headless`，或直接使用 `main_server.sh`（脚本会在无 DISPLAY 环境自动追加 headless 参数）。

### 2) 推送 GitHub 失败

- 先确认 Git LFS 已安装
- 使用 PAT（HTTPS）或 SSH key 完成 GitHub 认证
- 再执行 `git push -u <remote> main`

## License

本项目遵循仓库中的许可证文件（如存在 `LICENSE`）。
