#!/bin/bash

# 脚本功能：为 Isaac Sim 正确设置环境并循环执行多个任务配置
# Usage: bash main.sh [其他参数，如 --headless] [--background]
# 注意：在无显示器环境下运行时，会自动添加 --headless 参数
# 使用 --background 参数可以在后台运行，即使终端关闭也能继续执行

# 检查是否需要在后台运行
RUN_IN_BACKGROUND=false
ARGS_WITHOUT_BG=()
for arg in "$@"; do
    if [[ "$arg" == "--background" ]] || [[ "$arg" == "--bg" ]] || [[ "$arg" == "-d" ]]; then
        RUN_IN_BACKGROUND=true
    else
        ARGS_WITHOUT_BG+=("$arg")
    fi
done

# 如果需要后台运行，使用nohup重新启动脚本
if [ "$RUN_IN_BACKGROUND" = true ]; then
    SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
    mkdir -p "$SCRIPT_DIR/logs"
    LOG_FILE="$SCRIPT_DIR/logs/main_$(date +%Y%m%d_%H%M%S).log"
    PID_FILE="$SCRIPT_DIR/logs/main.pid"
    
    echo "--- 启动后台运行模式 ---"
    echo "日志文件: $LOG_FILE"
    echo "PID文件: $PID_FILE"
    
    # 使用nohup在后台运行，并将输出重定向到日志文件
    nohup bash "$SCRIPT_DIR/main_server.sh" "${ARGS_WITHOUT_BG[@]}" > "$LOG_FILE" 2>&1 &
    PID=$!
    
    # 保存PID到文件
    echo $PID > "$PID_FILE"
    
    echo "进程ID: $PID"
    echo "可以使用以下命令查看日志: tail -f $LOG_FILE"
    echo "可以使用以下命令停止进程: kill $PID"
    echo "或者: kill \$(cat $PID_FILE)"
    exit 0
fi

# 第一步：激活正确的 Conda 环境
echo "--- Activating Isaac Sim Environment ---"
source /root/miniconda3/bin/activate isaac_env42

# 第二步：检查是否已指定headless参数
HAS_HEADLESS=false
for arg in "${ARGS_WITHOUT_BG[@]}"; do
    if [[ "$arg" == "--headless" ]]; then
        HAS_HEADLESS=true
        break
    fi
done

# 第三步：如果没有指定headless且无显示器，自动添加
if [ "$HAS_HEADLESS" = false ] && [ -z "$DISPLAY" ]; then
    echo "--- 检测到无显示器环境，自动添加 --headless 参数 ---"
    ARGS_WITHOUT_BG+=("--headless")
fi

# 定义任务配置列表
configs=(
    # "level1_weight_aim"
    # "level1_weight_drop"
    # "level1_weight_tip_over_place"
    # "level1_weight_no_close_pick"
    # "level1_pick_fail_angle_1" 
    # "level1_pick_fail_angle_2"
    # "level1_pick_incomplete_close"
    # "level1_pick_fail_position"
    # "level1_pick_drop"
    # "level1_place_drop"
    # "level1_place_fail_grasp"
    # "level1_place_fail_position"
    # "level1_place_wrong_angle"
    # "level1_place_fail_angle_pick"
    # "level1_place_fail_angle_pick_2"
    # "level1_place_fail_position_pick"
    # "level1_place_incomplete_close_pick"
    # "level1_press_incomplete"
    # "level1_open_door_incomplete"
    # "level1_open_door_fail_grasp"
    # "level1_close_door_incomplete"
    # "level1_close_door_fail_position"
    # "level1_stir_fail_position_pick"
    # "level1_stir_fail_angle_pick_1"
    # "level1_stir_fail_angle_pick_2"
    # "level1_stir_no_pick"
    # "level1_stir_no_stir"
    # "level1_stir_pick_aim"
    # "level1_stir_fail_angle"
    # "level1_pour_fail_position_pick"
    # "level1_pour_fail_angle_pick_1"
    # "level1_pour_fail_angle_pick_2"
    # "level1_pour_incomplete_close_pick"
    # "level1_pour_fail_position"
    # "level1_pour_incomplete_angle"
    # "level1_pour_incomplete_angle_large"
    # "level4_DeviceOperation_no_open"
    # "level4_DeviceOperation_no_pick"
    # "level4_DeviceOperation_no_press"
    # "level4_DeviceOperation_no_close"
    # "level4_DeviceOperation_incomplete_open"
    # "level4_DeviceOperation_fail_position_pick"
    # "level4_DeviceOperation_drop_pick"
    # "level4_DeviceOperation_fail_angle_pick"
    # "level4_DeviceOperation_incomplete_close_pick"
    # "level4_DeviceOperation_fail_position"
    "level1_pick"
    # "level1_stir"
    # "level1_open_drawer"
    # "level1_close_drawer"
    # "level_pipette_offset_fail_pick"
    # "level1_place"
    # "level1_pour"
    # "level1_press"
    # "level1_stir"
    # "level1_open_door"
    # "level1_close_door"
    # "level3_pick"
    # "level3_press"
    # "level3_open"
    # "level3_pour"
    # "level3_stir"
    # "level4_DeviceOperation"
)

# 创建日志目录
mkdir -p logs

# 循环执行所有任务
for config in "${configs[@]}"; do
    echo "开始执行任务: $config"
    echo "================================"
    
    # 执行Python脚本，使用正确的环境变量
    VK_ICD_FILENAMES=/etc/vulkan/icd.d/nvidia_icd.json \
    LD_LIBRARY_PATH=/root/miniconda3/envs/isaac_env42/lib/python3.10/site-packages/omni:$LD_LIBRARY_PATH \
    python main_server.py --config-name "$config" "${ARGS_WITHOUT_BG[@]}" #--no-dataset
    
    # 检查执行结果
    if [ $? -eq 0 ]; then
        echo "✓ 任务 $config 完成"
    else
        echo "✗ 任务 $config 失败"
    fi
    
    echo "================================"
    echo
done

echo "所有任务执行完毕"
echo "--- Isaac Sim Script Execution Finished ---"
