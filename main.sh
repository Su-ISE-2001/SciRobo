#!/bin/bash

# 定义任务配置列表
configs=(
    #"level1_pick"
    #"level1_place" 
    #"level1_pour"
    # "level1_press"
    # "level1_stir"
    # "level1_open_door"
    # "level1_open_drawer"
    # "level8_close_door"
    # "level9_close_drawer"
#     "level2_ShakeBeaker"
#     "level2_StirGlassrod"
#     "level2_PourLiquid"
#     "level2_TransportBeaker"
#     "level2_Heat_Liquid"
#     "level2_openclose" 
     
     "level4_DeviceOperation"
# ##Level 3 Generalization Tasks:**
#     "level3_PourLiquid" 
#     "level3_Heat_Liquid" 
#     "level3_TrabsportBeaker"
#     "level3_open"
#     "level3_pick" 
#     "level3_press"
)

# 创建日志目录
mkdir -p logs

# 循环执行所有任务
for config in "${configs[@]}"; do
    echo "开始执行任务: $config"
    echo "================================"
    
    # 执行Python脚本
    python main.py --config-name "$config"
    
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
