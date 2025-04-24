#!/bin/bash

# 确保我们在recognition文件夹下运行
cd recognition

# 创建registry文件夹，如果不存在
mkdir -p ../registry

# 遍历recognition文件夹下的所有子文件夹
for dir in */; do
    # 创建对应的registry子文件夹
    mkdir -p "../registry/$dir"
    
    # 获取文件列表
    files=("$dir"*)
    
    # 计算移动文件的数量
    total_files=${#files[@]}
    if (( total_files % 3 == 0 )); then
        move_count=$(( total_files / 3 ))
    else
        move_count=$(( (total_files + 1) / 2 ))
    fi

    # 移动文件
    for ((i=0; i<move_count; i++)); do
        mv "${files[i]}" "../registry/$dir"
    done
done

