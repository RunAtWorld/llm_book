#!/bin/bash
# 更新版：批量查询昇腾HDK版本（直接输出，适配arm64架构路径）

# 定义IP范围和基础IP
START_IP=2
END_IP=25
BASE_IP="10.246.224"

# 遍历所有节点
for ((ip=$START_IP; ip<=$END_IP; ip++)); do
    node_ip="$BASE_IP.$ip"
    
    printf "\n\033[1;33m===== 节点 %s =====\033[0m\n" "$node_ip"
    
    # 查询HDK信息
    query_cmd='
    function get_value() {
        echo "$1" | grep "$2" | awk -F "|" "{print \$3}" | xargs
    }
    
    # 获取npu-smi信息
    if command -v npu-smi &> /dev/null; then
        npu_info=$(npu-smi info -l 2>/dev/null)
        
        if [ $? -eq 0 ] && [ -n "$npu_info" ]; then
            echo "[硬件信息]"
            get_value "$npu_info" "Chip Version"
            get_value "$npu_info" "Driver Version"
            get_value "$npu_info" "Firmware Version"
        else
            echo "[错误] 未找到NPU设备信息"
        fi
    else
        echo "[错误] npu-smi 命令未找到"
    fi
    
    # 获取CANN版本 - 使用您提供的路径
    cann_path="/usr/local/Ascend/ascend-toolkit/latest/arm64-linux"
    if [ -d "$cann_path" ]; then
        if [ -f "$cann_path/ascend_toolkit_install.info" ]; then
            echo -n "[CANN版本] "
            cat "$cann_path/ascend_toolkit_install.info" 2>/dev/null
        elif [ -f "$cann_path/../ascend_toolkit_install.info" ]; then
            echo -n "[CANN版本] "
            cat "$cann_path/../ascend_toolkit_install.info" 2>/dev/null
        else
            cann_ver=$(ls $cann_path | grep -E "^\d+\.\d+" | sort -V | tail -1)
            if [ -n "$cann_ver" ]; then
                echo "[CANN版本] $cann_ver"
            else
                echo "[警告] 在 $cann_path 下找到安装目录但无法确定版本"
            fi
        fi
    else
        echo "[警告] CANN目录未找到: $cann_path"
    fi
    '
    
    # 执行远程查询
    echo "正在查询..."
    ssh -o ConnectTimeout=5 -o BatchMode=yes "$node_ip" "$query_cmd"
    
    if [ $? -ne 0 ]; then
        echo "[失败] SSH连接或命令执行失败"
    fi
done