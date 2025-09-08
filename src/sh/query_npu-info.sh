#!/bin/bash
# 极简版: 批量查询昇腾节点npu-smi版本信息

# 定义IP范围
START_IP=2
END_IP=25
BASE_IP="10.246.224"

# 遍历所有节点
for ((ip=$START_IP; ip<=$END_IP; ip++)); do
    node_ip="$BASE_IP.$ip"
    
    printf "\n\033[1;33m===== 节点 %s =====\033[0m\n" "$node_ip"
    
    # 执行远程命令
    echo "正在查询..."
    result=$(ssh -o ConnectTimeout=5 -o BatchMode=yes "$node_ip" "npu-smi info 2>/dev/null")
    
    # 结果处理
    if [ $? -eq 0 ] && [ -n "$result" ]; then
        echo "npu-smi版本: $result"
    else
        echo "[错误] 查询失败"
    fi
done