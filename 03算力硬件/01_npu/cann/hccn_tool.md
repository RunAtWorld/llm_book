# hccn_tool 工具使用

> 参考： [Atlas 中心训练卡 23.0.RC1 HCCN Tool 接口参考 01](https://support.huawei.com/enterprise/zh/doc/EDOC1100303317/eb173a4e)

## 命令样例

> 设置 -s，获取 -g

### 配置参数

配置RoCE网卡IP地址和子网掩码

```
hccn_tool [-i %d] -ip -s[address %s] [netmask %s]
```

参数说明

| 参数    | 说明                         |
| ------- | ---------------------------- |
| -i      | 指定设备ID。取值范围：0~15。 |
| -ip     | 指定IP属性。                 |
| -s      | 设置属性。                   |
| address | IP地址。                     |
| netmask | 子网掩码。                   |

返回值：

- 0：成功
- 非0：失败

约束说明：该命令仅支持在物理机root用户下运行。

使用样例：
```
hccn_tool -i 0 -ip -s address 192.168.2.10 netmask 255.255.255.0
```

**注意事项**
1. 针对AI Server上每个Device侧OS管理4块昇腾AI处理器，需要为每个OS上的4块网卡配置不同的IP。
2. 首次配置IP时会出现15秒后link状态变为down，然后恢复up状态的情况。
3. 192.168.1.X、192.168.2.192、192.168.2.196、192.168.3.193、192.168.3.197、192.168.4.194、192.168.4.198、192.168.5.195和192.168.5.199用于板内网络通信使用，不支持配置。

### 获取参数

获取RoCE网卡默认网关

```
hccn_tool [-i %d] -gateway -g
```

参数说明

| 参数     | 说明                         |
| -------- | ---------------------------- |
| -i       | 指定设备ID。取值范围：0~15。 |
| -gateway | 指定网关属性。               |
| -g       | 获取属性。                   |

使用样例

```
hccn_tool -i 0 -gateway -g
```



## 常见命令解析

获取网络健康状态（本端与所配置的检测IP之间的连通状态）



## 常用脚本

```
for i in {0..15};do echo "==> $i"; hccn_tool -i $i -net_health -g;done
for i in {0..15};do echo "==> $i"; hccn_tool -i $i -link -g;done
for i in {0..15};do echo "==> $i"; hccn_tool -i $i -ip -g;done
for i in {0..15};do echo "==> $i"; hccn_tool -i $i -gateway -g;done
for i in {0..15};do echo "==> $i"; hccn_tool -i $i -netdetect -g;done
```

