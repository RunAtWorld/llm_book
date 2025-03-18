## 安装MindIE
> 参考：[安装MindIE](https://www.hiascend.com/document/detail/zh/mindie/100/envdeployment/instg/mindie_instg_0019.html)


安装过程
```
chmod +x 软件包名.run
source /usr/local/Ascend/ascend-toolkit/set_env.sh
./软件包名.run --check
./软件包名.run --install --quiet
```

配置环境变量

- root用户默认安装路径下配置环境变量：
    ```
    source /usr/local/Ascend/mindie/set_env.sh
    ```
- 非root用户默认安装路径下配置环境变量：
    ```
    source /home/{当前用户名}/Ascend/mindie/set_env.sh
    ```