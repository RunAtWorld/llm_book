find . -type f -exec sha256sum {} \; > sha256sums.txt
-exec sha256sum {} \; 这是 find 命令的执行动作部分，{} 代表当前找到的文件，\; 是exec动作的结束标志。
sort -k2 sha256sums.txt #按第2列排序
