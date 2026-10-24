# -*- coding: utf-8 -*-
"""校验仓库内所有 mermaid 代码围栏是否配平、有无嵌套错误。"""
import os, glob

root = r"E:/src/my/llm_book"
open_c = 0
close_c = 0
problems = []
mermaid_docs = 0

for f in glob.glob(os.path.join(root, "**", "*.md"), recursive=True):
    if ".workbuddy" in f or "\\site\\" in f or "/site/" in f:
        continue
    with open(f, encoding="utf-8") as fh:
        lines = fh.readlines()
    in_mermaid = False
    has_mermaid = False
    for ln in lines:
        s = ln.strip()
        if s == "```mermaid":
            open_c += 1
            has_mermaid = True
            if in_mermaid:
                problems.append(("nested-open", f))
            in_mermaid = True
        elif s == "```":
            if in_mermaid:
                close_c += 1
                in_mermaid = False
    if has_mermaid:
        mermaid_docs += 1
    if in_mermaid:  # 文件结束仍未闭合
        problems.append(("unclosed", f))

print("含 mermaid 的文档数:", mermaid_docs)
print("mermaid 开围栏:", open_c)
print("mermaid 闭围栏:", close_c)
print("问题:", problems if problems else "无 ✅")
