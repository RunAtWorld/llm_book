# ---- 阶段1: 安装依赖 ----
FROM python:3.12-slim AS deps

WORKDIR /docs
COPY requirements.txt .
RUN pip install --user --no-cache-dir -r requirements.txt

# ---- 阶段2: 最小运行镜像 ----
FROM python:3.12-slim

WORKDIR /docs

# 从 deps 阶段复制已安装的依赖（仅 site-packages 和 bin）
COPY --from=deps /root/.local /root/.local
ENV PATH=/root/.local/bin:$PATH

# 复制文档内容（运行时通过 volume 挂载覆盖）
COPY . .

EXPOSE 8000

CMD ["mkdocs", "serve", "-a", "0.0.0.0:8000"]
