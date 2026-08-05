# Docker 部署

> **尚无发布镜像：** 本项目目前没有发布任何 Docker 镜像（CI 不包含镜像发布流程），
> 请从源码**本地构建**。mock 引擎不需要 GPU，以下示例均不带 `--gpus`。

## Dockerfile

在仓库根目录创建 `Dockerfile`：

```dockerfile
FROM rust:1.82-bookworm AS builder

WORKDIR /app
COPY . .
RUN cargo build --release

FROM debian:bookworm-slim

RUN apt-get update && apt-get install -y \
    ca-certificates \
    curl \
    && rm -rf /var/lib/apt/lists/*

COPY --from=builder /app/target/release/hetero-infer /usr/local/bin/
COPY config.example.json /etc/hetero-infer/config.json

USER nobody
EXPOSE 3000

# 必须带 --serve，否则进程打印提示后立即退出
ENTRYPOINT ["hetero-infer"]
CMD ["--serve", "--config", "/etc/hetero-infer/config.json"]
```

## 构建与运行

```bash
git clone https://github.com/AICL-Lab/hetero-paged-infer.git
cd hetero-paged-infer

# 构建镜像
docker build -t hetero-infer:latest .

# 运行（服务模式，映射默认端口 3000）
docker run -d \
  --name hetero-infer \
  -p 3000:3000 \
  hetero-infer:latest

# 验证
curl http://127.0.0.1:3000/healthz
```

> **容器网络注意：** 默认配置监听 `127.0.0.1`，容器外**无法访问**。
> 对外暴露时挂载自定义配置，把 `serving.host` 设为 `0.0.0.0`：
>
> ```bash
> docker run -d \
>   --name hetero-infer \
>   -p 3000:3000 \
>   -v $(pwd)/config.json:/etc/hetero-infer/config.json:ro \
>   hetero-infer:latest
> ```
>
> 其中 `config.json` 至少包含：
>
> ```json
> {
>   "serving": { "host": "0.0.0.0", "port": 3000, "model_name": "hetero-infer" }
> }
> ```
>
> （其余字段省略时取默认值。）

## docker-compose.yml

```yaml
services:
  hetero-infer:
    build: .
    image: hetero-infer:latest
    container_name: hetero-infer
    command: ["--serve", "--config", "/etc/hetero-infer/config.json"]
    environment:
      - RUST_LOG=info
    volumes:
      - ./config.json:/etc/hetero-infer/config.json:ro
    ports:
      - "3000:3000"
    restart: unless-stopped
    healthcheck:
      test: ["CMD", "curl", "-fsS", "http://127.0.0.1:3000/healthz"]
      interval: 10s
      timeout: 3s
      retries: 3
```

`config.json` 需把 `serving.host` 设为 `0.0.0.0`（见上文说明）。

---

服务端口、探针与指标端点见[生产部署](production.md)。
