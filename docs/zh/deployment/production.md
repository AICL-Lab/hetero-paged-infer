# 生产部署

## 概述

本指南涵盖在生产环境中部署 Hetero-Paged-Infer：以服务模式运行、健康检查、
监控与运维清单。

> **现状说明：** 当前计算核心是 **mock 实现**（输出占位 token，不是自然语言），
> 默认 CPU 后端**不需要 GPU**。下文所有部署方式均针对 mock 引擎；GPU 调优类
> 内容对当前版本不适用。

## 系统要求

| 组件 | 要求 |
|------|------|
| **操作系统** | Linux（推荐 Ubuntu 20.04+） |
| **CPU** | x86_64 |
| **内存** | 8 GB RAM |
| **Rust** | 1.82+（2021 edition） |
| **GPU** | 不需要（mock 引擎；`cuda` feature 为实验性，见[安装指南](../setup/installation.md)） |

## 构建

```bash
git clone https://github.com/AICL-Lab/hetero-paged-infer.git
cd hetero-paged-infer

cargo build --release
cargo test --release

./target/release/hetero-infer --help
```

## 运行服务

生产部署应以**服务模式**运行：加 `--serve` 启动 OpenAI 兼容 HTTP 服务
（默认监听 `127.0.0.1:3000`，模型名 `hetero-infer`）。

> **重要：** 不带 `--serve`（且不带 `--input`）时进程打印提示后**立即退出**，
> 不适合常驻部署。需要对外访问时，在配置文件中把 `serving.host` 改为
> `0.0.0.0`（默认 `127.0.0.1` 只接受本机连接）。

### 配置文件

创建 `/etc/hetero-infer/config.json`（字段说明见[配置页面](../setup/configuration.md)）：

```json
{
  "block_size": 16,
  "max_num_blocks": 2048,
  "max_batch_size": 64,
  "max_num_seqs": 512,
  "max_model_len": 4096,
  "max_total_tokens": 8192,
  "memory_threshold": 0.9,
  "serving": {
    "host": "0.0.0.0",
    "port": 3000,
    "model_name": "hetero-infer"
  }
}
```

### Systemd 服务

创建 `/etc/systemd/system/hetero-infer.service`：

```ini
[Unit]
Description=Hetero-Paged-Infer 推理服务
After=network.target

[Service]
Type=simple
User=hetero
Group=hetero
WorkingDirectory=/opt/hetero-paged-infer
ExecStart=/opt/hetero-paged-infer/target/release/hetero-infer \
  --serve \
  --config /etc/hetero-infer/config.json
Restart=always
RestartSec=5

LimitNOFILE=65536

Environment=RUST_LOG=info
Environment=RUST_BACKTRACE=1

[Install]
WantedBy=multi-user.target
```

启用并启动：

```bash
sudo systemctl daemon-reload
sudo systemctl enable hetero-infer
sudo systemctl start hetero-infer
```

健康检查通过 HTTP 端点（**没有** `--version` flag）：

```bash
curl -fsS http://127.0.0.1:3000/healthz   # 存活探针
curl -fsS http://127.0.0.1:3000/readyz    # 就绪探针
```

### Kubernetes 部署

创建 `deployment.yaml`：

```yaml
apiVersion: apps/v1
kind: Deployment
metadata:
  name: hetero-infer
  labels:
    app: hetero-infer
spec:
  replicas: 1
  selector:
    matchLabels:
      app: hetero-infer
  template:
    metadata:
      labels:
        app: hetero-infer
    spec:
      containers:
      - name: hetero-infer
        image: hetero-infer:latest        # 本地构建镜像，见 docker.md
        args: ["--serve", "--config", "/etc/hetero-infer/config.json"]
        ports:
        - containerPort: 3000
        resources:
          limits:
            memory: "8Gi"
            cpu: "4"
          requests:
            memory: "2Gi"
            cpu: "1"
        livenessProbe:
          httpGet:
            path: /healthz
            port: 3000
        readinessProbe:
          httpGet:
            path: /readyz
            port: 3000
        volumeMounts:
        - name: config
          mountPath: /etc/hetero-infer
      volumes:
      - name: config
        configMap:
          name: hetero-infer-config
---
apiVersion: v1
kind: Service
metadata:
  name: hetero-infer
spec:
  selector:
    app: hetero-infer
  ports:
  - port: 3000
    targetPort: 3000
```

部署（ConfigMap 中的 `serving.host` 必须为 `0.0.0.0`）：

```bash
kubectl create configmap hetero-infer-config --from-file=config.json=/etc/hetero-infer/config.json
kubectl apply -f deployment.yaml
```

> mock 引擎不消费 GPU，因此清单中**没有** `nvidia.com/gpu` 资源申请；
> 仅在部署实验性 `cuda` 构建时才需要添加。

Docker 部署见 [Docker 部署](docker.md)。

## 监控

### 健康检查

| 端点 | 用途 |
|------|------|
| `GET /healthz` | 存活探针（liveness），进程存活即返回 200 |
| `GET /readyz` | 就绪探针（readiness），引擎可接收请求时返回 200 |

### 指标（已实现）

`GET /metrics` 返回 Prometheus 文本格式指标：

| 指标 | 类型 | 含义 |
|------|------|------|
| `hetero_requests_total` | counter | 接收的请求总数 |
| `hetero_errors_total` | counter | 错误响应总数 |
| `hetero_inflight_requests` | gauge | 当前在途请求数 |
| `hetero_streaming_requests_total` | counter | SSE 流式请求总数 |

```bash
curl http://127.0.0.1:3000/metrics
```

### 日志

通过 `RUST_LOG` 环境变量设置级别：

```bash
RUST_LOG=info ./target/release/hetero-infer --serve    # 默认推荐
RUST_LOG=debug ./target/release/hetero-infer --serve   # 排障用
```

## 生产清单

- [ ] 以 `--serve` 启动，进程常驻
- [ ] `serving.host` 按网络拓扑设置（容器内用 `0.0.0.0`，仅本机用默认 `127.0.0.1`）
- [ ] 编排系统接入 `/healthz`（liveness）与 `/readyz`（readiness）探针
- [ ] 抓取 `/metrics` 到 Prometheus 或其他监控系统
- [ ] 客户端处理 **429 + `Retry-After`** 背压信号（过载时服务端会拒绝新请求）
- [ ] 依赖**优雅关闭**：进程收到 SIGTERM / Ctrl+C 后停止接受新连接并排空在途请求，勿用 SIGKILL
- [ ] 设置 `RUST_LOG` 并接入集中日志

## 故障排除

| 问题 | 解决方案 |
|------|----------|
| 进程启动后立即退出 | 确认命令行带 `--serve`（服务模式） |
| 容器外无法访问服务 | 配置文件中 `serving.host` 改为 `0.0.0.0`，端口映射对齐 3000 |
| 请求被拒绝（429） | 引擎过载背压：按 `Retry-After` 退避；或调大 `max_num_seqs` / `max_num_blocks` |
| 请求失败（内存压力） | 减小 `max_model_len` / `max_total_tokens`，或增大 `max_num_blocks` |
| 构建失败 `linker not found` | 安装 build-essential：`sudo apt install build-essential` |

调试：

```bash
RUST_BACKTRACE=1 RUST_LOG=debug ./target/release/hetero-infer --serve
```

## 安全考虑

1. **以非 root 用户运行**
   ```bash
   useradd -r -s /bin/false hetero
   ```
2. **限制文件权限**
   ```bash
   chmod 750 /opt/hetero-paged-infer
   chmod 640 /etc/hetero-infer/config.json
   ```
3. **网络隔离**：默认仅监听 `127.0.0.1`；对外暴露时置于反向代理后并启用 TLS。

---

*API 详情见 [API 参考](../api/core-types)。配置选项见 [配置页面](../setup/configuration.md)。*
