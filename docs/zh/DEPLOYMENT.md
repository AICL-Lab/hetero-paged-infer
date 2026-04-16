# 部署指南

## 概述

本指南涵盖在生产环境中部署 Hetero-Paged-Infer，包括系统要求、构建说明和运维最佳实践。

## 系统要求

### 最低要求

| 组件 | 要求 |
|------|------|
| **操作系统** | Linux（推荐 Ubuntu 20.04+） |
| **CPU** | x86_64 支持 AVX2 |
| **内存** | 8 GB RAM |
| **GPU** | NVIDIA GPU 支持 CUDA 11.x+（可选） |
| **Rust** | 1.70+（2021 edition） |
| **Git** | 2.25+ |

### 生产环境推荐

| 组件 | 推荐配置 |
|------|----------|
| **操作系统** | Ubuntu 22.04 LTS |
| **CPU** | 16+ 核心 |
| **内存** | 32 GB RAM |
| **GPU** | NVIDIA A100 / H100 / RTX 4090 |
| **CUDA** | 12.x |
| **存储** | NVMe SSD 用于模型 |

## 安装

### 1. 安装 Rust

```bash
# 使用 rustup
curl --proto '=https' --tlsv1.2 -sSf https://sh.rustup.rs | sh
source $HOME/.cargo/env

# 验证安装
rustc --version  # 应为 1.70+
```

### 2. 安装 CUDA（可选）

用于 GPU 加速：

```bash
# Ubuntu 22.04 示例
wget https://developer.download.nvidia.com/compute/cuda/repos/ubuntu2204/x86_64/cuda-keyring_1.0-1_all.deb
sudo dpkg -i cuda-keyring_1.0-1_all.deb
sudo apt-get update
sudo apt-get install cuda-toolkit-12-1

# 验证
nvcc --version
nvidia-smi
```

### 3. 克隆与构建

```bash
# 克隆仓库
git clone https://github.com/LessUp/hetero-paged-infer.git
cd hetero-paged-infer

# 构建发布版本
cargo build --release

# 运行测试
cargo test --release

# 二进制文件位置
./target/release/hetero-infer --help
```

## 运行应用

### 基本用法

```bash
# 基本推理
./target/release/hetero-infer \
  --input "你好，世界！" \
  --max-tokens 50

# 自定义参数
./target/release/hetero-infer \
  --input "讲个故事" \
  --max-tokens 200 \
  --temperature 0.8 \
  --top-p 0.95
```

### 使用配置文件

创建 `production.json`：

```json
{
  "block_size": 16,
  "max_num_blocks": 2048,
  "max_batch_size": 64,
  "max_num_seqs": 512,
  "max_model_len": 4096,
  "max_total_tokens": 8192,
  "memory_threshold": 0.9
}
```

使用配置运行：

```bash
./target/release/hetero-infer \
  --config production.json \
  --input "你好，世界！"
```

## 生产部署

### Systemd 服务

创建 `/etc/systemd/system/hetero-infer.service`：

```ini
[Unit]
Description=Hetero-Paged-Infer 推理引擎
After=network.target

[Service]
Type=simple
User=hetero
Group=hetero
WorkingDirectory=/opt/hetero-paged-infer
ExecStart=/opt/hetero-paged-infer/target/release/hetero-infer \
  --config /etc/hetero-infer/config.json
Restart=always
RestartSec=5

# 资源限制
LimitNOFILE=65536

# 环境变量
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
sudo systemctl status hetero-infer
```

### Docker 部署

创建 `Dockerfile`：

```dockerfile
FROM rust:1.75-bullseye as builder

WORKDIR /app
COPY . .
RUN cargo build --release

FROM debian:bullseye-slim

RUN apt-get update && apt-get install -y \
    ca-certificates \
    libssl1.1 \
    && rm -rf /var/lib/apt/lists/*

COPY --from=builder /app/target/release/hetero-infer /usr/local/bin/
COPY --from=builder /app/config.example.json /etc/hetero-infer/config.json

USER nobody
EXPOSE 8080

ENTRYPOINT ["hetero-infer"]
CMD ["--config", "/etc/hetero-infer/config.json"]
```

构建并运行：

```bash
# 构建镜像
docker build -t hetero-infer:latest .

# 运行容器
docker run -d \
  --name hetero-infer \
  --gpus all \
  -v /path/to/config.json:/etc/hetero-infer/config.json \
  -p 8080:8080 \
  hetero-infer:latest
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
        image: hetero-infer:latest
        ports:
        - containerPort: 8080
        resources:
          limits:
            nvidia.com/gpu: 1
            memory: "32Gi"
            cpu: "8"
          requests:
            memory: "16Gi"
            cpu: "4"
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
  - port: 8080
    targetPort: 8080
  type: LoadBalancer
```

部署：

```bash
kubectl apply -f deployment.yaml
```

## 监控

### 日志级别

通过 `RUST_LOG` 环境变量设置：

```bash
# 仅错误
RUST_LOG=error ./hetero-infer

# 警告及以上
RUST_LOG=warn ./hetero-infer

# 信息（默认）
RUST_LOG=info ./hetero-infer

# 调试（详细）
RUST_LOG=debug ./hetero-infer

# 追踪（非常详细）
RUST_LOG=trace ./hetero-infer
```

### 指标（规划中）

计划中的指标端点：

- `GET /metrics` - Prometheus 兼容指标
- 请求率、延迟、批次大小
- 内存利用率、队列深度
- GPU 利用率、温度

### 健康检查

当前通过退出码检查状态：

```bash
# 在 systemd 服务中
ExecStartPre=/path/to/hetero-infer --version
```

## 性能优化

### CPU 优化

1. **CPU 亲和性**
   ```bash
   taskset -c 0-7 ./hetero-infer
   ```

2. **NUMA 感知**
   ```bash
   numactl --cpunodebind=0 --membind=0 ./hetero-infer
   ```

### GPU 优化

1. **持久模式**
   ```bash
   sudo nvidia-smi -pm 1
   ```

2. **GPU 时钟设置**
   ```bash
   sudo nvidia-smi -ac 877,1530
   ```

3. **ECC 内存**
   ```bash
   # 如可接受，禁用 ECC 以获得更好性能
   sudo nvidia-smi -e 0
   ```

### 内存优化

1. **大页内存**
   ```bash
   # 启用透明大页
   echo always > /sys/kernel/mm/transparent_hugepage/enabled
   ```

2. **内存限制**
   ```bash
   # 在 systemd 服务中
   MemoryMax=64G
   MemorySwapMax=0
   ```

## 故障排除

### 构建问题

| 问题 | 解决方案 |
|------|----------|
| `linker not found` | 安装 build-essential：`sudo apt install build-essential` |
| `CUDA not found` | 设置 `CUDA_HOME` 环境变量 |
| `proptest fails` | 使用 `--test-threads=1` 运行 |

### 运行时问题

| 问题 | 解决方案 |
|------|----------|
| OOM 错误 | 减小 `max_num_blocks` 或 `max_batch_size` |
| 推理缓慢 | 增大 `max_batch_size`，启用 CUDA Graphs |
| 请求被拒绝 | 检查 memory_threshold，减少并发请求 |
| GPU 未使用 | 验证 CUDA 安装，检查 `nvidia-smi` |

### 调试模式

```bash
# 启用回溯
RUST_BACKTRACE=1 ./hetero-infer ...

# 完整回溯
RUST_BACKTRACE=full ./hetero-infer ...

# 调试日志
RUST_LOG=debug ./hetero-infer ...
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

3. **网络隔离**
   - 使用防火墙规则
   - 部署在反向代理后
   - 为 API 端点启用 TLS

4. **资源限制**
   - 设置内存限制
   - 配置 CPU 配额
   - 限制 GPU 访问

## 备份与恢复

### 配置备份

```bash
# 备份配置
sudo tar czf hetero-infer-config-$(date +%Y%m%d).tar.gz \
  /etc/hetero-infer/

# 自动化备份（cron）
0 2 * * * tar czf /backup/hetero-infer-config-$(date +\%Y\%m\%d).tar.gz /etc/hetero-infer/
```

### 日志轮转

创建 `/etc/logrotate.d/hetero-infer`：

```
/var/log/hetero-infer/*.log {
    daily
    rotate 14
    compress
    delaycompress
    missingok
    notifempty
    create 0640 hetero hetero
}
```

---

*API 详情见 [API.md](./API.md)。配置选项见 [CONFIGURATION.md](./CONFIGURATION.md)。*
