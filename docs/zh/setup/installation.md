# 安装指南

各种环境的完整安装说明。

## 系统要求

### 最低要求

| 组件 | 规格 |
|-----------|--------------|
| 操作系统 | Linux (Ubuntu 20.04+, CentOS 8+) |
| CPU | x86_64，支持 AVX2 |
| 内存 | 8 GB |
| GPU | 可选（NVIDIA，需支持 CUDA 11.x） |
| 磁盘 | 2 GB 可用空间 |

### 生产环境推荐配置

| 组件 | 规格 |
|-----------|--------------|
| 操作系统 | Ubuntu 22.04 LTS |
| CPU | 16+ 核心，推荐支持 AVX-512 |
| 内存 | 32+ GB |
| GPU | NVIDIA A100、H100 或 RTX 4090 |
| 磁盘 | NVMe SSD |
| 网络 | 1 Gbps+ |

## 安装 Rust

### 使用 rustup（推荐）

```bash
# Install rustup
curl --proto '=https' --tlsv1.2 -sSf https://sh.rustup.rs | sh
source $HOME/.cargo/env

# Verify installation
rustc --version
cargo --version
```

### 必需组件

```bash
# Add required components for development
rustup component add rustfmt clippy

# Add target if cross-compiling (optional)
rustup target add x86_64-unknown-linux-musl
```

## 安装 CUDA（可选）

用于 GPU 加速：

### Ubuntu 22.04

```bash
# Install CUDA repository
wget https://developer.download.nvidia.com/compute/cuda/repos/ubuntu2204/x86_64/cuda-keyring_1.0-1_all.deb
sudo dpkg -i cuda-keyring_1.0-1_all.deb
sudo apt-get update

# Install CUDA toolkit
sudo apt-get install cuda-toolkit-12-1

# Add to PATH
echo 'export PATH=/usr/local/cuda/bin:$PATH' >> ~/.bashrc
source ~/.bashrc

# Verify
nvcc --version
nvidia-smi
```

### CentOS/RHEL 8

```bash
# Enable EPEL
sudo dnf install epel-release

# Install CUDA
sudo dnf config-manager --add-repo https://developer.download.nvidia.com/compute/cuda/repos/rhel8/x86_64/cuda-rhel8.repo
sudo dnf install cuda-toolkit-12-1

# Verify
nvcc --version
```

## 构建 Hetero-Paged-Infer

### 方法 1：从源码构建（推荐）

```bash
# Clone repository
git clone https://github.com/LessUp/hetero-paged-infer.git
cd hetero-paged-infer

# Build release version
cargo build --release

# Installation complete
# Binary: ./target/release/hetero-infer
```

### 方法 2：使用 Make

```bash
# Build with Makefile (if available)
make build

# Install to /usr/local/bin
sudo make install
```

### 方法 3：使用 Cargo Install

```bash
# Install directly from crates.io (when published)
cargo install hetero-infer
```

## Docker 安装

### 使用 Docker

```bash
# Pull image
docker pull ghcr.io/lessup/hetero-paged-infer:latest

# Run
docker run -it --rm \
  --gpus all \
  -v $(pwd)/config.json:/etc/hetero-infer/config.json \
  ghcr.io/lessup/hetero-paged-infer:latest
```

### 构建 Docker 镜像

```bash
# Clone repo
git clone https://github.com/LessUp/hetero-paged-infer.git
cd hetero-paged-infer

# Build image
docker build -t hetero-infer:latest .

# Run container
docker run -it --rm \
  --name hetero-infer \
  -p 8080:8080 \
  hetero-infer:latest
```

### docker-compose.yml

```yaml
version: '3.8'

services:
  hetero-infer:
    image: ghcr.io/lessup/hetero-paged-infer:latest
    container_name: hetero-infer
    runtime: nvidia
    environment:
      - NVIDIA_VISIBLE_DEVICES=all
      - RUST_LOG=info
    volumes:
      - ./config.json:/etc/hetero-infer/config.json:ro
    ports:
      - "8080:8080"
    restart: unless-stopped
    deploy:
      resources:
        reservations:
          devices:
            - driver: nvidia
              count: 1
              capabilities: [gpu]
```

## Kubernetes 部署

### Helm Chart（计划中）

```bash
# Add Helm repo (when available)
helm repo add hetero-infer https://lessup.github.io/hetero-paged-infer
helm repo update

# Install
helm install hetero-infer hetero-infer/hetero-infer \
  --set gpu.enabled=true \
  --set resources.limits.memory=32Gi
```

### 原始 Kubernetes 清单

```yaml
apiVersion: apps/v1
kind: Deployment
metadata:
  name: hetero-infer
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
        image: ghcr.io/lessup/hetero-paged-infer:latest
        resources:
          limits:
            nvidia.com/gpu: 1
            memory: "32Gi"
          requests:
            memory: "16Gi"
        volumeMounts:
        - name: config
          mountPath: /etc/hetero-infer
      volumes:
      - name: config
        configMap:
          name: hetero-infer-config
```

## 验证安装

### 测试安装

```bash
# Check version
./target/release/hetero-infer --version

# Run basic test
./target/release/hetero-infer \
  --input "Hello, world!" \
  --max-tokens 10

# Run test suite
cargo test --release
```

### 检查 CUDA 支持（如适用）

```bash
# Verify CUDA is available
nvidia-smi

# Test GPU executor
cargo test --features cuda --release
```

## 故障排除

### 常见问题

#### 构建失败

```
error: could not compile
```
**解决方案：**
```bash
# Update Rust
rustup update

# Clean and rebuild
cargo clean
cargo build --release
```

#### 缺少依赖

```
error: linker cc not found
```
```bash
# Ubuntu/Debian
sudo apt-get install build-essential

# CentOS/RHEL
sudo dnf install gcc gcc-c++ make
```

#### 找不到 CUDA

```
nvcc: command not found
```
```bash
# Add CUDA to PATH
export PATH=/usr/local/cuda/bin:$PATH
export LD_LIBRARY_PATH=/usr/local/cuda/lib64:$LD_LIBRARY_PATH
```

## 卸载

```bash
# Remove binary
rm /usr/local/bin/hetero-infer

# Remove config
rm -rf /etc/hetero-infer

# Uninstall from cargo
cargo uninstall hetero-infer
```

---

下一步：[配置指南](configuration.md)
