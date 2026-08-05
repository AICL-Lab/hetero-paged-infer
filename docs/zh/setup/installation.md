# 安装指南

各种环境的完整安装说明。

## 系统要求

| 组件 | 规格 |
|-----------|--------------|
| 操作系统 | Linux (Ubuntu 20.04+, CentOS 8+) |
| CPU | x86_64 |
| 内存 | 8 GB |
| 磁盘 | 2 GB 可用空间 |
| Rust | 1.82+（2021 edition） |
| GPU | 不需要（当前为 mock 计算核心；实验性 `cuda` feature 见下文） |

## 安装 Rust

### 使用 rustup（推荐）

```bash
# Install rustup
curl --proto '=https' --tlsv1.2 -sSf https://sh.rustup.rs | sh
source $HOME/.cargo/env

# Verify installation (需要 1.82+)
rustc --version
cargo --version
```

### 开发组件（可选）

```bash
rustup component add rustfmt clippy

# 交叉编译目标（可选）
rustup target add x86_64-unknown-linux-musl
```

## 安装 CUDA（可选）

仅用于实验性的 `cuda` feature；默认构建与 mock 引擎**不需要** CUDA。

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

### 从源码构建（推荐）

```bash
# Clone repository
git clone https://github.com/AICL-Lab/hetero-paged-infer.git
cd hetero-paged-infer

# Build release version
cargo build --release

# 二进制文件: ./target/release/hetero-infer
```

> 尚未发布到 crates.io，因此暂不支持 `cargo install`。

## Docker 安装

> 本项目**没有发布镜像**，需从源码本地构建。详见 [Docker 部署](../deployment/docker.md)。

```bash
git clone https://github.com/AICL-Lab/hetero-paged-infer.git
cd hetero-paged-infer

# Build image
docker build -t hetero-infer:latest .

# Run container（服务模式，端口 3000）
docker run -it --rm \
  --name hetero-infer \
  -p 3000:3000 \
  hetero-infer:latest
```

## Kubernetes 部署

没有 Helm chart；使用原始 Kubernetes 清单（镜像需先本地构建并推送到你的
registry，`serving.host` 需设为 `0.0.0.0`）：

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
        image: hetero-infer:latest
        args: ["--serve", "--config", "/etc/hetero-infer/config.json"]
        ports:
        - containerPort: 3000
        resources:
          limits:
            memory: "8Gi"
          requests:
            memory: "2Gi"
        volumeMounts:
        - name: config
          mountPath: /etc/hetero-infer
      volumes:
      - name: config
        configMap:
          name: hetero-infer-config
```

> mock 引擎不消费 GPU，清单中不含 `nvidia.com/gpu`；仅实验性 `cuda` 构建需要。
> 完整清单（探针、Service）见[生产部署](../deployment/production.md)。

## 验证安装

```bash
# 查看帮助（注意：没有 --version flag）
./target/release/hetero-infer --help

# 运行一次推理（输出为占位 token，见快速入门）
./target/release/hetero-infer \
  --input "Hello, world!" \
  --max-tokens 10

# 运行测试套件
cargo test --release
```

### 检查 CUDA 支持（如适用）

```bash
# 验证 nvcc 可用
nvcc --version

# 明确使用系统工具链，而不是 conda 包装的编译器
CC=/usr/bin/gcc-12 \
CXX=/usr/bin/g++-12 \
CUDAHOSTCXX=/usr/bin/g++-12 \
cargo test --all-features --release
```

如果环境里有 `nvcc`，这里验证的是 **最小真实 CUDA kernel 路径**；如果没有 `nvcc`，同一套 feature 会自动回退到 host 兼容后端，以便 CI 继续覆盖 CUDA 相关的 Rust 集成面。这并不代表生产级 CUDA 注意力 kernel 已经实现。

## 故障排除

### 常见问题

#### 构建失败

```
error: could not compile
```
**解决方案：**
```bash
# Update Rust (需要 1.82+)
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

如果你只是需要让 Rust 侧的 CUDA feature 通过编译和测试，现在构建会自动回退，不再强制要求 `nvcc`。

## 卸载

```bash
# Remove built binary
rm ./target/release/hetero-infer

# Remove config (if installed system-wide)
rm -rf /etc/hetero-infer
```

---

下一步：[配置指南](configuration)
