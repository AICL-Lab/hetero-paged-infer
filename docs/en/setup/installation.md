# Installation Guide

Complete installation instructions for various environments.

## System Requirements

### Minimum Requirements

| Component | Specification |
|-----------|--------------|
| OS | Linux (Ubuntu 20.04+, CentOS 8+) |
| CPU | x86_64 with AVX2 support |
| RAM | 8 GB |
| GPU | Optional (NVIDIA with CUDA 11.x) |
| Disk | 2 GB free space |

### Recommended for Production

| Component | Specification |
|-----------|--------------|
| OS | Ubuntu 22.04 LTS |
| CPU | 16+ cores, AVX-512 preferred |
| RAM | 32+ GB |
| GPU | NVIDIA A100, H100, or RTX 4090 |
| Disk | NVMe SSD |
| Network | 1 Gbps+ |

## Install Rust

### Using rustup (Recommended)

```bash
# Install rustup
curl --proto '=https' --tlsv1.2 -sSf https://sh.rustup.rs | sh
source $HOME/.cargo/env

# Verify installation
rustc --version
cargo --version
```

### Required Components

```bash
# Add required components for development
rustup component add rustfmt clippy

# Add target if cross-compiling (optional)
rustup target add x86_64-unknown-linux-musl
```

## Install CUDA (Optional)

For GPU acceleration:

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

## Build Hetero-Paged-Infer

### Method 1: From Source (Recommended)

```bash
# Clone repository
git clone https://github.com/LessUp/hetero-paged-infer.git
cd hetero-paged-infer

# Build release version
cargo build --release

# Installation complete
# Binary: ./target/release/hetero-infer
```

### Method 2: Using Make

```bash
# Build with Makefile (if available)
make build

# Install to /usr/local/bin
sudo make install
```

### Method 3: Cargo Install

```bash
# Install directly from crates.io (when published)
cargo install hetero-infer
```

## Docker Installation

### Using Docker

```bash
# Pull image
docker pull ghcr.io/lessup/hetero-paged-infer:latest

# Run
docker run -it --rm \
  --gpus all \
  -v $(pwd)/config.json:/etc/hetero-infer/config.json \
  ghcr.io/lessup/hetero-paged-infer:latest
```

### Build Docker Image

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

## Kubernetes Deployment

### Helm Chart (Planned)

```bash
# Add Helm repo (when available)
helm repo add hetero-infer https://lessup.github.io/hetero-paged-infer
helm repo update

# Install
helm install hetero-infer hetero-infer/hetero-infer \
  --set gpu.enabled=true \
  --set resources.limits.memory=32Gi
```

### Raw Kubernetes Manifest

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

## Verification

### Test Installation

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

### Check CUDA Support (if applicable)

```bash
# Verify CUDA is available
nvidia-smi

# Test GPU executor
cargo test --features cuda --release
```

## Troubleshooting

### Common Issues

#### Build Failures

```
error: could not compile
```
**Solutions:**
```bash
# Update Rust
rustup update

# Clean and rebuild
cargo clean
cargo build --release
```

#### Missing Dependencies

```
error: linker cc not found
```
```bash
# Ubuntu/Debian
sudo apt-get install build-essential

# CentOS/RHEL
sudo dnf install gcc gcc-c++ make
```

#### CUDA Not Found

```
nvcc: command not found
```
```bash
# Add CUDA to PATH
export PATH=/usr/local/cuda/bin:$PATH
export LD_LIBRARY_PATH=/usr/local/cuda/lib64:$LD_LIBRARY_PATH
```

## Uninstallation

```bash
# Remove binary
rm /usr/local/bin/hetero-infer

# Remove config
rm -rf /etc/hetero-infer

# Uninstall from cargo
cargo uninstall hetero-infer
```

---

Next: [Configuration Guide](configuration.md)
