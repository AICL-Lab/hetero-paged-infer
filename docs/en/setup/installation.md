# Installation Guide

Complete installation instructions for various environments.

## System Requirements

| Component | Specification |
|-----------|--------------|
| OS | Linux (Ubuntu 20.04+, CentOS 8+) |
| CPU | x86_64 |
| RAM | 8 GB |
| Disk | 2 GB free space |
| Rust | 1.82+ (2021 edition) |
| GPU | Not required (the compute core is currently a mock; the experimental `cuda` feature is covered below) |

## Install Rust

### Using rustup (Recommended)

```bash
# Install rustup
curl --proto '=https' --tlsv1.2 -sSf https://sh.rustup.rs | sh
source $HOME/.cargo/env

# Verify installation (1.82+ required)
rustc --version
cargo --version
```

### Development Components (Optional)

```bash
rustup component add rustfmt clippy

# Cross-compilation target (optional)
rustup target add x86_64-unknown-linux-musl
```

## Install CUDA (Optional)

Only needed for the experimental `cuda` feature; the default build and the mock
engine do **not** require CUDA.

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

### From Source (Recommended)

```bash
# Clone repository
git clone https://github.com/AICL-Lab/hetero-paged-infer.git
cd hetero-paged-infer

# Build release version
cargo build --release

# Binary: ./target/release/hetero-infer
```

> Not published on crates.io yet, so `cargo install` is not available.

## Docker Installation

> This project has **no published images**; build locally from source.
> See [Docker Deployment](../deployment/docker.md) for details.

```bash
git clone https://github.com/AICL-Lab/hetero-paged-infer.git
cd hetero-paged-infer

# Build image
docker build -t hetero-infer:latest .

# Run container (server mode, port 3000)
docker run -it --rm \
  --name hetero-infer \
  -p 3000:3000 \
  hetero-infer:latest
```

## Kubernetes Deployment

There is no Helm chart; use a raw Kubernetes manifest (build the image locally
first and push it to your registry; `serving.host` must be set to `0.0.0.0`):

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

> The mock engine does not consume GPUs, so the manifest contains no
> `nvidia.com/gpu`; only the experimental `cuda` build needs one.
> For the full manifest (probes, Service), see [Production Deployment](../deployment/production.md).

## Verification

```bash
# Show help (note: there is no --version flag)
./target/release/hetero-infer --help

# Run one inference (output is placeholder tokens, see Quick Start)
./target/release/hetero-infer \
  --input "Hello, world!" \
  --max-tokens 10

# Run the test suite
cargo test --release
```

### Check CUDA Support (if applicable)

```bash
# Verify nvcc is available
nvcc --version

# Use the system toolchain instead of a conda compiler wrapper
CC=/usr/bin/gcc-12 \
CXX=/usr/bin/g++-12 \
CUDAHOSTCXX=/usr/bin/g++-12 \
cargo test --all-features --release
```

With `nvcc` available, this validates a minimal real CUDA kernel path. Without `nvcc`, the same feature set falls back to a host-compatible backend so CI can still compile and test the CUDA-facing Rust integration. It does **not** yet imply that production CUDA attention kernels are implemented.

## Troubleshooting

### Common Issues

#### Build Failures

```
error: could not compile
```
**Solutions:**
```bash
# Update Rust (1.82+ required)
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

If you only need the Rust-side CUDA feature surface to compile and test, the build now falls back automatically and does not require `nvcc`.

## Uninstallation

```bash
# Remove built binary
rm ./target/release/hetero-infer

# Remove config (if installed system-wide)
rm -rf /etc/hetero-infer
```

---

Next: [Configuration Guide](configuration)
