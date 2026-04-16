# Deployment Guide

## Overview

This guide covers deploying Hetero-Paged-Infer in production environments, including system requirements, build instructions, and operational best practices.

## System Requirements

### Minimum Requirements

| Component | Requirement |
|-----------|-------------|
| **OS** | Linux (Ubuntu 20.04+ recommended) |
| **CPU** | x86_64 with AVX2 support |
| **Memory** | 8 GB RAM |
| **GPU** | NVIDIA GPU with CUDA 11.x+ (optional) |
| **Rust** | 1.70+ (2021 edition) |
| **Git** | 2.25+ |

### Recommended for Production

| Component | Recommendation |
|-----------|---------------|
| **OS** | Ubuntu 22.04 LTS |
| **CPU** | 16+ cores |
| **Memory** | 32 GB RAM |
| **GPU** | NVIDIA A100 / H100 / RTX 4090 |
| **CUDA** | 12.x |
| **Storage** | NVMe SSD for models |

## Installation

### 1. Install Rust

```bash
# Using rustup
curl --proto '=https' --tlsv1.2 -sSf https://sh.rustup.rs | sh
source $HOME/.cargo/env

# Verify installation
rustc --version  # Should be 1.70+
```

### 2. Install CUDA (Optional)

For GPU acceleration:

```bash
# Ubuntu 22.04 example
wget https://developer.download.nvidia.com/compute/cuda/repos/ubuntu2204/x86_64/cuda-keyring_1.0-1_all.deb
sudo dpkg -i cuda-keyring_1.0-1_all.deb
sudo apt-get update
sudo apt-get install cuda-toolkit-12-1

# Verify
nvcc --version
nvidia-smi
```

### 3. Clone and Build

```bash
# Clone repository
git clone https://github.com/LessUp/hetero-paged-infer.git
cd hetero-paged-infer

# Build release version
cargo build --release

# Run tests
cargo test --release

# Binary location
./target/release/hetero-infer --help
```

## Running the Application

### Basic Usage

```bash
# Basic inference
./target/release/hetero-infer \
  --input "Hello, world!" \
  --max-tokens 50

# With custom parameters
./target/release/hetero-infer \
  --input "Tell me a story" \
  --max-tokens 200 \
  --temperature 0.8 \
  --top-p 0.95
```

### Using Configuration File

Create `production.json`:

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

Run with configuration:

```bash
./target/release/hetero-infer \
  --config production.json \
  --input "Hello, world!"
```

## Production Deployment

### Systemd Service

Create `/etc/systemd/system/hetero-infer.service`:

```ini
[Unit]
Description=Hetero-Paged-Infer Inference Engine
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

# Resource limits
LimitNOFILE=65536

# Environment
Environment=RUST_LOG=info
Environment=RUST_BACKTRACE=1

[Install]
WantedBy=multi-user.target
```

Enable and start:

```bash
sudo systemctl daemon-reload
sudo systemctl enable hetero-infer
sudo systemctl start hetero-infer
sudo systemctl status hetero-infer
```

### Docker Deployment

Create `Dockerfile`:

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

Build and run:

```bash
# Build image
docker build -t hetero-infer:latest .

# Run container
docker run -d \
  --name hetero-infer \
  --gpus all \
  -v /path/to/config.json:/etc/hetero-infer/config.json \
  -p 8080:8080 \
  hetero-infer:latest
```

### Kubernetes Deployment

Create `deployment.yaml`:

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

Deploy:

```bash
kubectl apply -f deployment.yaml
```

## Monitoring

### Log Levels

Set via `RUST_LOG` environment variable:

```bash
# Error only
RUST_LOG=error ./hetero-infer

# Warning and above
RUST_LOG=warn ./hetero-infer

# Info (default)
RUST_LOG=info ./hetero-infer

# Debug (verbose)
RUST_LOG=debug ./hetero-infer

# Trace (very verbose)
RUST_LOG=trace ./hetero-infer
```

### Metrics (Future)

Planned metrics endpoints:

- `GET /metrics` - Prometheus-compatible metrics
- Request rate, latency, batch size
- Memory utilization, queue depth
- GPU utilization, temperature

### Health Checks

Current status check via exit codes:

```bash
# In systemd service
ExecStartPre=/path/to/hetero-infer --version
```

## Performance Optimization

### CPU Optimization

1. **CPU Affinity**
   ```bash
   taskset -c 0-7 ./hetero-infer
   ```

2. **NUMA Awareness**
   ```bash
   numactl --cpunodebind=0 --membind=0 ./hetero-infer
   ```

### GPU Optimization

1. **Persistent Mode**
   ```bash
   sudo nvidia-smi -pm 1
   ```

2. **GPU Clock Settings**
   ```bash
   sudo nvidia-smi -ac 877,1530
   ```

3. **ECC Memory**
   ```bash
   # Disable ECC for better performance (if acceptable)
   sudo nvidia-smi -e 0
   ```

### Memory Optimization

1. **Huge Pages**
   ```bash
   # Enable transparent huge pages
   echo always > /sys/kernel/mm/transparent_hugepage/enabled
   ```

2. **Memory Limits**
   ```bash
   # In systemd service
   MemoryMax=64G
   MemorySwapMax=0
   ```

## Troubleshooting

### Build Issues

| Issue | Solution |
|-------|----------|
| `linker not found` | Install build-essential: `sudo apt install build-essential` |
| `CUDA not found` | Set `CUDA_HOME` environment variable |
| `proptest fails` | Run with `--test-threads=1` |

### Runtime Issues

| Issue | Solution |
|-------|----------|
| OOM errors | Reduce `max_num_blocks` or `max_batch_size` |
| Slow inference | Increase `max_batch_size`, enable CUDA Graphs |
| Request rejected | Check memory_threshold, reduce concurrent requests |
| GPU not used | Verify CUDA installation, check `nvidia-smi` |

### Debug Mode

```bash
# Enable backtrace
RUST_BACKTRACE=1 ./hetero-infer ...

# Full backtrace
RUST_BACKTRACE=full ./hetero-infer ...

# Debug logging
RUST_LOG=debug ./hetero-infer ...
```

## Security Considerations

1. **Run as non-root user**
   ```bash
   useradd -r -s /bin/false hetero
   ```

2. **Restrict file permissions**
   ```bash
   chmod 750 /opt/hetero-paged-infer
   chmod 640 /etc/hetero-infer/config.json
   ```

3. **Network isolation**
   - Use firewall rules
   - Deploy behind reverse proxy
   - Enable TLS for API endpoints

4. **Resource limits**
   - Set memory limits
   - Configure CPU quotas
   - Limit GPU access

## Backup and Recovery

### Configuration Backup

```bash
# Backup configuration
sudo tar czf hetero-infer-config-$(date +%Y%m%d).tar.gz \
  /etc/hetero-infer/

# Automated backup via cron
0 2 * * * tar czf /backup/hetero-infer-config-$(date +\%Y\%m\%d).tar.gz /etc/hetero-infer/
```

### Log Rotation

Create `/etc/logrotate.d/hetero-infer`:

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

*For API details, see [API.md](./API.md). For configuration options, see [CONFIGURATION.md](./CONFIGURATION.md).*
