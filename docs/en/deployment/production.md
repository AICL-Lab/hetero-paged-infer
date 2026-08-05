# Production Deployment

## Overview

This guide covers deploying Hetero-Paged-Infer in production: running in server
mode, health checks, monitoring, and an operations checklist.

> **Current state:** The compute core is currently a **mock implementation**
> (it emits placeholder tokens, not natural language), and the default CPU
> backend **does not require a GPU**. Everything below targets the mock engine;
> GPU tuning advice does not apply to the current version.

## System Requirements

| Component | Requirement |
|-----------|-------------|
| **OS** | Linux (Ubuntu 20.04+ recommended) |
| **CPU** | x86_64 |
| **Memory** | 8 GB RAM |
| **Rust** | 1.82+ (2021 edition) |
| **GPU** | Not required (mock engine; the `cuda` feature is experimental, see [Installation](../setup/installation.md)) |

## Build

```bash
git clone https://github.com/AICL-Lab/hetero-paged-infer.git
cd hetero-paged-infer

cargo build --release
cargo test --release

./target/release/hetero-infer --help
```

## Running the Server

Production deployments should run in **server mode**: add `--serve` to start the
OpenAI-compatible HTTP server (binds `127.0.0.1:3000` by default, model name
`hetero-infer`).

> **Important:** without `--serve` (and without `--input`) the process prints a
> hint and **exits immediately**, which is unsuitable for a long-running
> deployment. To accept external connections, set `serving.host` to `0.0.0.0`
> in the config file (the default `127.0.0.1` only accepts loopback connections).

### Configuration File

Create `/etc/hetero-infer/config.json` (field reference: [Configuration](../setup/configuration.md)):

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

### Systemd Service

Create `/etc/systemd/system/hetero-infer.service`:

```ini
[Unit]
Description=Hetero-Paged-Infer inference server
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

Enable and start:

```bash
sudo systemctl daemon-reload
sudo systemctl enable hetero-infer
sudo systemctl start hetero-infer
```

Health checks go through the HTTP endpoints (there is **no** `--version` flag):

```bash
curl -fsS http://127.0.0.1:3000/healthz   # liveness probe
curl -fsS http://127.0.0.1:3000/readyz    # readiness probe
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
        image: hetero-infer:latest        # locally built image, see docker.md
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

Deploy (`serving.host` in the ConfigMap must be `0.0.0.0`):

```bash
kubectl create configmap hetero-infer-config --from-file=config.json=/etc/hetero-infer/config.json
kubectl apply -f deployment.yaml
```

> The mock engine does not consume GPUs, so the manifest contains **no**
> `nvidia.com/gpu` resource request; add one only when deploying an
> experimental `cuda` build.

For Docker deployment see [Docker Deployment](docker.md).

## Monitoring

### Health Checks

| Endpoint | Purpose |
|----------|---------|
| `GET /healthz` | Liveness probe; returns 200 while the process is up |
| `GET /readyz` | Readiness probe; returns 200 when the engine can accept requests |

### Metrics (Implemented)

`GET /metrics` returns Prometheus text-format metrics:

| Metric | Type | Meaning |
|--------|------|---------|
| `hetero_requests_total` | counter | Total requests received |
| `hetero_errors_total` | counter | Total error responses |
| `hetero_inflight_requests` | gauge | Requests currently in flight |
| `hetero_streaming_requests_total` | counter | Total SSE streaming requests |

```bash
curl http://127.0.0.1:3000/metrics
```

### Logging

Set the log level via the `RUST_LOG` environment variable:

```bash
RUST_LOG=info ./target/release/hetero-infer --serve    # recommended default
RUST_LOG=debug ./target/release/hetero-infer --serve   # for troubleshooting
```

## Production Checklist

- [ ] Start with `--serve` so the process stays resident
- [ ] Set `serving.host` for your network topology (`0.0.0.0` inside containers; keep the default `127.0.0.1` for loopback-only)
- [ ] Wire `/healthz` (liveness) and `/readyz` (readiness) probes into the orchestrator
- [ ] Scrape `/metrics` into Prometheus or another monitoring system
- [ ] Handle **429 + `Retry-After`** backpressure on the client side (the server rejects new requests when overloaded)
- [ ] Rely on **graceful shutdown**: on SIGTERM / Ctrl+C the process stops accepting connections and drains in-flight requests — do not send SIGKILL
- [ ] Set `RUST_LOG` and ship logs to your log pipeline

## Troubleshooting

| Issue | Solution |
|-------|----------|
| Process exits immediately after start | Make sure the command line includes `--serve` (server mode) |
| Service unreachable from outside the container | Set `serving.host` to `0.0.0.0` in the config file and map port 3000 |
| Requests rejected (429) | Overload backpressure: honor `Retry-After`, or raise `max_num_seqs` / `max_num_blocks` |
| Requests failing (memory pressure) | Lower `max_model_len` / `max_total_tokens`, or raise `max_num_blocks` |
| Build failure `linker not found` | Install build-essential: `sudo apt install build-essential` |

Debugging:

```bash
RUST_BACKTRACE=1 RUST_LOG=debug ./target/release/hetero-infer --serve
```

## Security Considerations

1. **Run as a non-root user**
   ```bash
   useradd -r -s /bin/false hetero
   ```
2. **Restrict file permissions**
   ```bash
   chmod 750 /opt/hetero-paged-infer
   chmod 640 /etc/hetero-infer/config.json
   ```
3. **Network isolation**: the server binds `127.0.0.1` by default; when exposing it externally, place it behind a reverse proxy with TLS.

---

*For API details, see [API Reference](../api/core-types). For configuration options, see [Configuration](../setup/configuration.md).*
