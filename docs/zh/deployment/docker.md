# Docker Deployment

## Quick Start with Docker

```bash
docker run -d \
  --name hetero-infer \
  --gpus all \
  -p 8080:8080 \
  ghcr.io/lessup/hetero-paged-infer:latest
```

## Dockerfile

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
COPY config.example.json /etc/hetero-infer/config.json

USER nobody
EXPOSE 8080

ENTRYPOINT ["hetero-infer"]
CMD ["--config", "/etc/hetero-infer/config.json"]
```

## docker-compose.yml

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

## Build and Run

```bash
# Build image
docker build -t hetero-infer:latest .

# Run with GPU
docker run -d --gpus all hetero-infer:latest

# Run with custom config
docker run -d \
  --gpus all \
  -v $(pwd)/config.json:/etc/hetero-infer/config.json \
  hetero-infer:latest
```
