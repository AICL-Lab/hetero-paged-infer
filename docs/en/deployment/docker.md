# Docker Deployment

> **No published images:** This project does not publish any Docker images (CI
> has no image-release pipeline). Please **build locally** from source. The mock
> engine does not need a GPU, so none of the examples below use `--gpus`.

## Dockerfile

Create a `Dockerfile` in the repository root:

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

# --serve is required, otherwise the process prints a hint and exits immediately
ENTRYPOINT ["hetero-infer"]
CMD ["--serve", "--config", "/etc/hetero-infer/config.json"]
```

## Build and Run

```bash
git clone https://github.com/AICL-Lab/hetero-paged-infer.git
cd hetero-paged-infer

# Build the image
docker build -t hetero-infer:latest .

# Run (server mode, mapping the default port 3000)
docker run -d \
  --name hetero-infer \
  -p 3000:3000 \
  hetero-infer:latest

# Verify
curl http://127.0.0.1:3000/healthz
```

> **Container networking note:** the default configuration binds `127.0.0.1`,
> which is **unreachable from outside the container**. To expose the service,
> mount a custom config with `serving.host` set to `0.0.0.0`:
>
> ```bash
> docker run -d \
>   --name hetero-infer \
>   -p 3000:3000 \
>   -v $(pwd)/config.json:/etc/hetero-infer/config.json:ro \
>   hetero-infer:latest
> ```
>
> where `config.json` contains at least:
>
> ```json
> {
>   "serving": { "host": "0.0.0.0", "port": 3000, "model_name": "hetero-infer" }
> }
> ```
>
> (Omitted fields fall back to their defaults.)

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

`config.json` must set `serving.host` to `0.0.0.0` (see note above).

---

For server ports, probes, and metrics endpoints, see [Production Deployment](production.md).
