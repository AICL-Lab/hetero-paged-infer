# 快速入门

## 环境要求

- Rust 1.82+（2021 edition）
- Linux 环境（推荐 Ubuntu 20.04+）
- NVIDIA GPU + CUDA 11.x+（可选，默认 CPU 后端不需要）

## 安装

```bash
# 克隆仓库
git clone https://github.com/AICL-Lab/hetero-paged-infer.git
cd hetero-paged-infer

# 构建发布版本
cargo build --release

# 运行测试
cargo test
```

## 首次推理

```bash
# 简单推理
./target/release/hetero-infer --input "你好，世界！" --max-tokens 50

# 自定义参数
./target/release/hetero-infer \
  --input "解释机器学习" \
  --max-tokens 200 \
  --temperature 0.8 \
  --top-p 0.95
```

> **注意：** 当前计算核心是 **mock 实现**：命令会完整走通调度与 KV Cache 分页流程，
> 但生成内容是**占位 token**（CR/TAB 等控制字符，不是自然语言）。这是预期行为。

实际输出形如（`Output:` 行为占位控制字符）：

```text
Heterogeneous Inference System
==============================
Configuration:
  Block size: 16
  Max blocks: 1024
  Max batch size: 32
  Max sequences: 256

Input: 你好，世界！
Generating up to 50 tokens...

Output: <占位 token>
Tokens generated: 5
```

## 库用法

```rust
use hetero_infer::{EngineConfig, GenerationParams, InferenceEngine};

fn main() -> Result<(), Box<dyn std::error::Error>> {
    let config = EngineConfig::default();
    let mut engine = InferenceEngine::new(config)?;

    let params = GenerationParams {
        max_tokens: 100,
        temperature: 0.8,
        top_p: 0.95,
    };

    let (request_id, _prompt_tokens) = engine.submit_request("你好，世界！", params)?;
    println!("已提交请求: {}", request_id);

    let results = engine.run();

    for result in results {
        if result.success {
            println!("输出: {}", result.output_text);
            println!("Token 数: {}", result.output_tokens.len());
        } else {
            println!("错误: {:?}", result.error);
        }
    }

    // 引擎指标（请求计数、内存利用率等）
    let metrics = engine.get_metrics();
    println!("内存利用率: {:.2}%", metrics.memory_utilization * 100.0);

    Ok(())
}
```

## HTTP 服务

加 `--serve` 启动 OpenAI 兼容服务（默认监听 `127.0.0.1:3000`，模型名 `hetero-infer`）：

```bash
./target/release/hetero-infer --serve
```

```bash
# 另开终端
curl http://127.0.0.1:3000/healthz
curl http://127.0.0.1:3000/v1/completions \
  -H 'Content-Type: application/json' \
  -d '{"model": "hetero-infer", "prompt": "你好，世界！", "max_tokens": 20}'
```

服务还提供 `/readyz`（就绪探针）、`/metrics`（Prometheus 文本格式）、
`/v1/chat/completions` 以及 token 级 SSE 流式输出；过载时返回 429 + `Retry-After`。
详见[生产部署](../deployment/production.md)。

## 目录

- [安装指南](installation) - 详细安装说明
- [配置说明](configuration) - 所有配置选项
- [API 参考](../api/core-types) - 完整 API 文档
