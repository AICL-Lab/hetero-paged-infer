# 快速入门

## 环境要求

- Rust 1.70+ (2021 edition)
- Linux 环境（推荐 Ubuntu 20.04+）
- NVIDIA GPU + CUDA 11.x+（可选）

## 安装

```bash
# 克隆仓库
git clone https://github.com/LessUp/hetero-paged-infer.git
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

## 库用法

```rust
use hetero_infer::{EngineConfig, GenerationParams, InferenceEngine};

let config = EngineConfig::default();
let mut engine = InferenceEngine::new(config)?;

let params = GenerationParams {
    max_tokens: 100,
    temperature: 0.8,
    top_p: 0.95,
};

let request_id = engine.submit_request("你好，世界！", params)?;
let results = engine.run();

for result in results {
    println!("输出: {}", result.output_text);
}
```

## 目录

- [安装指南](installation.md) - 详细安装说明
- [配置说明](configuration.md) - 所有配置选项
- [API 参考](../api/core-types.md) - 完整 API 文档
