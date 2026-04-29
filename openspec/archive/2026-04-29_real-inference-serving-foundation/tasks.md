# Tasks

- [x] 1. 扩展 OpenSpec delta，明确真实 tokenizer、服务运行时与 OpenAI 兼容接口的契约
- [x] 2. 扩展 `EngineConfig`，增加 tokenizer / serving / backend bridge 配置，并补齐验证与序列化测试
- [x] 3. 在 `src/tokenizer.rs` 中加入 HuggingFace tokenizer 加载与工厂函数，并保留现有 simple tokenizer 路径
- [x] 4. 在 `src/engine.rs` / `src/error.rs` 中接入可失败的 tokenization 流程，确保配置驱动初始化可用
- [x] 5. 新增 `src/server.rs`，实现 `healthz`、`readyz`、`metrics`、`/v1/completions`、`/v1/chat/completions`
- [x] 6. 为 server 增加命令桥接后端和最小 SSE streaming 支持
- [x] 7. 更新 `src/main.rs`、`src/lib.rs` 与 README，暴露 server 启动入口和新配置说明
- [x] 8. 增加/更新测试，覆盖 tokenizer 装载、服务接口、streaming 和 metrics
