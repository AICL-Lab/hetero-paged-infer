# 变更日志

本项目的所有重要变更都将记录在此文件中。

格式基于 [Keep a Changelog](https://keepachangelog.com/zh-CN/1.0.0/)，
版本号遵循 [语义化版本](https://semver.org/lang/zh-CN/)。

## [Unreleased]

### 新增

- 完整的中文 API 文档 (rustdoc)
- GitHub Pages 文档站点增强
- CI/CD workflow 优化

## [0.1.0] - 2026-04-16

### 新增

- **文档系统**
  - 完整的中文 README
  - CONTRIBUTING.md 贡献指南
  - CHANGELOG.md 变更日志
  - config.example.json 示例配置
  - 所有公共 API 的 rustdoc 文档注释

- **PagedAttention KV Cache**
  - 分页式块管理，按需分配/释放物理块
  - BlockPool 实现 FIFO 空闲列表管理
  - PageTable 实现逻辑块到物理块的映射

- **Continuous Batching 调度器**
  - prefill/decode 分阶段管理
  - decode 优先调度策略
  - 内存压力感知与自动拒绝

- **推理引擎**
  - InferenceEngine 主编排器
  - 错误恢复策略（重试/跳过/重置/关闭）
  - EngineMetrics 指标收集

- **模块化架构**
  - TokenizerTrait 分词器接口
  - SchedulerTrait 调度器接口
  - GPUExecutorTrait GPU 执行器接口
  - KVCacheManagerTrait KV Cache 管理器接口

- **测试覆盖**
  - 78 个单元测试
  - 15 个属性测试 (proptest)
  - 13 个集成测试
  - 29 个文档测试

- **Mock 实现**
  - MockGPUExecutor 用于测试
  - SimpleTokenizer 字符级分词器

### 变更

- 2026-03-13: 修复 clippy 警告，优化 `div_ceil` 和 `HashMap::entry` 使用
- 2026-03-10: 统一 GitHub Actions workflow 配置

---

## 详细变更记录

### 2026-04-16 - 文档全面优化重构

**文档**
- 重写 README.md 为完整中文版
- 为所有源文件添加 rustdoc 注释
- 新增 CONTRIBUTING.md 贡献指南
- 新增 CHANGELOG.md 统一变更日志
- 新增 config.example.json 示例配置
- 更新 index.md GitHub Pages 首页

**CI/CD**
- 增强 CI workflow 配置
- 优化 Pages workflow 部署流程

### 2026-03-13 - Workflow CPU-safe CI 调整

**修复**
- 修复 `div_ceil` API 使用
- 修复 `HashMap::entry` 写法
- 使 CI 在 GitHub Hosted Runner 上正常运行

### 2026-03-10 - Workflow 深度标准化

**变更**
- 统一 workflow 权限配置
- 添加并发控制
- 添加路径过滤减少无效构建

---

## 版本说明

### [0.1.0] - 初始发布

首个发布版本，包含核心功能实现：

1. **KV Cache 管理**
   - 支持分页式内存管理
   - 支持动态块分配
   - 支持 copy-on-write（接口设计）

2. **调度器**
   - 支持 continuous batching
   - 支持 decode 优先
   - 支持内存压力检测

3. **推理引擎**
   - 支持请求提交和执行
   - 支持错误恢复
   - 支持指标收集

4. **测试**
   - 完整的单元测试覆盖
   - 属性测试验证不变量
   - 集成测试验证端到端流程

5. **文档**
   - 完整的中文文档
   - API 文档 (rustdoc)
   - 贡献指南
