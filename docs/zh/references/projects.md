# 项目引用

本页是相邻实现的阅读指南，而不是一张排行榜。

## vLLM

**仓库：** [github.com/vllm-project/vllm](https://github.com/vllm-project/vllm)

- **它为何与本仓库相关：** vLLM 是 PagedAttention 风格 LLM serving 最直接的开源参考对象。
- **它影响了哪个子系统：** KV Cache 设计语言、benchmark 预期，以及 API/serving 方向。
- **采纳了什么，以及刻意没有采纳什么：** 采纳了分页式 KV Cache 管理的整体系统视角；刻意没有把 vLLM 的生产结论、Python/CUDA 技术栈或特性完备度当成本仓库既有事实。

## TensorRT-LLM

**仓库：** [github.com/NVIDIA/TensorRT-LLM](https://github.com/NVIDIA/TensorRT-LLM)

- **它为何与本仓库相关：** 它定义了 NVIDIA 专用高优化 serving 的当前上限。
- **它影响了哪个子系统：** 未来 executor 目标、CUDA graph 讨论，以及对生产差距的诚实描述。
- **采纳了什么，以及刻意没有采纳什么：** 采纳了它作为性能参考点和内核设计标杆；刻意没有把厂商专属优化或其成熟度写成本仓库当前已达到的状态。

## llama.cpp

**仓库：** [github.com/ggerganov/llama.cpp](https://github.com/ggerganov/llama.cpp)

- **它为何与本仓库相关：** 它提供了一个很好的反例：优先追求部署简单与广泛可达性，而不是完全相同的 serving 假设。
- **它影响了哪个子系统：** 对比页的定位方式，以及对“本项目优化目标是什么/不是什么”的文档表达。
- **采纳了什么，以及刻意没有采纳什么：** 采纳了“可读的推理代码本身就是价值”的理念；刻意没有采纳其 CPU-first 范围、GGUF 生态或边缘设备定位。

## Text Generation Inference (TGI)

**仓库：** [github.com/huggingface/text-generation-inference](https://github.com/huggingface/text-generation-inference)

- **它为何与本仓库相关：** TGI 代表了一类把 API 体验与运维打包能力看得和核心内核同样重要的 serving 栈。
- **它影响了哪个子系统：** HTTP serving 预期、流式/API 叙述方式，以及对比页中的评估标准。
- **采纳了什么，以及刻意没有采纳什么：** 采纳了它作为面向产品的 serving 参考；刻意没有把它的生产运维成熟度描述成当前仓库已经具备的能力。

## 如何使用本页

把这些项目当作“设计空间地图”来读，然后回到[对比页](/zh/comparison/)，继续判断哪些优势属于本仓库本地事实，哪些仍是未来目标，哪些本来就属于其他系统。
