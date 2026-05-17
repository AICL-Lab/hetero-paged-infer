# Projects

This page is a curated reading guide to adjacent implementations, not a scoreboard.

## vLLM

**Repository:** [github.com/vllm-project/vllm](https://github.com/vllm-project/vllm)

- **Why it matters to this repo:** vLLM is the most direct open-source reference for PagedAttention-style LLM serving.
- **Which subsystem it influenced:** KV-cache design language, benchmark expectations, and API/serving direction.
- **What was adopted vs deliberately not adopted:** Adopted the overall systems framing around paged KV-cache management; deliberately not adopted vLLM's production claims, Python/CUDA stack, or feature completeness as if they were already local facts.

## TensorRT-LLM

**Repository:** [github.com/NVIDIA/TensorRT-LLM](https://github.com/NVIDIA/TensorRT-LLM)

- **Why it matters to this repo:** It defines the current bar for highly optimized NVIDIA-specific serving.
- **Which subsystem it influenced:** Future executor ambitions, CUDA-graph discussion, and comparison-page honesty about production gaps.
- **What was adopted vs deliberately not adopted:** Adopted it as a performance reference point and kernel-design benchmark; deliberately not adopted vendor-specific optimizations or the claim that this repository currently matches them.

## llama.cpp

**Repository:** [github.com/ggerganov/llama.cpp](https://github.com/ggerganov/llama.cpp)

- **Why it matters to this repo:** It is a useful counterexample that prioritizes deployment simplicity and broad accessibility over the same serving assumptions.
- **Which subsystem it influenced:** Comparison-page positioning and documentation around what this project is, and is not, optimized for.
- **What was adopted vs deliberately not adopted:** Adopted the idea that readable inference code can be a product in itself; deliberately not adopted its CPU-first scope, GGUF ecosystem, or edge-device positioning.

## Text Generation Inference (TGI)

**Repository:** [github.com/huggingface/text-generation-inference](https://github.com/huggingface/text-generation-inference)

- **Why it matters to this repo:** TGI represents a serving stack where API ergonomics and operational packaging matter as much as core kernels.
- **Which subsystem it influenced:** HTTP-serving expectations, streaming/API framing, and comparison-page evaluation criteria.
- **What was adopted vs deliberately not adopted:** Adopted it as a reference for product-facing serving concerns; deliberately not adopted its production-operational maturity as if it were already present here.

## How to use this page

Use these projects to understand the surrounding design space. Then return to the [comparison page](/en/comparison/) and ask which advantages are local, which are aspirational, and which belong to other systems.
