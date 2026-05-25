#include <cuda_runtime.h>

__global__ void hetero_generate_next_tokens_kernel(unsigned int seed,
                                                   unsigned int vocab_size,
                                                   unsigned int* out_tokens,
                                                   unsigned long long num_sequences) {
  const unsigned long long index =
      static_cast<unsigned long long>(blockIdx.x) * blockDim.x + threadIdx.x;
  if (index >= num_sequences) {
    return;
  }

  out_tokens[index] = (seed + static_cast<unsigned int>(index)) % vocab_size;
}

extern "C" int hetero_cuda_compiled_with_nvcc() { return 1; }

extern "C" const char* hetero_cuda_backend_name() {
  return "nvcc-compiled-cuda-backend";
}

extern "C" int hetero_cuda_device_available() {
  int device_count = 0;
  const cudaError_t status = cudaGetDeviceCount(&device_count);
  if (status != cudaSuccess) {
    cudaGetLastError();
    return 0;
  }

  return device_count > 0 ? 1 : 0;
}

extern "C" int hetero_cuda_generate_next_tokens(const unsigned long long* seq_ids,
                                                unsigned long long num_sequences,
                                                unsigned int seed,
                                                unsigned int vocab_size,
                                                unsigned int* out_tokens,
                                                int* used_device) {
  if (vocab_size == 0) {
    return -3;
  }
  if (out_tokens == nullptr) {
    return -1;
  }
  if (num_sequences > 0 && seq_ids == nullptr) {
    return -2;
  }

  if (used_device != nullptr) {
    *used_device = 0;
  }

  if (num_sequences == 0) {
    return 0;
  }

  if (!hetero_cuda_device_available()) {
    for (unsigned long long i = 0; i < num_sequences; ++i) {
      out_tokens[i] =
          (seed + static_cast<unsigned int>(i)) % vocab_size;
    }
    return 0;
  }

  unsigned int* device_out_tokens = nullptr;
  const std::size_t output_size = sizeof(unsigned int) * num_sequences;
  cudaError_t status = cudaMalloc(&device_out_tokens, output_size);
  if (status != cudaSuccess) {
    cudaGetLastError();
    return -10;
  }

  if (used_device != nullptr) {
    *used_device = 1;
  }

  constexpr unsigned int kBlockSize = 128;
  const unsigned int grid_size =
      static_cast<unsigned int>((num_sequences + kBlockSize - 1) / kBlockSize);
  hetero_generate_next_tokens_kernel<<<grid_size, kBlockSize>>>(
      seed, vocab_size, device_out_tokens, num_sequences);

  status = cudaGetLastError();
  if (status != cudaSuccess) {
    cudaFree(device_out_tokens);
    return -11;
  }

  status = cudaDeviceSynchronize();
  if (status != cudaSuccess) {
    cudaFree(device_out_tokens);
    return -12;
  }

  status = cudaMemcpy(out_tokens, device_out_tokens, output_size, cudaMemcpyDeviceToHost);
  cudaFree(device_out_tokens);
  if (status != cudaSuccess) {
    return -13;
  }

  return 0;
}
