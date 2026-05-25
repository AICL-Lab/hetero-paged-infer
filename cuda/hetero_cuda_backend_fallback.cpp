extern "C" int hetero_cuda_compiled_with_nvcc() { return 0; }

extern "C" const char* hetero_cuda_backend_name() {
  return "host-fallback-cuda-backend";
}

extern "C" int hetero_cuda_device_available() { return 0; }

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

  for (unsigned long long i = 0; i < num_sequences; ++i) {
    out_tokens[i] = (seed + static_cast<unsigned int>(i)) % vocab_size;
  }

  return 0;
}
