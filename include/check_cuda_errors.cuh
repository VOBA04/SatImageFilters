#include <cuda_runtime.h>
#include <iostream>

#define checkCudaErrors(val) Check((val), #val, __FILE__, __LINE__)

template <typename T>
void Check(T err, const char* const func, const char* const file,
           const int line) {
  if (err != cudaSuccess) {
    std::cerr << "CUDA error at: " << file << ":" << line << std::endl;
    std::cerr << cudaGetErrorString(err) << " " << func << std::endl;
    exit(1);
  }
}