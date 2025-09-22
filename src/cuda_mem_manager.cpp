#include "cuda_mem_manager.h"

#include <algorithm>
#include <cstddef>
#include <cstdint>
#include <stdexcept>

#ifdef BUILD_WITH_CUDA
#include <cuda_runtime.h>
#include "check_cuda_errors.h"
#endif

#include "image_operation.h"
#include "kernel.h"

CudaMemManager::CudaMemManager() {
}

CudaMemManager::~CudaMemManager() {
#ifdef BUILD_WITH_CUDA
  if (is_allocated_) {
    FreeMemory();
  }
#endif
}

void CudaMemManager::AllocateMemory() {
#ifndef BUILD_WITH_CUDA
  throw std::runtime_error("CUDA support disabled at build time.");
#else
  if (is_allocated_) {
    throw std::runtime_error("Memory already allocated.");
  }
  if (image_size_ == 0) {
    throw std::runtime_error("Image size is not set.");
  }
  if (image_operations_.empty() ||
      image_operations_[0] == ImageOperation::None) {
    throw std::runtime_error("Image operations are not set.");
  }
  bool sobel_required =
      std::find(image_operations_.begin(), image_operations_.end(),
                ImageOperation::Sobel) != image_operations_.end();
  bool prewitt_required =
      std::find(image_operations_.begin(), image_operations_.end(),
                ImageOperation::Prewitt) != image_operations_.end();
  bool separation_required =
      std::find(image_operations_.begin(), image_operations_.end(),
                ImageOperation::Separated) != image_operations_.end();
  bool gaussian_blur_required =
      std::find(image_operations_.begin(), image_operations_.end(),
                ImageOperation::GaussianBlur) != image_operations_.end();
  bool gaussian_blur_sep_required =
      std::find(image_operations_.begin(), image_operations_.end(),
                ImageOperation::GaussianBlurSep) != image_operations_.end();
  if (gaussian_blur_required || gaussian_blur_sep_required) {
    if (gaussian_kernel_size_ == 0 || gaussian_sigma_ < 0.0f) {
      throw std::runtime_error("Gaussian parameters are not set.");
    }
  }
  size_t image_size = image_size_ * sizeof(uint16_t);
  size_t required_memory = image_size * 2;
  if ((sobel_required || prewitt_required) && separation_required) {
    required_memory += image_size_ * sizeof(int) * 4;
  }
  if (gaussian_blur_required) {
    required_memory +=
        gaussian_kernel_size_ * gaussian_kernel_size_ * sizeof(float);
  } else if (gaussian_blur_sep_required) {
    required_memory += gaussian_kernel_size_ * sizeof(float);
    required_memory += image_size_ * sizeof(float);
  }
  if (!CheckFreeMemory(required_memory)) {
    throw std::runtime_error("Not enough free memory on the device.");
  }
  checkCudaErrors(cudaMalloc(&d_src_, image_size));
  checkCudaErrors(cudaMalloc(&d_dst_, image_size));
  if ((sobel_required || prewitt_required) && separation_required) {
    checkCudaErrors(cudaMalloc(&d_sep_g_x_, image_size_ * sizeof(int)));
    checkCudaErrors(cudaMalloc(&d_sep_g_y_, image_size_ * sizeof(int)));
    checkCudaErrors(cudaMalloc(&d_sep_result_x_, image_size_ * sizeof(int)));
    checkCudaErrors(cudaMalloc(&d_sep_result_y_, image_size_ * sizeof(int)));
  }
  if (gaussian_blur_required) {
    checkCudaErrors(cudaMalloc(
        &d_gaussian_kernel_,
        gaussian_kernel_size_ * gaussian_kernel_size_ * sizeof(float)));
  } else if (gaussian_blur_sep_required) {
    checkCudaErrors(
        cudaMalloc(&d_gaussian_kernel_, gaussian_kernel_size_ * sizeof(float)));
    checkCudaErrors(
        cudaMalloc(&d_gaussian_sep_temp_, image_size_ * sizeof(float)));
  }
  is_allocated_ = true;
  if (gaussian_blur_required || gaussian_blur_sep_required) {
    InitializeGaussianKernel();
  }
#endif
}

void CudaMemManager::FreeMemory() {
#ifndef BUILD_WITH_CUDA
  throw std::runtime_error("CUDA support disabled at build time.");
#else
  if (!is_allocated_) {
    throw std::runtime_error("Memory is not allocated.");
  }
  checkCudaErrors(cudaFree(d_src_));
  d_src_ = nullptr;
  checkCudaErrors(cudaFree(d_dst_));
  d_dst_ = nullptr;
  if (d_sep_g_x_ != nullptr) {
    checkCudaErrors(cudaFree(d_sep_g_x_));
    d_sep_g_x_ = nullptr;
  }
  if (d_sep_g_y_ != nullptr) {
    checkCudaErrors(cudaFree(d_sep_g_y_));
    d_sep_g_y_ = nullptr;
  }
  if (d_sep_result_x_ != nullptr) {
    checkCudaErrors(cudaFree(d_sep_result_x_));
    d_sep_result_x_ = nullptr;
  }
  if (d_sep_result_y_ != nullptr) {
    checkCudaErrors(cudaFree(d_sep_result_y_));
    d_sep_result_y_ = nullptr;
  }
  if (d_gaussian_kernel_ != nullptr) {
    checkCudaErrors(cudaFree(d_gaussian_kernel_));
    d_gaussian_kernel_ = nullptr;
  }
  if (d_gaussian_sep_temp_ != nullptr) {
    checkCudaErrors(cudaFree(d_gaussian_sep_temp_));
    d_gaussian_sep_temp_ = nullptr;
  }
  is_allocated_ = false;
#endif
}

void CudaMemManager::ReallocateMemory() {
#ifndef BUILD_WITH_CUDA
  throw std::runtime_error("CUDA support disabled at build time.");
#else
  if (is_allocated_) {
    FreeMemory();
  }
  AllocateMemory();
#endif
}

void CudaMemManager::CopyImageToDevice(const uint16_t* src) {
#ifndef BUILD_WITH_CUDA
  (void)src;  // suppress unused in non-CUDA builds
  throw std::runtime_error("CUDA support disabled at build time.");
#else
  if (!is_allocated_) {
    throw std::runtime_error("Memory is not allocated.");
  }
  checkCudaErrors(cudaMemcpy(d_src_, src, image_size_ * sizeof(uint16_t),
                             cudaMemcpyHostToDevice));
#endif
}

void CudaMemManager::CopyImageFromDevice(uint16_t* dst) {
#ifndef BUILD_WITH_CUDA
  (void)dst;  // suppress unused in non-CUDA builds
  throw std::runtime_error("CUDA support disabled at build time.");
#else
  if (!is_allocated_) {
    throw std::runtime_error("Memory is not allocated.");
  }
  checkCudaErrors(cudaMemcpy(dst, d_dst_, image_size_ * sizeof(uint16_t),
                             cudaMemcpyDeviceToHost));
#endif
}

void CudaMemManager::SetImageSize(size_t width, size_t height) {
  if (width == 0 || height == 0) {
    throw std::invalid_argument("Width and height must be positive numbers.");
  }
  image_size_ = width * height;
}

void CudaMemManager::SetImageSize(size_t image_size) {
  if (image_size == 0) {
    throw std::invalid_argument("Image size must be a positive number.");
  }
  image_size_ = image_size;
}

void CudaMemManager::SetGaussianParameters(size_t kernel_size, float sigma) {
  if (kernel_size % 2 == 0 || sigma < 0.0f) {
    throw std::invalid_argument(
        "Kernel size must be a positive odd number and sigma must be "
        "non-negative.");
  }
  gaussian_kernel_size_ = kernel_size;
  gaussian_sigma_ = sigma;
}

void CudaMemManager::InitializeGaussianKernel() {
  if (gaussian_kernel_size_ == 0 || gaussian_sigma_ < 0.0f) {
    throw std::runtime_error("Gaussian parameters are not set.");
  }
#ifdef BUILD_WITH_CUDA
  if (!is_allocated_) {
    throw std::runtime_error("Memory is not allocated.");
  }
  if (d_gaussian_kernel_ == nullptr) {
    throw std::runtime_error("Gaussian kernel memory is not allocated.");
  }
  if (std::find(image_operations_.begin(), image_operations_.end(),
                ImageOperation::GaussianBlur) != image_operations_.end()) {
    Kernel<float> kernel = Kernel<float>::GetGaussianKernel(
        gaussian_kernel_size_, gaussian_sigma_);
    float* gaussian_kernel;
    kernel.CopyKernelTo(&gaussian_kernel);
    checkCudaErrors(cudaMemcpy(
        d_gaussian_kernel_, gaussian_kernel,
        gaussian_kernel_size_ * gaussian_kernel_size_ * sizeof(float),
        cudaMemcpyHostToDevice));
    delete[] gaussian_kernel;
  } else if (std::find(image_operations_.begin(), image_operations_.end(),
                       ImageOperation::GaussianBlurSep) !=
             image_operations_.end()) {
    Kernel<float> kernel = Kernel<float>::GetGaussianKernelSep(
        gaussian_kernel_size_, gaussian_sigma_);
    float* gaussian_kernel;
    kernel.CopyKernelTo(&gaussian_kernel);
    checkCudaErrors(cudaMemcpy(d_gaussian_kernel_, gaussian_kernel,
                               gaussian_kernel_size_ * sizeof(float),
                               cudaMemcpyHostToDevice));
    delete[] gaussian_kernel;
  }
#endif
}

void CudaMemManager::ReallocateGaussianKernel() {
#ifdef BUILD_WITH_CUDA
  checkCudaErrors(cudaFree(d_gaussian_kernel_));
  if (std::find(image_operations_.begin(), image_operations_.end(),
                ImageOperation::GaussianBlur) != image_operations_.end()) {
    checkCudaErrors(cudaMalloc(
        &d_gaussian_kernel_,
        gaussian_kernel_size_ * gaussian_kernel_size_ * sizeof(float)));
  } else if (std::find(image_operations_.begin(), image_operations_.end(),
                       ImageOperation::GaussianBlurSep) !=
             image_operations_.end()) {
    checkCudaErrors(
        cudaMalloc(&d_gaussian_kernel_, gaussian_kernel_size_ * sizeof(float)));
  }
#else
  throw std::runtime_error("CUDA support disabled at build time.");
#endif
}

void CudaMemManager::CheckGaussianKernel(size_t kernel_size, float sigma) {
#ifndef BUILD_WITH_CUDA
  (void)kernel_size;
  (void)sigma;  // suppress unused in non-CUDA builds
  throw std::runtime_error("CUDA support disabled at build time.");
#else
  if (!is_allocated_) {
    throw std::runtime_error("Memory is not allocated.");
  }
  if (d_gaussian_kernel_ == nullptr) {
    throw std::runtime_error("Gaussian kernel memory is not allocated.");
  }
  if (kernel_size != gaussian_kernel_size_ || sigma != gaussian_sigma_) {
    gaussian_kernel_size_ = kernel_size;
    gaussian_sigma_ = sigma;
    ReallocateGaussianKernel();
    InitializeGaussianKernel();
  }
#endif
}

void CudaMemManager::SetImageOperations(const ImageOperation operations) {
  auto new_operations = DecomposeOperations(operations);
  if (new_operations != image_operations_) {
    image_operations_ = new_operations;
    if (is_allocated_) {
      FreeMemory();
      AllocateMemory();
    }
  }
}

bool CudaMemManager::CheckFreeMemory(size_t required_memory) const {
#ifndef BUILD_WITH_CUDA
  (void)required_memory;
  return false;
#else
  size_t free_memory, total_memory;
  cudaFree(nullptr);
  checkCudaErrors(cudaMemGetInfo(&free_memory, &total_memory));
  return free_memory > required_memory;
#endif
}

uint16_t* CudaMemManager::GetDeviceSrc() const {
  return d_src_;
}

uint16_t* CudaMemManager::GetDeviceDst() const {
  return d_dst_;
}

int* CudaMemManager::GetDeviceSepGx() const {
  return d_sep_g_x_;
}

int* CudaMemManager::GetDeviceSepGy() const {
  return d_sep_g_y_;
}

int* CudaMemManager::GetDeviceSepResultX() const {
  return d_sep_result_x_;
}

int* CudaMemManager::GetDeviceSepResultY() const {
  return d_sep_result_y_;
}

float* CudaMemManager::GetDeviceGaussianKernel() const {
  return d_gaussian_kernel_;
}

float* CudaMemManager::GetDeviceGaussianSepTemp() const {
  return d_gaussian_sep_temp_;
}

bool CudaMemManager::IsAllocated() const {
  return is_allocated_;
}
