#include "cuda_mem_manager.h"

#include <algorithm>
#include <cstddef>
#include <cstdint>
#include <stdexcept>

#include "check_cuda_errors.h"
#include "image_operation.h"
#include "kernel.h"

CudaMemManager::CudaMemManager() {
}

CudaMemManager::~CudaMemManager() {
  if (is_allocated_) {
    FreeMemory();
  }
}

void CudaMemManager::AllocateMemory() {
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
  // allocate second slot for ping-pong if enough memory
  if (CheckFreeMemory(image_size * 2)) {
    checkCudaErrors(cudaMalloc(&d_src_2_, image_size));
    checkCudaErrors(cudaMalloc(&d_dst_2_, image_size));
    pingpong_allocated_ = true;
  } else {
    d_src_2_ = nullptr;
    d_dst_2_ = nullptr;
    pingpong_allocated_ = false;
  }
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
}

void CudaMemManager::FreeMemory() {
  if (!is_allocated_) {
    throw std::runtime_error("Memory is not allocated.");
  }
  checkCudaErrors(cudaFree(d_src_));
  d_src_ = nullptr;
  checkCudaErrors(cudaFree(d_dst_));
  d_dst_ = nullptr;
  if (d_src_2_ != nullptr) {
    checkCudaErrors(cudaFree(d_src_2_));
    d_src_2_ = nullptr;
  }
  if (d_dst_2_ != nullptr) {
    checkCudaErrors(cudaFree(d_dst_2_));
    d_dst_2_ = nullptr;
  }
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
}

void CudaMemManager::ReallocateMemory() {
  if (is_allocated_) {
    FreeMemory();
  }
  AllocateMemory();
}

void CudaMemManager::CopyImageToDevice(const uint16_t* src) {
  if (!is_allocated_) {
    throw std::runtime_error("Memory is not allocated.");
  }
  checkCudaErrors(cudaMemcpy(d_src_, src, image_size_ * sizeof(uint16_t),
                             cudaMemcpyHostToDevice));
}

void CudaMemManager::CopyImageToDeviceAsync(const uint16_t* src,
                                            cudaStream_t stream) {
  if (!is_allocated_) {
    throw std::runtime_error("Memory is not allocated.");
  }
  checkCudaErrors(cudaMemcpyAsync(d_src_, src, image_size_ * sizeof(uint16_t),
                                  cudaMemcpyHostToDevice, stream));
}

void CudaMemManager::CopyImageToDeviceAsyncSlot(const uint16_t* src,
                                                cudaStream_t stream, int slot) {
  if (!is_allocated_) {
    throw std::runtime_error("Memory is not allocated.");
  }
  uint16_t* target = (slot == 1 && pingpong_allocated_) ? d_src_2_ : d_src_;
  if (target == nullptr) {
    throw std::runtime_error("Ping-pong slot not available.");
  }
  checkCudaErrors(cudaMemcpyAsync(target, src, image_size_ * sizeof(uint16_t),
                                  cudaMemcpyHostToDevice, stream));
}

void CudaMemManager::CopyImageFromDeviceAsync(uint16_t* dst,
                                              cudaStream_t stream) {
  if (!is_allocated_) {
    throw std::runtime_error("Memory is not allocated.");
  }
  checkCudaErrors(cudaMemcpyAsync(dst, d_dst_, image_size_ * sizeof(uint16_t),
                                  cudaMemcpyDeviceToHost, stream));
}

void CudaMemManager::CopyImageFromDeviceAsyncSlot(uint16_t* dst,
                                                  cudaStream_t stream,
                                                  int slot) {
  if (!is_allocated_) {
    throw std::runtime_error("Memory is not allocated.");
  }
  uint16_t* source = (slot == 1 && pingpong_allocated_) ? d_dst_2_ : d_dst_;
  if (source == nullptr) {
    throw std::runtime_error("Ping-pong slot not available.");
  }
  checkCudaErrors(cudaMemcpyAsync(dst, source, image_size_ * sizeof(uint16_t),
                                  cudaMemcpyDeviceToHost, stream));
}

void CudaMemManager::CopyImageFromDevice(uint16_t* dst) {
  if (!is_allocated_) {
    throw std::runtime_error("Memory is not allocated.");
  }
  checkCudaErrors(cudaMemcpy(dst, d_dst_, image_size_ * sizeof(uint16_t),
                             cudaMemcpyDeviceToHost));
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
}

void CudaMemManager::ReallocateGaussianKernel() {
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
}

void CudaMemManager::CheckGaussianKernel(size_t kernel_size, float sigma) {
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
  size_t free_memory, total_memory;
  cudaFree(nullptr);
  checkCudaErrors(cudaMemGetInfo(&free_memory, &total_memory));
  return free_memory > required_memory;
}

uint16_t* CudaMemManager::GetDeviceSrc() const {
  return d_src_;
}

uint16_t* CudaMemManager::GetDeviceSrcSlot(int slot) const {
  return (slot == 1 && pingpong_allocated_) ? d_src_2_ : d_src_;
}

uint16_t* CudaMemManager::GetDeviceDst() const {
  return d_dst_;
}

uint16_t* CudaMemManager::GetDeviceDstSlot(int slot) const {
  return (slot == 1 && pingpong_allocated_) ? d_dst_2_ : d_dst_;
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
