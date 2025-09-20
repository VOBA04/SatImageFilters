#pragma once
#include <cstddef>
#include <cstdint>
#include <vector>
#include "image_operation.h"

class CudaMemManager {
  size_t image_size_ = 0;
  size_t gaussian_kernel_size_ = 0;
  float gaussian_sigma_ = 0.0f;
  uint16_t* d_src_ = nullptr;
  uint16_t* d_dst_ = nullptr;
  int* d_sep_g_x_ = nullptr;
  int* d_sep_g_y_ = nullptr;
  int* d_sep_result_x_ = nullptr;
  int* d_sep_result_y_ = nullptr;
  float* d_gaussian_kernel_ = nullptr;
  float* d_gaussian_sep_temp_ = nullptr;
  std::vector<ImageOperation> image_operations_;
  bool is_allocated_ = false;

  void InitializeGaussianKernel();
  void ReallocateGaussianKernel();

 public:
  CudaMemManager();
  ~CudaMemManager();
  void AllocateMemory();
  void FreeMemory();
  void ReallocateMemory();
  void CopyImageToDevice(const uint16_t* src);
  void CopyImageFromDevice(uint16_t* dst);
  void SetImageSize(size_t width, size_t height);
  void SetImageSize(size_t image_size);
  void SetGaussianParameters(size_t kernel_size, float sigma = 0.0f);
  void CheckGaussianKernel(size_t kernel_size, float sigma = 0.0f);
  void SetImageOperations(const ImageOperation operations);
  bool CheckFreeMemory(size_t required_memory) const;
  uint16_t* GetDeviceSrc() const;
  uint16_t* GetDeviceDst() const;
  int* GetDeviceSepGx() const;
  int* GetDeviceSepGy() const;
  int* GetDeviceSepResultX() const;
  int* GetDeviceSepResultY() const;
  float* GetDeviceGaussianKernel() const;
  float* GetDeviceGaussianSepTemp() const;
  bool IsAllocated() const;
};