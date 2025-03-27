#include <cstddef>
#include <cstdint>
#include "kernel.h"
#include "tiff_image.h"
#include "vector_types.h"
#include "cuda_runtime.h"
#include "cuda_runtime_api.h"
#include "driver_types.h"
#include "check_cuda_errors.cuh"

__constant__ int d_kernel_sobel[9] = {-1, 0, 1, -2, 0, 2, -1, 0, 1};
__constant__ int d_kernel_prewitt[9] = {-1, 0, 1, -1, 0, 1, -1, 0, 1};

__global__ void CudaSetKernel(uint16_t* src, uint16_t* dst, size_t height,
                              size_t width, int* kernel, size_t ksize,
                              bool rotate) {
  int i = blockIdx.y * blockDim.y + threadIdx.y;
  int j = blockIdx.x * blockDim.x + threadIdx.x;
  if (i < height && j < width) {
    if (rotate) {
      int g_x = 0, g_y = 0;
      for (int k = 0; k < ksize; k++) {
        for (int l = 0; l < ksize; l++) {
          int x = j + l - ksize / 2;
          int y = i + k - ksize / 2;
          if (x >= 0 && x < width && y >= 0 && y < height) {
            g_x += src[y * width + x] * kernel[k * ksize + l];
            g_y += src[y * width + x] * kernel[(ksize - 1 - l) * ksize + k];
          }
        }
      }
      dst[i * width + j] = abs(g_x) + abs(g_y);
    } else {
      int g = 0;
      for (int k = 0; k < ksize; k++) {
        for (int l = 0; l < ksize; l++) {
          int x = j + l - ksize / 2;
          int y = i + k - ksize / 2;
          if (x >= 0 && x < width && y >= 0 && y < height) {
            g += src[y * width + x] * kernel[k * ksize + l];
          }
        }
      }
      dst[i * width + j] = g;
    }
  }
}

__global__ void CudaSetSobelKernel(uint16_t* src, uint16_t* dst, size_t height,
                                   size_t width) {
  int i = blockIdx.y * blockDim.y + threadIdx.y;
  int j = blockIdx.x * blockDim.x + threadIdx.x;
  if (i < height && j < width) {
    int g_x = 0, g_y = 0;
    for (int k = 0; k < 3; k++) {
      for (int l = 0; l < 3; l++) {
        int x = j + l - 1;
        int y = i + k - 1;
        if (x >= 0 && x < width && y >= 0 && y < height) {
          g_x += src[y * width + x] * d_kernel_sobel[k * 3 + l];
          g_y += src[y * width + x] * d_kernel_sobel[(3 - 1 - l) * 3 + k];
        }
      }
    }
    dst[i * width + j] = abs(g_x) + abs(g_y);
  }
}

__global__ void CudaSetPrewittKernel(uint16_t* src, uint16_t* dst,
                                     size_t height, size_t width) {
  int i = blockIdx.y * blockDim.y + threadIdx.y;
  int j = blockIdx.x * blockDim.x + threadIdx.x;
  if (i < height && j < width) {
    int g_x = 0, g_y = 0;
    for (int k = 0; k < 3; k++) {
      for (int l = 0; l < 3; l++) {
        int x = j + l - 1;
        int y = i + k - 1;
        if (x >= 0 && x < width && y >= 0 && y < height) {
          g_x += src[y * width + x] * d_kernel_prewitt[k * 3 + l];
          g_y += src[y * width + x] * d_kernel_prewitt[(3 - 1 - l) * 3 + k];
        }
      }
    }
    dst[i * width + j] = abs(g_x) + abs(g_y);
  }
}

__global__ void CudaGaussianBlur(uint16_t* src, uint16_t* dst, size_t height,
                                 size_t width, double* kernel, size_t ksize) {
  int i = blockIdx.y * blockDim.y + threadIdx.y;
  int j = blockIdx.x * blockDim.x + threadIdx.x;
  if (i < height && j < width) {
    double sum = 0;
    for (int k = 0; k < ksize; k++) {
      for (int l = 0; l < ksize; l++) {
        int x = j + l - ksize / 2;
        int y = i + k - ksize / 2;
        if (x >= 0 && x < width && y >= 0 && y < height) {
          sum += src[y * width + x] * kernel[k * ksize + l];
        }
      }
    }
    dst[i * width + j] = static_cast<uint16_t>(sum);
  }
}

TIFFImage TIFFImage::SetKernelCuda(const Kernel<int>& kernel,
                                   const bool rotate) const {
  uint16_t* h_src = new uint16_t[width_ * height_];
  uint16_t* d_src;
  uint16_t* h_dst = new uint16_t[width_ * height_];
  uint16_t* d_dst;
  size_t image_size = width_ * height_ * sizeof(uint16_t);
  for (size_t i = 0; i < height_; i++) {
    for (size_t j = 0; j < width_; j++) {
      h_src[i * width_ + j] = Get(j, i);
    }
  }
  checkCudaErrors(cudaMalloc(&d_src, image_size));
  checkCudaErrors(cudaMemcpy(d_src, h_src, image_size, cudaMemcpyHostToDevice));
  checkCudaErrors(cudaMalloc(&d_dst, image_size));
  // dim3 threads(32, 32);  // 2D-блоки по 32x32
  // dim3 blocks((width_ + 31) / 32, (height_ + 31) / 32);
  dim3 threads(1024);
  dim3 blocks((width_ + 1023) / 1024, height_);
  if (kernel == kKernelSobel) {
    CudaSetSobelKernel<<<blocks, threads>>>(d_src, d_dst, height_, width_);
  } else if (kernel == kKernelPrewitt) {
    CudaSetPrewittKernel<<<blocks, threads>>>(d_src, d_dst, height_, width_);
  } else {
    int* h_kernel = new int[kernel.GetHeight() * kernel.GetWidth()];
    size_t kernel_size = kernel.GetHeight() * kernel.GetWidth() * sizeof(int);
    for (size_t i = 0; i < kernel.GetHeight(); i++) {
      for (size_t j = 0; j < kernel.GetWidth(); j++) {
        h_kernel[i * kernel.GetWidth() + j] = kernel.Get(i, j);
      }
    }
    int* d_kernel;
    checkCudaErrors(cudaMalloc(&d_kernel, kernel_size));
    checkCudaErrors(
        cudaMemcpy(d_kernel, h_kernel, kernel_size, cudaMemcpyHostToDevice));
    delete[] h_kernel;
    CudaSetKernel<<<blocks, threads>>>(d_src, d_dst, height_, width_, d_kernel,
                                       kernel.GetHeight(), rotate);
    checkCudaErrors(cudaFree(d_kernel));
  }
  checkCudaErrors(cudaDeviceSynchronize());
  checkCudaErrors(cudaMemcpy(h_dst, d_dst, image_size, cudaMemcpyDeviceToHost));
  checkCudaErrors(cudaFree(d_src));
  checkCudaErrors(cudaFree(d_dst));
  delete[] h_src;
  TIFFImage result(*this);
  for (size_t i = 0; i < height_; i++) {
    for (size_t j = 0; j < width_; j++) {
      result.Set(j, i, h_dst[i * width_ + j]);
    }
  }
  delete[] h_dst;
  return result;
}

TIFFImage TIFFImage::GaussianBlurCuda(const size_t size,
                                      const float sigma) const {
  uint16_t* h_src = new uint16_t[width_ * height_];
  uint16_t* d_src;
  uint16_t* h_dst = new uint16_t[width_ * height_];
  uint16_t* d_dst;
  size_t image_size = width_ * height_ * sizeof(uint16_t);
  for (size_t i = 0; i < height_; i++) {
    for (size_t j = 0; j < width_; j++) {
      h_src[i * width_ + j] = Get(j, i);
    }
  }
  checkCudaErrors(cudaMalloc(&d_src, image_size));
  checkCudaErrors(cudaMemcpy(d_src, h_src, image_size, cudaMemcpyHostToDevice));
  checkCudaErrors(cudaMalloc(&d_dst, image_size));
  Kernel<double> kernel = Kernel<double>::GetGaussianKernel(size, sigma);
  double* h_kernel = new double[kernel.GetHeight() * kernel.GetWidth()];
  double* d_kernel;
  for (size_t i = 0; i < kernel.GetHeight(); i++) {
    for (size_t j = 0; j < kernel.GetWidth(); j++) {
      h_kernel[i * kernel.GetWidth() + j] = kernel.Get(j, i);
    }
  }
  size_t kernel_size = kernel.GetHeight() * kernel.GetWidth() * sizeof(double);
  checkCudaErrors(cudaMalloc(&d_kernel, kernel_size));
  checkCudaErrors(
      cudaMemcpy(d_kernel, h_kernel, kernel_size, cudaMemcpyHostToDevice));
  delete[] h_kernel;
  // dim3 threads(32, 32);  // 2D-блоки по 32x32
  // dim3 blocks((width_ + 31) / 32, (height_ + 31) / 32);
  dim3 threads(1024);
  dim3 blocks((width_ + 1023) / 1024, height_);
  CudaGaussianBlur<<<blocks, threads>>>(d_src, d_dst, height_, width_, d_kernel,
                                        size);
  checkCudaErrors(cudaDeviceSynchronize());
  checkCudaErrors(cudaMemcpy(h_dst, d_dst, image_size, cudaMemcpyDeviceToHost));
  checkCudaErrors(cudaFree(d_src));
  checkCudaErrors(cudaFree(d_dst));
  checkCudaErrors(cudaFree(d_kernel));
  delete[] h_src;
  TIFFImage result(*this);
  for (size_t i = 0; i < height_; i++) {
    for (size_t j = 0; j < width_; j++) {
      result.Set(j, i, h_dst[i * width_ + j]);
    }
  }
  delete[] h_dst;
  return result;
}