#include <cstddef>
#include <cstdint>
#include "kernel.h"
#include "tiff_image.h"
#include <vector_types.h>
#include <cuda_runtime.h>
#include <cuda_runtime_api.h>
#include <driver_types.h>
#include "check_cuda_errors.cuh"

__constant__ int d_kernel_sobel[9] = {-1, 0, 1, -2, 0, 2, -1, 0, 1};
__constant__ int d_kernel_prewitt[9] = {-1, 0, 1, -1, 0, 1, -1, 0, 1};
__constant__ int d_kernel_sobel_sep[3] = {1, 2, 1};
__constant__ int d_kernel_prewitt_sep[3] = {1, 1, 1};
__constant__ int d_kernel_gradient[3] = {-1, 0, 1};

/**
 * @brief Применяет пользовательское ядро к входному изображению.
 *
 * @param src Указатель на исходное изображение на устройстве.
 * @param dst Указатель на результирующее изображение на устройстве.
 * @param height Высота изображения.
 * @param width Ширина изображения.
 * @param kernel Указатель на ядро на устройстве.
 * @param ksize Размер ядра (предполагается квадратное).
 * @param rotate Указывает, применять ли ядро в повернутом виде.
 */
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

/**
 * @brief Применяет ядро Собеля для вычисления границ.
 *
 * @param src Указатель на исходное изображение на устройстве.
 * @param dst Указатель на результирующее изображение на устройстве.
 * @param height Высота изображения.
 * @param width Ширина изображения.
 */
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

/**
 * @brief Применяет ядро Собеля в раздельной форме для сглаживания.
 *
 * @param src Указатель на исходное изображение на устройстве.
 * @param g_x Указатель на градиент по оси X.
 * @param g_y Указатель на градиент по оси Y.
 * @param height Высота изображения.
 * @param width Ширина изображения.
 */
__global__ void CudaSetSobelKernelSmooth(uint16_t* src, uint16_t* g_x,
                                         uint16_t* g_y, size_t height,
                                         size_t width) {
  int i = blockIdx.y * blockDim.y + threadIdx.y;
  int j = blockIdx.x * blockDim.x + threadIdx.x;
  if (i < height && j < width) {
    int sum_x = 0, sum_y = 0;
    for (int k = 0; k < 3; k++) {
      int x = j + k - 1;
      int y = i + k - 1;
      if (x >= 0 && x < width) {
        sum_x += src[i * width + x] * d_kernel_sobel_sep[k];
      }
      if (y >= 0 && y < height) {
        sum_y += src[y * width + j] * d_kernel_sobel_sep[k];
      }
    }
    g_x[i * width + j] = sum_x;
    g_y[i * width + j] = sum_y;
  }
}

/**
 * @brief Применяет ядро Превитта для вычисления границ.
 *
 * @param src Указатель на исходное изображение на устройстве.
 * @param dst Указатель на результирующее изображение на устройстве.
 * @param height Высота изображения.
 * @param width Ширина изображения.
 */
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

/**
 * @brief Применяет ядро Превитта в раздельной форме для усреднения.
 *
 * @param src Указатель на исходное изображение на устройстве.
 * @param g_x Указатель на градиент по оси X.
 * @param g_y Указатель на градиент по оси Y.
 * @param height Высота изображения.
 * @param width Ширина изображения.
 */
__global__ void CudaSetPrewittKernelAverage(uint16_t* src, uint16_t* g_x,
                                            uint16_t* g_y, size_t height,
                                            size_t width) {
  int i = blockIdx.y * blockDim.y + threadIdx.y;
  int j = blockIdx.x * blockDim.x + threadIdx.x;
  if (i < height && j < width) {
    int sum_x = 0, sum_y = 0;
    for (int k = 0; k < 3; k++) {
      int x = j + k - 1;
      int y = i + k - 1;
      if (x >= 0 && x < width) {
        sum_x += src[i * width + x] * d_kernel_prewitt_sep[k];
      }
      if (y >= 0 && y < height) {
        sum_y += src[y * width + j] * d_kernel_prewitt_sep[k];
      }
    }
    g_x[i * width + j] = sum_x;
    g_y[i * width + j] = sum_y;
  }
}

/**
 * @brief Вычисляет разность градиентов с использованием раздельного ядра.
 *
 * @param g_x Указатель на градиент по оси X.
 * @param g_y Указатель на градиент по оси Y.
 * @param result_x Указатель на результат по оси X.
 * @param result_y Указатель на результат по оси Y.
 * @param height Высота изображения.
 * @param width Ширина изображения.
 */
__global__ void CudaSepKernelDiff(uint16_t* g_x, uint16_t* g_y,
                                  uint16_t* result_x, uint16_t* result_y,
                                  size_t height, size_t width) {
  int i = blockIdx.y * blockDim.y + threadIdx.y;
  int j = blockIdx.x * blockDim.x + threadIdx.x;
  if (i < height && j < width) {
    int sum_x = 0;
    int sum_y = 0;
    for (int k = 0; k < 3; k++) {
      int x = j + k - 1;
      int y = i + k - 1;
      if (x >= 0 && x < width) {
        sum_y += g_y[i * width + x] * d_kernel_gradient[k];
      }
      if (y >= 0 && y < height) {
        sum_x += g_x[y * width + j] * d_kernel_gradient[k];
      }
    }
    result_x[i * width + j] = sum_x;
    result_y[i * width + j] = sum_y;
  }
}

/**
 * @brief Складывает абсолютные значения двух матриц поэлементно.
 *
 * @param mtx1 Указатель на первую матрицу.
 * @param mtx2 Указатель на вторую матрицу.
 * @param result Указатель на результирующую матрицу.
 * @param height Высота матриц.
 * @param width Ширина матриц.
 */
__global__ void CudaAddAbsMtx(uint16_t* mtx1, uint16_t* mtx2, uint16_t* result,
                              size_t height, size_t width) {
  int i = blockIdx.y * blockDim.y + threadIdx.y;
  int j = blockIdx.x * blockDim.x + threadIdx.x;
  if (i < height && j < width) {
    int sum = abs(mtx1[i * width + j]) + abs(mtx2[i * width + j]);
    result[i * width + j] = static_cast<uint16_t>((sum > 65535 ? 65535 : sum));
  }
}

/**
 * @brief Применяет размытие по Гауссу к входному изображению.
 *
 * @param src Указатель на исходное изображение на устройстве.
 * @param dst Указатель на результирующее изображение на устройстве.
 * @param height Высота изображения.
 * @param width Ширина изображения.
 * @param kernel Указатель на ядро Гаусса на устройстве.
 * @param ksize Размер ядра (предполагается квадратное).
 */
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

/**
 * @brief Применяет раздельное размытие по Гауссу к входному изображению.
 *
 * @param src Указатель на исходное изображение на устройстве.
 * @param dst Указатель на результирующее изображение на устройстве.
 * @param height Высота изображения.
 * @param width Ширина изображения.
 * @param kernel Указатель на ядро Гаусса на устройстве.
 * @param ksize Размер ядра.
 * @param horizontal_vertical Указывает, применять ли ядро горизонтально или
 * вертикально.
 */
__global__ void CudaGaussianBlurSep(uint16_t* src, uint16_t* dst, size_t height,
                                    size_t width, double* kernel, size_t ksize,
                                    bool horizontal_vertical) {
  int i = blockIdx.y * blockDim.y + threadIdx.y;
  int j = blockIdx.x * blockDim.x + threadIdx.x;
  if (i < height && j < width) {
    double sum = 0;
    if (horizontal_vertical) {
      for (int k = 0; k < ksize; k++) {
        int x = j + k - ksize / 2;
        if (x >= 0 && x < width) {
          sum += src[i * width + x] * kernel[k];
        }
      }
    } else {
      for (int k = 0; k < ksize; k++) {
        int y = i + k - ksize / 2;
        if (y >= 0 && y < height) {
          sum += src[y * width + j] * kernel[k];
        }
      }
    }
    dst[i * width + j] = static_cast<uint16_t>(sum);
  }
}

TIFFImage TIFFImage::SetKernelCuda(const Kernel<int>& kernel,
                                   const bool rotate) const {
  uint16_t* h_src = image_;
  uint16_t* d_src;
  uint16_t* h_dst = new uint16_t[width_ * height_];
  uint16_t* d_dst;
  size_t image_size = width_ * height_ * sizeof(uint16_t);
  size_t free_memory, total_memory;
  checkCudaErrors(cudaMemGetInfo(&free_memory, &total_memory));
  if (free_memory < image_size * 2) {
    throw std::runtime_error("Изображение слишком большое для GPU");
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
  TIFFImage result(*this);
  std::memcpy(result.image_, h_dst, image_size);
  delete[] h_dst;
  return result;
}

TIFFImage TIFFImage::SetKernelSobelSepCuda() const {
  uint16_t* h_src = image_;
  uint16_t* d_src;
  uint16_t* d_g_x;
  uint16_t* d_g_y;
  uint16_t* d_result_x;
  uint16_t* d_result_y;
  uint16_t* h_dst = new uint16_t[width_ * height_];
  uint16_t* d_dst;
  size_t image_size = width_ * height_ * sizeof(uint16_t);
  size_t free_memory, total_memory;
  checkCudaErrors(cudaMemGetInfo(&free_memory, &total_memory));
  if (free_memory < image_size * 6) {
    throw std::runtime_error("Изображение слишком большое для GPU");
  }
  checkCudaErrors(cudaMalloc(&d_src, image_size));
  checkCudaErrors(cudaMemcpy(d_src, h_src, image_size, cudaMemcpyHostToDevice));
  checkCudaErrors(cudaMalloc(&d_g_x, image_size));
  checkCudaErrors(cudaMalloc(&d_g_y, image_size));
  checkCudaErrors(cudaMalloc(&d_result_x, image_size));
  checkCudaErrors(cudaMalloc(&d_result_y, image_size));
  checkCudaErrors(cudaMalloc(&d_dst, image_size));
  dim3 threads(32, 32);  // 2D-блоки по 32x32
  dim3 blocks((width_ + 31) / 32, (height_ + 31) / 32);
  // dim3 threads(1024);
  // dim3 blocks((width_ + 1023) / 1024, height_);
  CudaSetSobelKernelSmooth<<<blocks, threads>>>(d_src, d_g_x, d_g_y, height_,
                                                width_);
  checkCudaErrors(cudaDeviceSynchronize());
  CudaSepKernelDiff<<<blocks, threads>>>(d_g_x, d_g_y, d_result_x, d_result_y,
                                         height_, width_);
  checkCudaErrors(cudaDeviceSynchronize());
  CudaAddAbsMtx<<<blocks, threads>>>(d_result_x, d_result_y, d_dst, height_,
                                     width_);
  checkCudaErrors(cudaDeviceSynchronize());
  checkCudaErrors(cudaMemcpy(h_dst, d_dst, image_size, cudaMemcpyDeviceToHost));
  checkCudaErrors(cudaFree(d_src));
  checkCudaErrors(cudaFree(d_g_x));
  checkCudaErrors(cudaFree(d_g_y));
  checkCudaErrors(cudaFree(d_result_x));
  checkCudaErrors(cudaFree(d_result_y));
  checkCudaErrors(cudaFree(d_dst));
  TIFFImage result(*this);
  std::memcpy(result.image_, h_dst, image_size);
  delete[] h_dst;
  return result;
}

TIFFImage TIFFImage::SetKernelPrewittSepCuda() const {
  uint16_t* h_src = image_;
  uint16_t* d_src;
  uint16_t* d_g_x;
  uint16_t* d_g_y;
  uint16_t* d_result_x;
  uint16_t* d_result_y;
  uint16_t* h_dst = new uint16_t[width_ * height_];
  uint16_t* d_dst;
  size_t image_size = width_ * height_ * sizeof(uint16_t);
  size_t free_memory, total_memory;
  checkCudaErrors(cudaMemGetInfo(&free_memory, &total_memory));
  if (free_memory < image_size * 6) {
    throw std::runtime_error("Изображение слишком большое для GPU");
  }
  checkCudaErrors(cudaMalloc(&d_src, image_size));
  checkCudaErrors(cudaMemcpy(d_src, h_src, image_size, cudaMemcpyHostToDevice));
  checkCudaErrors(cudaMalloc(&d_g_x, image_size));
  checkCudaErrors(cudaMalloc(&d_g_y, image_size));
  checkCudaErrors(cudaMalloc(&d_result_x, image_size));
  checkCudaErrors(cudaMalloc(&d_result_y, image_size));
  checkCudaErrors(cudaMalloc(&d_dst, image_size));
  // dim3 threads(32, 32);  // 2D-блоки по 32x32
  // dim3 blocks((width_ + 31) / 32, (height_ + 31) / 32);
  dim3 threads(1024);
  dim3 blocks((width_ + 1023) / 1024, height_);
  CudaSetPrewittKernelAverage<<<blocks, threads>>>(d_src, d_g_x, d_g_y, height_,
                                                   width_);
  checkCudaErrors(cudaDeviceSynchronize());
  CudaSepKernelDiff<<<blocks, threads>>>(d_g_x, d_g_y, d_result_x, d_result_y,
                                         height_, width_);
  checkCudaErrors(cudaDeviceSynchronize());
  CudaAddAbsMtx<<<blocks, threads>>>(d_result_x, d_result_y, d_dst, height_,
                                     width_);
  checkCudaErrors(cudaDeviceSynchronize());
  checkCudaErrors(cudaMemcpy(h_dst, d_dst, image_size, cudaMemcpyDeviceToHost));
  checkCudaErrors(cudaFree(d_src));
  checkCudaErrors(cudaFree(d_g_x));
  checkCudaErrors(cudaFree(d_g_y));
  checkCudaErrors(cudaFree(d_result_x));
  checkCudaErrors(cudaFree(d_result_y));
  checkCudaErrors(cudaFree(d_dst));
  TIFFImage result(*this);
  std::memcpy(result.image_, h_dst, image_size);
  delete[] h_dst;
  return result;
}

TIFFImage TIFFImage::GaussianBlurCuda(const size_t size,
                                      const float sigma) const {
  uint16_t* h_src = image_;
  uint16_t* d_src;
  uint16_t* h_dst = new uint16_t[width_ * height_];
  uint16_t* d_dst;
  size_t image_size = width_ * height_ * sizeof(uint16_t);
  size_t free_memory, total_memory;
  checkCudaErrors(cudaMemGetInfo(&free_memory, &total_memory));
  if (free_memory < image_size * 2) {
    throw std::runtime_error("Изображение слишком большое для GPU");
  }
  checkCudaErrors(cudaMalloc(&d_src, image_size));
  checkCudaErrors(cudaMemcpy(d_src, h_src, image_size, cudaMemcpyHostToDevice));
  checkCudaErrors(cudaMalloc(&d_dst, image_size));
  Kernel<double> kernel = Kernel<double>::GetGaussianKernel(size, sigma);
  double* h_kernel;
  double* d_kernel;
  size_t kernel_size = kernel.GetHeight() * kernel.GetWidth() * sizeof(double);
  checkCudaErrors(cudaMemGetInfo(&free_memory, &total_memory));
  if (free_memory < kernel_size) {
    throw std::runtime_error("Ядро слишком большое для GPU");
  }
  kernel.CopyKernelTo(&h_kernel);
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
  TIFFImage result(*this);
  std::memcpy(result.image_, h_dst, image_size);
  delete[] h_dst;
  return result;
}

TIFFImage TIFFImage::GaussianBlurSepCuda(const size_t size,
                                         const float sigma) const {
  uint16_t* h_src = image_;
  uint16_t* d_src;
  uint16_t* d_temp;
  uint16_t* h_dst = new uint16_t[width_ * height_];
  uint16_t* d_dst;
  size_t image_size = width_ * height_ * sizeof(uint16_t);
  size_t free_memory, total_memory;
  checkCudaErrors(cudaMemGetInfo(&free_memory, &total_memory));
  if (free_memory < image_size * 3) {
    throw std::runtime_error("Изображение слишком большое для GPU");
  }
  checkCudaErrors(cudaMalloc(&d_src, image_size));
  checkCudaErrors(cudaMemcpy(d_src, h_src, image_size, cudaMemcpyHostToDevice));
  checkCudaErrors(cudaMalloc(&d_temp, image_size));
  checkCudaErrors(cudaMalloc(&d_dst, image_size));
  Kernel<double> kernel = Kernel<double>::GetGaussianKernelSep(size, sigma);
  double* h_kernel;
  double* d_kernel;
  size_t kernel_size = kernel.GetHeight() * kernel.GetWidth() * sizeof(double);
  checkCudaErrors(cudaMemGetInfo(&free_memory, &total_memory));
  if (free_memory < kernel_size) {
    throw std::runtime_error("Ядро слишком большое для GPU");
  }
  kernel.CopyKernelTo(&h_kernel);
  checkCudaErrors(cudaMalloc(&d_kernel, kernel_size));
  checkCudaErrors(
      cudaMemcpy(d_kernel, h_kernel, kernel_size, cudaMemcpyHostToDevice));
  delete[] h_kernel;
  // dim3 threads(32, 32);  // 2D-блоки по 32x32
  // dim3 blocks((width_ + 31) / 32, (height_ + 31) / 32);
  dim3 threads(1024);
  dim3 blocks((width_ + 1023) / 1024, height_);
  CudaGaussianBlurSep<<<blocks, threads>>>(d_src, d_temp, height_, width_,
                                           d_kernel, size, true);
  checkCudaErrors(cudaDeviceSynchronize());
  CudaGaussianBlurSep<<<blocks, threads>>>(d_temp, d_dst, height_, width_,
                                           d_kernel, size, false);
  checkCudaErrors(cudaDeviceSynchronize());
  checkCudaErrors(cudaMemcpy(h_dst, d_dst, image_size, cudaMemcpyDeviceToHost));
  checkCudaErrors(cudaFree(d_src));
  checkCudaErrors(cudaFree(d_temp));
  checkCudaErrors(cudaFree(d_dst));
  checkCudaErrors(cudaFree(d_kernel));
  TIFFImage result(*this);
  std::memcpy(result.image_, h_dst, image_size);
  delete[] h_dst;
  return result;
}