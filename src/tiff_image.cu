#include <cmath>
#include <cstddef>
#include <cstdint>
#include "kernel.h"
#include "tiff_image.h"
#include <vector_types.h>
#include <cuda_runtime.h>
#include <cuda_runtime_api.h>
#include <driver_types.h>
#include "check_cuda_errors.h"

const uint8_t kBlockSize = 16;

__constant__ int d_kernel_sobel[9] = {-1, 0, 1, -2, 0, 2, -1, 0, 1};
__constant__ int d_kernel_prewitt[9] = {-1, 0, 1, -1, 0, 1, -1, 0, 1};
__constant__ int d_kernel_sobel_sep[3] = {1, 2, 1};
__constant__ int d_kernel_prewitt_sep[3] = {1, 1, 1};
__constant__ int d_kernel_gradient[3] = {-1, 0, 1};

bool CheckFreeMem(size_t required_memory) {
  size_t free_memory, total_memory;
  cudaFree(nullptr);
  checkCudaErrors(cudaMemGetInfo(&free_memory, &total_memory));
  return free_memory > required_memory;
}

__device__ __forceinline__ int Clamp(int val, int min_val, int max_val) {
  return max(min_val, min(val, max_val));
}

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
          x = Clamp(x, 0, width - 1);
          y = Clamp(y, 0, height - 1);
          g_x += src[y * width + x] * kernel[k * ksize + l];
          g_y += src[y * width + x] * kernel[(ksize - 1 - l) * ksize + k];
        }
      }
      dst[i * width + j] = min(abs(g_x) + abs(g_y), 65535);
    } else {
      int g = 0;
      for (int k = 0; k < ksize; k++) {
        for (int l = 0; l < ksize; l++) {
          int x = j + l - ksize / 2;
          int y = i + k - ksize / 2;
          x = Clamp(x, 0, width - 1);
          y = Clamp(y, 0, height - 1);
          g += src[y * width + x] * kernel[k * ksize + l];
        }
      }
      dst[i * width + j] = Clamp(g, 0, 65535);
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
#pragma unroll
    for (int k = 0; k < 3; k++) {
#pragma unroll
      for (int l = 0; l < 3; l++) {
        int x = j + l - 1;
        int y = i + k - 1;
        x = Clamp(x, 0, width - 1);
        y = Clamp(y, 0, height - 1);
        g_x += src[y * width + x] * d_kernel_sobel[k * 3 + l];
        g_y += src[y * width + x] * d_kernel_sobel[(3 - 1 - l) * 3 + k];
      }
    }
    dst[i * width + j] = min(abs(g_x) + abs(g_y), 65535);
  }
}

__global__ void CudaSetSobelKernelShared(uint16_t* src, uint16_t* dst,
                                         size_t height, size_t width) {
  int i = blockIdx.y * blockDim.y + threadIdx.y;
  int j = blockIdx.x * blockDim.x + threadIdx.x;
  int local_x = threadIdx.x;
  int local_y = threadIdx.y;
  int li = Clamp(i, 0, height - 1);
  int lj = Clamp(j, 0, width - 1);
  __shared__ uint16_t s_tile[(kBlockSize + 2) * (kBlockSize + 2)];
  s_tile[(local_y + 1) * (kBlockSize + 2) + (local_x + 1)] =
      src[li * width + lj];
  if (local_y == 0) {
    s_tile[0 * (kBlockSize + 2) + (local_x + 1)] =
        src[max(li - 1, 0) * width + lj];
  }
  if (local_x == 0) {
    s_tile[(local_y + 1) * (kBlockSize + 2) + 0] =
        src[li * width + max(lj - 1, 0)];
  }
  if (local_y == kBlockSize - 1) {
    s_tile[(local_y + 2) * (kBlockSize + 2) + (local_x + 1)] =
        src[min(li + 1, (int)height - 1) * width + lj];
  }
  if (local_x == kBlockSize - 1) {
    s_tile[(local_y + 1) * (kBlockSize + 2) + (local_x + 2)] =
        src[li * width + min(lj + 1, (int)width - 1)];
  }
  if (local_y == 0 && local_x == 0) {
    s_tile[0 * (kBlockSize + 2) + 0] =
        src[max(li - 1, 0) * width + max(lj - 1, 0)];
  }
  if (local_y == 0 && local_x == kBlockSize - 1) {
    s_tile[0 * (kBlockSize + 2) + (local_x + 2)] =
        src[max(li - 1, 0) * width + min(lj + 1, (int)width - 1)];
  }
  if (local_y == kBlockSize - 1 && local_x == 0) {
    s_tile[(local_y + 2) * (kBlockSize + 2) + 0] =
        src[min(li + 1, (int)height - 1) * width + max(lj - 1, 0)];
  }
  if (local_y == kBlockSize - 1 && local_x == kBlockSize - 1) {
    s_tile[(local_y + 2) * (kBlockSize + 2) + (local_x + 2)] =
        src[min(li + 1, (int)height - 1) * width + min(lj + 1, (int)width - 1)];
  }
  // #pragma unroll
  //   for (int dy = -1; dy <= 1; dy++) {
  // #pragma unroll
  //     for (int dx = -1; dx <= 1; dx++) {
  //       int shared_y = local_y + 1 + dy;
  //       int shared_x = local_x + 1 + dx;
  //       int global_y = Clamp(i + dy, 0, (int)height - 1);
  //       int global_x = Clamp(j + dx, 0, (int)width - 1);
  //       s_tile[shared_y * (kBlockSize + 2) + shared_x] =
  //           src[global_y * width + global_x];
  //     }
  //   }
  __syncthreads();
  if (i < height && j < width) {
    int g_x = 0, g_y = 0;
#pragma unroll
    for (int k = 0; k < 3; k++) {
#pragma unroll
      for (int l = 0; l < 3; l++) {
        int val = s_tile[(local_y + k) * (kBlockSize + 2) + (local_x + l)];
        g_x += val * d_kernel_sobel[k * 3 + l];
        g_y += val * d_kernel_sobel[(2 - l) * 3 + k];
      }
    }
    dst[i * width + j] = min(abs(g_x) + abs(g_y), 65535);
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
__global__ void CudaSetSobelKernelSmooth(uint16_t* src, int* g_x, int* g_y,
                                         size_t height, size_t width) {
  int i = blockIdx.y * blockDim.y + threadIdx.y;
  int j = blockIdx.x * blockDim.x + threadIdx.x;
  if (i < height && j < width) {
    int sum_x = 0, sum_y = 0;
#pragma unroll
    for (int k = 0; k < 3; k++) {
      int x = j + k - 1;
      int y = i + k - 1;
      x = Clamp(x, 0, width - 1);
      y = Clamp(y, 0, height - 1);
      sum_x += src[i * width + x] * d_kernel_sobel_sep[k];
      sum_y += src[y * width + j] * d_kernel_sobel_sep[k];
    }
    g_x[i * width + j] = sum_x;
    g_y[i * width + j] = sum_y;
  }
}

__global__ void CudaSetSobelKernelSmoothShared(uint16_t* src, int* g_x,
                                               int* g_y, size_t height,
                                               size_t width) {
  int i = blockIdx.y * blockDim.y + threadIdx.y;
  int j = blockIdx.x * blockDim.x + threadIdx.x;
  int local_x = threadIdx.x;
  int local_y = threadIdx.y;
  int li = Clamp(i, 0, height - 1);
  int lj = Clamp(j, 0, width - 1);
  __shared__ uint16_t s_tile[(kBlockSize + 2) * (kBlockSize + 2)];
  s_tile[(local_y + 1) * (kBlockSize + 2) + (local_x + 1)] =
      src[li * width + lj];
  if (local_y == 0) {
    s_tile[0 * (kBlockSize + 2) + (local_x + 1)] =
        src[max(li - 1, 0) * width + lj];
  }
  if (local_x == 0) {
    s_tile[(local_y + 1) * (kBlockSize + 2) + 0] =
        src[li * width + max(lj - 1, 0)];
  }
  if (local_y == kBlockSize - 1) {
    s_tile[(local_y + 2) * (kBlockSize + 2) + (local_x + 1)] =
        src[min(li + 1, (int)height - 1) * width + lj];
  }
  if (local_x == kBlockSize - 1) {
    s_tile[(local_y + 1) * (kBlockSize + 2) + (local_x + 2)] =
        src[li * width + min(lj + 1, (int)width - 1)];
  }
  if (local_y == 0 && local_x == 0) {
    s_tile[0 * (kBlockSize + 2) + 0] =
        src[max(li - 1, 0) * width + max(lj - 1, 0)];
  }
  if (local_y == 0 && local_x == kBlockSize - 1) {
    s_tile[0 * (kBlockSize + 2) + (local_x + 2)] =
        src[max(li - 1, 0) * width + min(lj + 1, (int)width - 1)];
  }
  if (local_y == kBlockSize - 1 && local_x == 0) {
    s_tile[(local_y + 2) * (kBlockSize + 2) + 0] =
        src[min(li + 1, (int)height - 1) * width + max(lj - 1, 0)];
  }
  if (local_y == kBlockSize - 1 && local_x == kBlockSize - 1) {
    s_tile[(local_y + 2) * (kBlockSize + 2) + (local_x + 2)] =
        src[min(li + 1, (int)height - 1) * width + min(lj + 1, (int)width - 1)];
  }
  __syncthreads();
  if (i < height && j < width) {
    int sum_x = 0, sum_y = 0;
#pragma unroll
    for (int k = 0; k < 3; ++k) {
      sum_x += s_tile[(local_y + 1) * (kBlockSize + 2) + (local_x + k)] *
               d_kernel_sobel_sep[k];
      sum_y += s_tile[(local_y + k) * (kBlockSize + 2) + (local_x + 1)] *
               d_kernel_sobel_sep[k];
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
#pragma unroll
    for (int k = 0; k < 3; k++) {
#pragma unroll
      for (int l = 0; l < 3; l++) {
        int x = j + l - 1;
        int y = i + k - 1;
        x = Clamp(x, 0, width - 1);
        y = Clamp(y, 0, height - 1);
        g_x += src[y * width + x] * d_kernel_prewitt[k * 3 + l];
        g_y += src[y * width + x] * d_kernel_prewitt[(3 - 1 - l) * 3 + k];
      }
    }
    dst[i * width + j] = Clamp(abs(g_x) + abs(g_y), 0, 65535);
  }
}

__global__ void CudaSetPrewittKernelShared(uint16_t* src, uint16_t* dst,
                                           size_t height, size_t width) {
  int i = blockIdx.y * blockDim.y + threadIdx.y;
  int j = blockIdx.x * blockDim.x + threadIdx.x;
  int local_x = threadIdx.x;
  int local_y = threadIdx.y;
  int li = Clamp(i, 0, height - 1);
  int lj = Clamp(j, 0, width - 1);
  __shared__ uint16_t s_tile[(kBlockSize + 2) * (kBlockSize + 2)];
  s_tile[(local_y + 1) * (kBlockSize + 2) + (local_x + 1)] =
      src[li * width + lj];
  if (local_y == 0) {
    s_tile[0 * (kBlockSize + 2) + (local_x + 1)] =
        src[max(li - 1, 0) * width + lj];
  }
  if (local_x == 0) {
    s_tile[(local_y + 1) * (kBlockSize + 2) + 0] =
        src[li * width + max(lj - 1, 0)];
  }
  if (local_y == kBlockSize - 1) {
    s_tile[(local_y + 2) * (kBlockSize + 2) + (local_x + 1)] =
        src[min(li + 1, (int)height - 1) * width + lj];
  }
  if (local_x == kBlockSize - 1) {
    s_tile[(local_y + 1) * (kBlockSize + 2) + (local_x + 2)] =
        src[li * width + min(lj + 1, (int)width - 1)];
  }
  if (local_y == 0 && local_x == 0) {
    s_tile[0 * (kBlockSize + 2) + 0] =
        src[max(li - 1, 0) * width + max(lj - 1, 0)];
  }
  if (local_y == 0 && local_x == kBlockSize - 1) {
    s_tile[0 * (kBlockSize + 2) + (local_x + 2)] =
        src[max(li - 1, 0) * width + min(lj + 1, (int)width - 1)];
  }
  if (local_y == kBlockSize - 1 && local_x == 0) {
    s_tile[(local_y + 2) * (kBlockSize + 2) + 0] =
        src[min(li + 1, (int)height - 1) * width + max(lj - 1, 0)];
  }
  if (local_y == kBlockSize - 1 && local_x == kBlockSize - 1) {
    s_tile[(local_y + 2) * (kBlockSize + 2) + (local_x + 2)] =
        src[min(li + 1, (int)height - 1) * width + min(lj + 1, (int)width - 1)];
  }
  __syncthreads();
  if (i < height && j < width) {
    int g_x = 0, g_y = 0;
#pragma unroll
    for (int k = 0; k < 3; k++) {
#pragma unroll
      for (int l = 0; l < 3; l++) {
        int val = s_tile[(local_y + k) * (kBlockSize + 2) + (local_x + l)];
        g_x += val * d_kernel_prewitt[k * 3 + l];
        g_y += val * d_kernel_prewitt[(2 - l) * 3 + k];
      }
    }
    dst[i * width + j] = min(abs(g_x) + abs(g_y), 65535);
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
__global__ void CudaSetPrewittKernelAverage(uint16_t* src, int* g_x, int* g_y,
                                            size_t height, size_t width) {
  int i = blockIdx.y * blockDim.y + threadIdx.y;
  int j = blockIdx.x * blockDim.x + threadIdx.x;
  if (i < height && j < width) {
    int sum_x = 0, sum_y = 0;
#pragma unroll
    for (int k = 0; k < 3; k++) {
      int x = j + k - 1;
      int y = i + k - 1;
      x = Clamp(x, 0, width - 1);
      y = Clamp(y, 0, height - 1);
      sum_x += src[i * width + x] * d_kernel_prewitt_sep[k];
      sum_y += src[y * width + j] * d_kernel_prewitt_sep[k];
    }
    g_x[i * width + j] = sum_x;
    g_y[i * width + j] = sum_y;
  }
}

__global__ void CudaSetPrewittKernelAverageShared(uint16_t* src, int* g_x,
                                                  int* g_y, size_t height,
                                                  size_t width) {
  int i = blockIdx.y * blockDim.y + threadIdx.y;
  int j = blockIdx.x * blockDim.x + threadIdx.x;
  int local_x = threadIdx.x;
  int local_y = threadIdx.y;
  int li = Clamp(i, 0, height - 1);
  int lj = Clamp(j, 0, width - 1);
  __shared__ uint16_t s_tile[(kBlockSize + 2) * (kBlockSize + 2)];
  s_tile[(local_y + 1) * (kBlockSize + 2) + (local_x + 1)] =
      src[li * width + lj];
  if (local_y == 0) {
    s_tile[0 * (kBlockSize + 2) + (local_x + 1)] =
        src[max(li - 1, 0) * width + lj];
  }
  if (local_x == 0) {
    s_tile[(local_y + 1) * (kBlockSize + 2) + 0] =
        src[li * width + max(lj - 1, 0)];
  }
  if (local_y == kBlockSize - 1) {
    s_tile[(local_y + 2) * (kBlockSize + 2) + (local_x + 1)] =
        src[min(li + 1, (int)height - 1) * width + lj];
  }
  if (local_x == kBlockSize - 1) {
    s_tile[(local_y + 1) * (kBlockSize + 2) + (local_x + 2)] =
        src[li * width + min(lj + 1, (int)width - 1)];
  }
  if (local_y == 0 && local_x == 0) {
    s_tile[0 * (kBlockSize + 2) + 0] =
        src[max(li - 1, 0) * width + max(lj - 1, 0)];
  }
  if (local_y == 0 && local_x == kBlockSize - 1) {
    s_tile[0 * (kBlockSize + 2) + (local_x + 2)] =
        src[max(li - 1, 0) * width + min(lj + 1, (int)width - 1)];
  }
  if (local_y == kBlockSize - 1 && local_x == 0) {
    s_tile[(local_y + 2) * (kBlockSize + 2) + 0] =
        src[min(li + 1, (int)height - 1) * width + max(lj - 1, 0)];
  }
  if (local_y == kBlockSize - 1 && local_x == kBlockSize - 1) {
    s_tile[(local_y + 2) * (kBlockSize + 2) + (local_x + 2)] =
        src[min(li + 1, (int)height - 1) * width + min(lj + 1, (int)width - 1)];
  }
  __syncthreads();
  if (i < height && j < width) {
    int sum_x = 0, sum_y = 0;
#pragma unroll
    for (int k = 0; k < 3; ++k) {
      sum_x += s_tile[(local_y + 1) * (kBlockSize + 2) + (local_x + k)] *
               d_kernel_prewitt_sep[k];
      sum_y += s_tile[(local_y + k) * (kBlockSize + 2) + (local_x + 1)] *
               d_kernel_prewitt_sep[k];
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
__global__ void CudaSepKernelDiff(int* g_x, int* g_y, int* result_x,
                                  int* result_y, size_t height, size_t width) {
  int i = blockIdx.y * blockDim.y + threadIdx.y;
  int j = blockIdx.x * blockDim.x + threadIdx.x;
  if (i < height && j < width) {
    int sum_x = 0;
    int sum_y = 0;
#pragma unroll
    for (int k = 0; k < 3; k++) {
      int x = j + k - 1;
      int y = i + k - 1;
      x = Clamp(x, 0, width - 1);
      y = Clamp(y, 0, height - 1);
      sum_y += g_y[i * width + x] * d_kernel_gradient[k];
      sum_x += g_x[y * width + j] * d_kernel_gradient[k];
    }
    result_x[i * width + j] = sum_x;
    result_y[i * width + j] = sum_y;
  }
}

__global__ void CudaSepKernelDiff(int* g_x, int* g_y, uint16_t* dst,
                                  size_t height, size_t width) {
  int i = blockIdx.y * blockDim.y + threadIdx.y;
  int j = blockIdx.x * blockDim.x + threadIdx.x;
  if (i < height && j < width) {
    int sum_x = 0;
    int sum_y = 0;
#pragma unroll
    for (int k = 0; k < 3; k++) {
      int x = j + k - 1;
      int y = i + k - 1;
      x = Clamp(x, 0, width - 1);
      y = Clamp(y, 0, height - 1);
      sum_y += g_y[i * width + x] * d_kernel_gradient[k];
      sum_x += g_x[y * width + j] * d_kernel_gradient[k];
    }
    int sum = abs(sum_x) + abs(sum_y);
    dst[i * width + j] = static_cast<uint16_t>(Clamp(sum, 0, 65535));
  }
}

__global__ void CudaSepKernelDiffShared(int* g_x, int* g_y, int* result_x,
                                        int* result_y, size_t height,
                                        size_t width) {
  int i = blockIdx.y * blockDim.y + threadIdx.y;
  int j = blockIdx.x * blockDim.x + threadIdx.x;
  int local_x = threadIdx.x;
  int local_y = threadIdx.y;
  int li = Clamp(i, 0, height - 1);
  int lj = Clamp(j, 0, width - 1);
  __shared__ int s_tile_gx[kBlockSize + 2][kBlockSize];
  __shared__ int s_tile_gy[kBlockSize][kBlockSize + 2];
  s_tile_gx[local_y + 1][local_x] = g_x[li * width + lj];
  s_tile_gy[local_y][local_x + 1] = g_y[li * width + lj];
  if (local_y == 0) {
    s_tile_gx[0][local_x] = g_x[max(li - 1, 0) * width + lj];
  }
  if (local_x == 0) {
    s_tile_gy[local_y][0] = g_y[li * width + max(lj - 1, 0)];
  }
  if (local_y == kBlockSize - 1) {
    s_tile_gx[local_y + 2][local_x] =
        g_x[min(li + 1, (int)height - 1) * width + lj];
  }
  if (local_x == kBlockSize - 1) {
    s_tile_gy[local_y][local_x + 2] =
        g_y[li * width + min(lj + 1, (int)width - 1)];
  }
  __syncthreads();
  if (i < height && j < width) {
    int sum_x = 0, sum_y = 0;
#pragma unroll
    for (int k = 0; k < 3; ++k) {
      sum_x += s_tile_gx[local_y + k][local_x] * d_kernel_gradient[k];
      sum_y += s_tile_gy[local_y][local_x + k] * d_kernel_gradient[k];
    }
    result_x[i * width + j] = sum_x;
    result_y[i * width + j] = sum_y;
  }
}

__global__ void CudaSepKernelDiffShared(int* g_x, int* g_y, uint16_t* dst,
                                        size_t height, size_t width) {
  int i = blockIdx.y * blockDim.y + threadIdx.y;
  int j = blockIdx.x * blockDim.x + threadIdx.x;
  int local_x = threadIdx.x;
  int local_y = threadIdx.y;
  int li = Clamp(i, 0, height - 1);
  int lj = Clamp(j, 0, width - 1);
  __shared__ int s_tile_gx[kBlockSize + 2][kBlockSize];
  __shared__ int s_tile_gy[kBlockSize][kBlockSize + 2];
  s_tile_gx[local_y + 1][local_x] = g_x[li * width + lj];
  s_tile_gy[local_y][local_x + 1] = g_y[li * width + lj];
  if (local_y == 0) {
    s_tile_gx[0][local_x] = g_x[max(li - 1, 0) * width + lj];
  }
  if (local_x == 0) {
    s_tile_gy[local_y][0] = g_y[li * width + max(lj - 1, 0)];
  }
  if (local_y == kBlockSize - 1) {
    s_tile_gx[local_y + 2][local_x] =
        g_x[min(li + 1, (int)height - 1) * width + lj];
  }
  if (local_x == kBlockSize - 1) {
    s_tile_gy[local_y][local_x + 2] =
        g_y[li * width + min(lj + 1, (int)width - 1)];
  }
  __syncthreads();
  if (i < height && j < width) {
    int sum_x = 0, sum_y = 0;
#pragma unroll
    for (int k = 0; k < 3; ++k) {
      sum_x += s_tile_gx[local_y + k][local_x] * d_kernel_gradient[k];
      sum_y += s_tile_gy[local_y][local_x + k] * d_kernel_gradient[k];
    }
    int sum = abs(sum_x) + abs(sum_y);
    dst[i * width + j] = static_cast<uint16_t>(Clamp(sum, 0, 65535));
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
__global__ void CudaAddAbsMtx(int* mtx1, int* mtx2, uint16_t* result,
                              size_t height, size_t width) {
  int i = blockIdx.y * blockDim.y + threadIdx.y;
  int j = blockIdx.x * blockDim.x + threadIdx.x;
  if (i < height && j < width) {
    int sum = abs(mtx1[i * width + j]) + abs(mtx2[i * width + j]);
    result[i * width + j] = static_cast<uint16_t>(Clamp(sum, 0, 65535));
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
                                 size_t width, float* kernel, size_t ksize) {
  int i = blockIdx.y * blockDim.y + threadIdx.y;
  int j = blockIdx.x * blockDim.x + threadIdx.x;
  if (i < height && j < width) {
    float sum = 0;
    for (int k = 0; k < ksize; k++) {
      for (int l = 0; l < ksize; l++) {
        int x = j + l - ksize / 2;
        int y = i + k - ksize / 2;
        x = Clamp(x, 0, width - 1);
        y = Clamp(y, 0, height - 1);
        sum += src[y * width + x] * kernel[k * ksize + l];
      }
    }
    dst[i * width + j] = static_cast<uint16_t>(Clamp(round(sum), 0, 65535));
  }
}

/**
 * @brief Применяет горизонтальное размытие по Гауссу к входному изображению.
 *
 * @param src Указатель на исходное изображение на устройстве.
 * @param dst Указатель на промежуточное изображение (результат
 * горизонтального размытия).
 * @param height Высота изображения.
 * @param width Ширина изображения.
 * @param kernel Указатель на горизонтальное ядро Гаусса на устройстве.
 * @param ksize Размер ядра.
 */
__global__ void CudaGaussianBlurSepHorizontal(uint16_t* src, float* dst,
                                              size_t height, size_t width,
                                              float* kernel, size_t ksize) {
  int i = blockIdx.y * blockDim.y + threadIdx.y;
  int j = blockIdx.x * blockDim.x + threadIdx.x;
  if (i < height && j < width) {
    float sum = 0;
    for (int k = 0; k < ksize; k++) {
      int x = j + k - ksize / 2;
      x = Clamp(x, 0, width - 1);
      sum += src[i * width + x] * kernel[k];
    }
    dst[i * width + j] = sum;
  }
}

/**
 * @brief Применяет вертикальное размытие по Гауссу к промежуточному
 * изображению.
 *
 * @param src Указатель на промежуточное изображение (результат
 * горизонтального размытия).
 * @param dst Указатель на результирующее изображение на устройстве.
 * @param height Высота изображения.
 * @param width Ширина изображения.
 * @param kernel Указатель на вертикальное ядро Гаусса на устройстве.
 * @param ksize Размер ядра.
 */
__global__ void CudaGaussianBlurSepVertical(float* src, uint16_t* dst,
                                            size_t height, size_t width,
                                            float* kernel, size_t ksize) {
  int i = blockIdx.y * blockDim.y + threadIdx.y;
  int j = blockIdx.x * blockDim.x + threadIdx.x;
  if (i < height && j < width) {
    float sum = 0;
    for (int k = 0; k < ksize; k++) {
      int y = i + k - ksize / 2;
      y = Clamp(y, 0, height - 1);
      sum += src[y * width + j] * kernel[k];
    }
    dst[i * width + j] = static_cast<uint16_t>(Clamp(round(sum), 0, 65535));
  }
}

TIFFImage TIFFImage::SetKernelCuda(const Kernel<int>& kernel,
                                   const bool shared_memory,
                                   const bool rotate) const {
  uint16_t* h_src = image_;
  uint16_t* d_src;
  uint16_t* h_dst = new uint16_t[width_ * height_];
  uint16_t* d_dst;
  size_t image_size = width_ * height_ * sizeof(uint16_t);
  if (!cuda_mem_manager_.IsAllocated()) {
    size_t free_memory, total_memory;
    checkCudaErrors(cudaMemGetInfo(&free_memory, &total_memory));
    if (free_memory < image_size * 2) {
      throw std::runtime_error("Изображение слишком большое для GPU");
    }
    checkCudaErrors(cudaMalloc(&d_src, image_size));
    checkCudaErrors(
        cudaMemcpy(d_src, h_src, image_size, cudaMemcpyHostToDevice));
    checkCudaErrors(cudaMalloc(&d_dst, image_size));
  } else {
    d_src = cuda_mem_manager_.GetDeviceSrc();
    d_dst = cuda_mem_manager_.GetDeviceDst();
  }
  dim3 threads = shared_memory ? dim3(kBlockSize, kBlockSize) : dim3(1024);
  dim3 blocks = shared_memory ? dim3((width_ + kBlockSize - 1) / kBlockSize,
                                     (height_ + kBlockSize - 1) / kBlockSize)
                              : dim3((width_ + 1023) / 1024, height_);
  if (kernel == kKernelSobel) {
    if (shared_memory) {
      CudaSetSobelKernelShared<<<blocks, threads>>>(d_src, d_dst, height_,
                                                    width_);
    } else {
      CudaSetSobelKernel<<<blocks, threads>>>(d_src, d_dst, height_, width_);
    }
  } else if (kernel == kKernelPrewitt) {
    if (shared_memory) {
      CudaSetPrewittKernelShared<<<blocks, threads>>>(d_src, d_dst, height_,
                                                      width_);
    } else {
      CudaSetPrewittKernel<<<blocks, threads>>>(d_src, d_dst, height_, width_);
    }
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
  if (!cuda_mem_manager_.IsAllocated()) {
    checkCudaErrors(cudaFree(d_src));
    checkCudaErrors(cudaFree(d_dst));
  }
  TIFFImage result(*this);
  std::memcpy(result.image_, h_dst, image_size);
  delete[] h_dst;
  return result;
}

void TIFFImage::SetKernelCudaAsync(const Kernel<int>& kernel,
                                   const bool shared_memory,
                                   const bool rotate) const {
  uint16_t* d_src;
  uint16_t* d_dst;
  if (!cuda_mem_manager_.IsAllocated()) {
    throw std::runtime_error(
        "Для асинхронной версии требуется "
        "предварительное выделение памяти на GPU");
  } else {
    d_src = cuda_mem_manager_.GetDeviceSrc();
    d_dst = cuda_mem_manager_.GetDeviceDst();
  }
  dim3 threads = shared_memory ? dim3(kBlockSize, kBlockSize) : dim3(1024);
  dim3 blocks = shared_memory ? dim3((width_ + kBlockSize - 1) / kBlockSize,
                                     (height_ + kBlockSize - 1) / kBlockSize)
                              : dim3((width_ + 1023) / 1024, height_);
  if (kernel == kKernelSobel) {
    if (shared_memory) {
      CudaSetSobelKernelShared<<<blocks, threads, 0, cuda_stream_>>>(
          d_src, d_dst, height_, width_);
    } else {
      CudaSetSobelKernel<<<blocks, threads, 0, cuda_stream_>>>(d_src, d_dst,
                                                               height_, width_);
    }
  } else if (kernel == kKernelPrewitt) {
    if (shared_memory) {
      CudaSetPrewittKernelShared<<<blocks, threads, 0, cuda_stream_>>>(
          d_src, d_dst, height_, width_);
    } else {
      CudaSetPrewittKernel<<<blocks, threads, 0, cuda_stream_>>>(
          d_src, d_dst, height_, width_);
    }
  } else {
    throw std::runtime_error(
        "Для асинхронной версии поддерживаются только ядра Собеля и Превитта");
  }
  checkCudaErrors(cudaGetLastError());
}

TIFFImage TIFFImage::SetKernelSobelSepCuda(const bool shared_memory) const {
  uint16_t* h_src = image_;
  uint16_t* d_src;
  int* d_g_x;
  int* d_g_y;
  int* d_result_x;
  int* d_result_y;
  uint16_t* h_dst = new uint16_t[width_ * height_];
  uint16_t* d_dst;
  size_t image_size = width_ * height_ * sizeof(uint16_t);
  size_t temps_size = image_size * 2;
  if (!cuda_mem_manager_.IsAllocated()) {
    size_t free_memory, total_memory;
    checkCudaErrors(cudaMemGetInfo(&free_memory, &total_memory));
    if (free_memory < image_size * 2 + temps_size * 4) {
      throw std::runtime_error("Изображение слишком большое для GPU");
    }
    checkCudaErrors(cudaMalloc(&d_src, image_size));
    checkCudaErrors(
        cudaMemcpy(d_src, h_src, image_size, cudaMemcpyHostToDevice));
    checkCudaErrors(cudaMalloc(&d_g_x, temps_size));
    checkCudaErrors(cudaMalloc(&d_g_y, temps_size));
    checkCudaErrors(cudaMalloc(&d_result_x, temps_size));
    checkCudaErrors(cudaMalloc(&d_result_y, temps_size));
    checkCudaErrors(cudaMalloc(&d_dst, image_size));
  } else {
    d_src = cuda_mem_manager_.GetDeviceSrc();
    d_g_x = cuda_mem_manager_.GetDeviceSepGx();
    d_g_y = cuda_mem_manager_.GetDeviceSepGy();
    d_result_x = cuda_mem_manager_.GetDeviceSepResultX();
    d_result_y = cuda_mem_manager_.GetDeviceSepResultY();
    d_dst = cuda_mem_manager_.GetDeviceDst();
  }
  dim3 threads = shared_memory ? dim3(kBlockSize, kBlockSize) : dim3(1024);
  dim3 blocks = shared_memory ? dim3((width_ + kBlockSize - 1) / kBlockSize,
                                     (height_ + kBlockSize - 1) / kBlockSize)
                              : dim3((width_ + 1023) / 1024, height_);
  if (shared_memory) {
    CudaSetSobelKernelSmoothShared<<<blocks, threads>>>(d_src, d_g_x, d_g_y,
                                                        height_, width_);
    checkCudaErrors(cudaDeviceSynchronize());
    CudaSepKernelDiffShared<<<blocks, threads>>>(d_g_x, d_g_y, d_dst, height_,
                                                 width_);
    // CudaSepKernelDiffShared<<<blocks, threads>>>(d_g_x, d_g_y, d_result_x,
    //                                              d_result_y, height_,
    //                                              width_);
    // checkCudaErrors(cudaDeviceSynchronize());
    // CudaAddAbsMtx<<<blocks, threads>>>(d_result_x, d_result_y, d_dst,
    // height_,
    //                                    width_);
    checkCudaErrors(cudaDeviceSynchronize());
  } else {
    CudaSetSobelKernelSmooth<<<blocks, threads>>>(d_src, d_g_x, d_g_y, height_,
                                                  width_);
    checkCudaErrors(cudaDeviceSynchronize());
    CudaSepKernelDiff<<<blocks, threads>>>(d_g_x, d_g_y, d_dst, height_,
                                           width_);
    // CudaSepKernelDiff<<<blocks, threads>>>(d_g_x, d_g_y, d_result_x,
    // d_result_y,
    //                                        height_, width_);
    // checkCudaErrors(cudaDeviceSynchronize());
    // CudaAddAbsMtx<<<blocks, threads>>>(d_result_x, d_result_y, d_dst,
    // height_,
    //                                    width_);
    checkCudaErrors(cudaDeviceSynchronize());
  }
  checkCudaErrors(cudaMemcpy(h_dst, d_dst, image_size, cudaMemcpyDeviceToHost));
  if (!cuda_mem_manager_.IsAllocated()) {
    checkCudaErrors(cudaFree(d_src));
    checkCudaErrors(cudaFree(d_g_x));
    checkCudaErrors(cudaFree(d_g_y));
    checkCudaErrors(cudaFree(d_result_x));
    checkCudaErrors(cudaFree(d_result_y));
    checkCudaErrors(cudaFree(d_dst));
  }
  TIFFImage result(*this);
  std::memcpy(result.image_, h_dst, image_size);
  delete[] h_dst;
  return result;
}

void TIFFImage::SetKernelSobelSepCudaAsync(const bool shared_memory) const {
  uint16_t* d_src;
  int* d_g_x;
  int* d_g_y;
  uint16_t* d_dst;
  if (!cuda_mem_manager_.IsAllocated()) {
    throw std::runtime_error(
        "Для асинхронной версии требуется "
        "предварительное выделение памяти на GPU");
  } else {
    d_src = cuda_mem_manager_.GetDeviceSrc();
    d_g_x = cuda_mem_manager_.GetDeviceSepGx();
    d_g_y = cuda_mem_manager_.GetDeviceSepGy();
    d_dst = cuda_mem_manager_.GetDeviceDst();
  }
  dim3 threads = shared_memory ? dim3(kBlockSize, kBlockSize) : dim3(1024);
  dim3 blocks = shared_memory ? dim3((width_ + kBlockSize - 1) / kBlockSize,
                                     (height_ + kBlockSize - 1) / kBlockSize)
                              : dim3((width_ + 1023) / 1024, height_);
  if (shared_memory) {
    CudaSetSobelKernelSmoothShared<<<blocks, threads, 0, cuda_stream_>>>(
        d_src, d_g_x, d_g_y, height_, width_);
    CudaSepKernelDiffShared<<<blocks, threads, 0, cuda_stream_>>>(
        d_g_x, d_g_y, d_dst, height_, width_);
    // CudaSepKernelDiffShared<<<blocks, threads>>>(d_g_x, d_g_y, d_result_x,
    //                                              d_result_y, height_,
    //                                              width_);
    // checkCudaErrors(cudaDeviceSynchronize());
    // CudaAddAbsMtx<<<blocks, threads>>>(d_result_x, d_result_y, d_dst,
    // height_,
    //                                    width_);
  } else {
    CudaSetSobelKernelSmooth<<<blocks, threads, 0, cuda_stream_>>>(
        d_src, d_g_x, d_g_y, height_, width_);
    CudaSepKernelDiff<<<blocks, threads, 0, cuda_stream_>>>(d_g_x, d_g_y, d_dst,
                                                            height_, width_);
    // CudaSepKernelDiff<<<blocks, threads>>>(d_g_x, d_g_y, d_result_x,
    // d_result_y,
    //                                        height_, width_);
    // checkCudaErrors(cudaDeviceSynchronize());
    // CudaAddAbsMtx<<<blocks, threads>>>(d_result_x, d_result_y, d_dst,
    // height_,
    //                                    width_);
  }
  checkCudaErrors(cudaGetLastError());
}

TIFFImage TIFFImage::SetKernelPrewittSepCuda(const bool shared_memory) const {
  uint16_t* h_src = image_;
  uint16_t* d_src;
  int* d_g_x;
  int* d_g_y;
  int* d_result_x;
  int* d_result_y;
  uint16_t* h_dst = new uint16_t[width_ * height_];
  uint16_t* d_dst;
  size_t image_size = width_ * height_ * sizeof(uint16_t);
  size_t temps_size = image_size * 2;
  if (!cuda_mem_manager_.IsAllocated()) {
    size_t free_memory, total_memory;
    checkCudaErrors(cudaMemGetInfo(&free_memory, &total_memory));
    if (free_memory < image_size * 2 + temps_size * 4) {
      throw std::runtime_error("Изображение слишком большое для GPU");
    }
    checkCudaErrors(cudaMalloc(&d_src, image_size));
    checkCudaErrors(
        cudaMemcpy(d_src, h_src, image_size, cudaMemcpyHostToDevice));
    checkCudaErrors(cudaMalloc(&d_g_x, temps_size));
    checkCudaErrors(cudaMalloc(&d_g_y, temps_size));
    checkCudaErrors(cudaMalloc(&d_result_x, temps_size));
    checkCudaErrors(cudaMalloc(&d_result_y, temps_size));
    checkCudaErrors(cudaMalloc(&d_dst, image_size));
  } else {
    d_src = cuda_mem_manager_.GetDeviceSrc();
    d_g_x = cuda_mem_manager_.GetDeviceSepGx();
    d_g_y = cuda_mem_manager_.GetDeviceSepGy();
    d_result_x = cuda_mem_manager_.GetDeviceSepResultX();
    d_result_y = cuda_mem_manager_.GetDeviceSepResultY();
    d_dst = cuda_mem_manager_.GetDeviceDst();
  }
  dim3 threads = shared_memory ? dim3(kBlockSize, kBlockSize) : dim3(1024);
  dim3 blocks = shared_memory ? dim3((width_ + kBlockSize - 1) / kBlockSize,
                                     (height_ + kBlockSize - 1) / kBlockSize)
                              : dim3((width_ + 1023) / 1024, height_);
  if (shared_memory) {
    CudaSetPrewittKernelAverageShared<<<blocks, threads>>>(d_src, d_g_x, d_g_y,
                                                           height_, width_);
    checkCudaErrors(cudaDeviceSynchronize());
    CudaSepKernelDiffShared<<<blocks, threads>>>(d_g_x, d_g_y, d_dst, height_,
                                                 width_);
    // CudaSepKernelDiffShared<<<blocks, threads>>>(d_g_x, d_g_y, d_result_x,
    //                                              d_result_y, height_,
    //                                              width_);
    // checkCudaErrors(cudaDeviceSynchronize());
    // CudaAddAbsMtx<<<blocks, threads>>>(d_result_x, d_result_y, d_dst,
    // height_,
    //                                    width_);
    checkCudaErrors(cudaDeviceSynchronize());
  } else {
    CudaSetPrewittKernelAverage<<<blocks, threads>>>(d_src, d_g_x, d_g_y,
                                                     height_, width_);
    checkCudaErrors(cudaDeviceSynchronize());
    CudaSepKernelDiff<<<blocks, threads>>>(d_g_x, d_g_y, d_dst, height_,
                                           width_);
    // CudaSepKernelDiff<<<blocks, threads>>>(d_g_x, d_g_y, d_result_x,
    // d_result_y,
    //                                        height_, width_);
    // checkCudaErrors(cudaDeviceSynchronize());
    // CudaAddAbsMtx<<<blocks, threads>>>(d_result_x, d_result_y, d_dst,
    // height_,
    //                                    width_);
    checkCudaErrors(cudaDeviceSynchronize());
  }
  checkCudaErrors(cudaMemcpy(h_dst, d_dst, image_size, cudaMemcpyDeviceToHost));
  if (!cuda_mem_manager_.IsAllocated()) {
    checkCudaErrors(cudaFree(d_src));
    checkCudaErrors(cudaFree(d_g_x));
    checkCudaErrors(cudaFree(d_g_y));
    checkCudaErrors(cudaFree(d_result_x));
    checkCudaErrors(cudaFree(d_result_y));
    checkCudaErrors(cudaFree(d_dst));
  }
  TIFFImage result(*this);
  std::memcpy(result.image_, h_dst, image_size);
  delete[] h_dst;
  return result;
}

void TIFFImage::SetKernelPrewittSepCudaAsync(const bool shared_memory) const {
  uint16_t* d_src;
  int* d_g_x;
  int* d_g_y;
  uint16_t* d_dst;
  if (!cuda_mem_manager_.IsAllocated()) {
    throw std::runtime_error(
        "Для асинхронной версии требуется "
        "предварительное выделение памяти на GPU");
  } else {
    d_src = cuda_mem_manager_.GetDeviceSrc();
    d_g_x = cuda_mem_manager_.GetDeviceSepGx();
    d_g_y = cuda_mem_manager_.GetDeviceSepGy();
    d_dst = cuda_mem_manager_.GetDeviceDst();
  }
  dim3 threads = shared_memory ? dim3(kBlockSize, kBlockSize) : dim3(1024);
  dim3 blocks = shared_memory ? dim3((width_ + kBlockSize - 1) / kBlockSize,
                                     (height_ + kBlockSize - 1) / kBlockSize)
                              : dim3((width_ + 1023) / 1024, height_);
  if (shared_memory) {
    CudaSetPrewittKernelAverageShared<<<blocks, threads, 0, cuda_stream_>>>(
        d_src, d_g_x, d_g_y, height_, width_);
    CudaSepKernelDiffShared<<<blocks, threads, 0, cuda_stream_>>>(
        d_g_x, d_g_y, d_dst, height_, width_);
    // CudaSepKernelDiffShared<<<blocks, threads>>>(d_g_x, d_g_y, d_result_x,
    //                                              d_result_y, height_,
    //                                              width_);
    // checkCudaErrors(cudaDeviceSynchronize());
    // CudaAddAbsMtx<<<blocks, threads>>>(d_result_x, d_result_y, d_dst,
    // height_,
    //                                    width_);
  } else {
    CudaSetPrewittKernelAverage<<<blocks, threads, 0, cuda_stream_>>>(
        d_src, d_g_x, d_g_y, height_, width_);
    CudaSepKernelDiff<<<blocks, threads, 0, cuda_stream_>>>(d_g_x, d_g_y, d_dst,
                                                            height_, width_);
    // CudaSepKernelDiff<<<blocks, threads>>>(d_g_x, d_g_y, d_result_x,
    // d_result_y,
    //                                        height_, width_);
    // checkCudaErrors(cudaDeviceSynchronize());
    // CudaAddAbsMtx<<<blocks, threads>>>(d_result_x, d_result_y, d_dst,
    // height_,
    //                                    width_);
  }
  checkCudaErrors(cudaGetLastError());
}

TIFFImage TIFFImage::GaussianBlurCuda(const size_t size, const float sigma) {
  uint16_t* h_src = image_;
  uint16_t* d_src;
  uint16_t* h_dst = new uint16_t[width_ * height_];
  uint16_t* d_dst;
  float* d_kernel;
  size_t image_size = width_ * height_ * sizeof(uint16_t);
  if (!cuda_mem_manager_.IsAllocated()) {
    size_t free_memory, total_memory;
    checkCudaErrors(cudaMemGetInfo(&free_memory, &total_memory));
    if (free_memory < image_size * 2) {
      throw std::runtime_error("Изображение слишком большое для GPU");
    }
    checkCudaErrors(cudaMalloc(&d_src, image_size));
    checkCudaErrors(
        cudaMemcpy(d_src, h_src, image_size, cudaMemcpyHostToDevice));
    checkCudaErrors(cudaMalloc(&d_dst, image_size));
    Kernel<float> kernel = Kernel<float>::GetGaussianKernel(size, sigma);
    float* h_kernel;
    size_t kernel_size = kernel.GetHeight() * kernel.GetWidth() * sizeof(float);
    checkCudaErrors(cudaMemGetInfo(&free_memory, &total_memory));
    if (free_memory < kernel_size) {
      throw std::runtime_error("Ядро слишком большое для GPU");
    }
    kernel.CopyKernelTo(&h_kernel);
    checkCudaErrors(cudaMalloc(&d_kernel, kernel_size));
    checkCudaErrors(
        cudaMemcpy(d_kernel, h_kernel, kernel_size, cudaMemcpyHostToDevice));
    delete[] h_kernel;
  } else {
    d_src = cuda_mem_manager_.GetDeviceSrc();
    d_dst = cuda_mem_manager_.GetDeviceDst();
    cuda_mem_manager_.CheckGaussianKernel(size, sigma);
    d_kernel = cuda_mem_manager_.GetDeviceGaussianKernel();
  }
  // dim3 threads(32, 32);  // 2D-блоки по 32x32
  // dim3 blocks((width_ + 31) / 32, (height_ + 31) / 32);
  dim3 threads(1024);
  dim3 blocks((width_ + 1023) / 1024, height_);
  CudaGaussianBlur<<<blocks, threads>>>(d_src, d_dst, height_, width_, d_kernel,
                                        size);
  checkCudaErrors(cudaDeviceSynchronize());
  checkCudaErrors(cudaMemcpy(h_dst, d_dst, image_size, cudaMemcpyDeviceToHost));
  if (!cuda_mem_manager_.IsAllocated()) {
    checkCudaErrors(cudaFree(d_src));
    checkCudaErrors(cudaFree(d_dst));
    checkCudaErrors(cudaFree(d_kernel));
  }
  TIFFImage result(*this);
  std::memcpy(result.image_, h_dst, image_size);
  delete[] h_dst;
  return result;
}

TIFFImage TIFFImage::GaussianBlurSepCuda(const size_t size, const float sigma) {
  uint16_t* h_src = image_;
  uint16_t* d_src;
  float* d_temp;
  uint16_t* h_dst = new uint16_t[width_ * height_];
  uint16_t* d_dst;
  float* d_kernel;
  size_t image_size = width_ * height_ * sizeof(uint16_t);
  size_t temp_size = width_ * height_ * sizeof(float);
  if (!cuda_mem_manager_.IsAllocated()) {
    size_t free_memory, total_memory;
    checkCudaErrors(cudaMemGetInfo(&free_memory, &total_memory));
    if (free_memory < image_size * 2 + temp_size) {
      throw std::runtime_error("Изображение слишком большое для GPU");
    }
    checkCudaErrors(cudaMalloc(&d_src, image_size));
    checkCudaErrors(
        cudaMemcpy(d_src, h_src, image_size, cudaMemcpyHostToDevice));
    checkCudaErrors(cudaMalloc(&d_temp, temp_size));
    checkCudaErrors(cudaMalloc(&d_dst, image_size));
    Kernel<float> kernel = Kernel<float>::GetGaussianKernelSep(size, sigma);
    float* h_kernel;
    size_t kernel_size = kernel.GetHeight() * kernel.GetWidth() * sizeof(float);
    checkCudaErrors(cudaMemGetInfo(&free_memory, &total_memory));
    if (free_memory < kernel_size) {
      throw std::runtime_error("Ядро слишком большое для GPU");
    }
    kernel.CopyKernelTo(&h_kernel);
    checkCudaErrors(cudaMalloc(&d_kernel, kernel_size));
    checkCudaErrors(
        cudaMemcpy(d_kernel, h_kernel, kernel_size, cudaMemcpyHostToDevice));
    delete[] h_kernel;
  } else {
    d_src = cuda_mem_manager_.GetDeviceSrc();
    d_temp = cuda_mem_manager_.GetDeviceGaussianSepTemp();
    d_dst = cuda_mem_manager_.GetDeviceDst();
    cuda_mem_manager_.CheckGaussianKernel(size, sigma);
    d_kernel = cuda_mem_manager_.GetDeviceGaussianKernel();
  }
  // dim3 threads(32, 32);  // 2D-блоки по 32x32
  // dim3 blocks((width_ + 31) / 32, (height_ + 31) / 32);
  dim3 threads(1024);
  dim3 blocks((width_ + 1023) / 1024, height_);
  CudaGaussianBlurSepHorizontal<<<blocks, threads>>>(d_src, d_temp, height_,
                                                     width_, d_kernel, size);
  checkCudaErrors(cudaDeviceSynchronize());
  CudaGaussianBlurSepVertical<<<blocks, threads>>>(d_temp, d_dst, height_,
                                                   width_, d_kernel, size);
  checkCudaErrors(cudaDeviceSynchronize());
  checkCudaErrors(cudaMemcpy(h_dst, d_dst, image_size, cudaMemcpyDeviceToHost));
  if (!cuda_mem_manager_.IsAllocated()) {
    checkCudaErrors(cudaFree(d_src));
    checkCudaErrors(cudaFree(d_temp));
    checkCudaErrors(cudaFree(d_dst));
    checkCudaErrors(cudaFree(d_kernel));
  }
  TIFFImage result(*this);
  std::memcpy(result.image_, h_dst, image_size);
  delete[] h_dst;
  return result;
}

void TIFFImage::SetCudaStream(cudaStream_t stream) noexcept {
  cuda_stream_ = stream;
}

void TIFFImage::CreateCudaStream() {
  if (cuda_stream_ == nullptr) {
    checkCudaErrors(cudaStreamCreate(&cuda_stream_));
  }
}

void TIFFImage::DestroyCudaStream() {
  if (cuda_stream_ != nullptr) {
    checkCudaErrors(cudaStreamDestroy(cuda_stream_));
    cuda_stream_ = nullptr;
  }
}

void TIFFImage::CopyImageToDeviceAsync() {
  if (!cuda_mem_manager_.IsAllocated()) {
    throw std::runtime_error(
        "Device memory is not allocated. Call AllocateDeviceMemory() before "
        "using async API.");
  }
  cudaStream_t s = (cuda_stream_ == nullptr) ? nullptr : cuda_stream_;
  cuda_mem_manager_.CopyImageToDeviceAsync(image_, s);
}

void TIFFImage::CopyImageToDeviceAsyncSlot(int slot) {
  if (!cuda_mem_manager_.IsAllocated()) {
    throw std::runtime_error(
        "Device memory is not allocated. Call AllocateDeviceMemory() before "
        "using async API.");
  }
  cudaStream_t s = (cuda_stream_ == nullptr) ? nullptr : cuda_stream_;
  cuda_mem_manager_.CopyImageToDeviceAsyncSlot(image_, s, slot);
}

TIFFImage TIFFImage::CopyImageFromDeviceAsync() {
  if (!cuda_mem_manager_.IsAllocated()) {
    throw std::runtime_error(
        "Для асинхронного копирования требуется предварительное выделение "
        "device памяти");
  }
  cudaStream_t s = (cuda_stream_ == nullptr) ? nullptr : cuda_stream_;
  uint16_t* h_dst = new uint16_t[width_ * height_];
  size_t image_size = width_ * height_ * sizeof(uint16_t);
  cuda_mem_manager_.CopyImageFromDeviceAsync(h_dst, s);
  checkCudaErrors(cudaStreamSynchronize(s));
  TIFFImage result(*this);
  std::memcpy(result.image_, h_dst, image_size);
  delete[] h_dst;
  return result;
}

TIFFImage TIFFImage::CopyImageFromDeviceAsyncSlot(int slot) {
  if (!cuda_mem_manager_.IsAllocated()) {
    throw std::runtime_error(
        "Для асинхронного копирования требуется предварительное выделение "
        "device памяти");
  }
  cudaStream_t s = (cuda_stream_ == nullptr) ? nullptr : cuda_stream_;
  uint16_t* h_dst = new uint16_t[width_ * height_];
  size_t image_size = width_ * height_ * sizeof(uint16_t);
  cuda_mem_manager_.CopyImageFromDeviceAsyncSlot(h_dst, s, slot);
  checkCudaErrors(cudaStreamSynchronize(s));
  TIFFImage result(*this);
  std::memcpy(result.image_, h_dst, image_size);
  delete[] h_dst;
  return result;
}

// Slot-based kernel launches
void TIFFImage::SetKernelCudaAsyncSlot(const Kernel<int>& kernel,
                                       const bool shared_memory,
                                       const bool rotate, int slot) const {
  uint16_t* d_src;
  uint16_t* d_dst;
  if (!cuda_mem_manager_.IsAllocated()) {
    throw std::runtime_error(
        "Для асинхронной версии требуется предварительное выделение памяти на "
        "GPU");
  } else {
    d_src = cuda_mem_manager_.GetDeviceSrcSlot(slot);
    d_dst = cuda_mem_manager_.GetDeviceDstSlot(slot);
  }
  dim3 threads = shared_memory ? dim3(kBlockSize, kBlockSize) : dim3(1024);
  dim3 blocks = shared_memory ? dim3((width_ + kBlockSize - 1) / kBlockSize,
                                     (height_ + kBlockSize - 1) / kBlockSize)
                              : dim3((width_ + 1023) / 1024, height_);
  if (kernel == kKernelSobel) {
    if (shared_memory) {
      CudaSetSobelKernelShared<<<blocks, threads, 0, cuda_stream_>>>(
          d_src, d_dst, height_, width_);
    } else {
      CudaSetSobelKernel<<<blocks, threads, 0, cuda_stream_>>>(d_src, d_dst,
                                                               height_, width_);
    }
  } else if (kernel == kKernelPrewitt) {
    if (shared_memory) {
      CudaSetPrewittKernelShared<<<blocks, threads, 0, cuda_stream_>>>(
          d_src, d_dst, height_, width_);
    } else {
      CudaSetPrewittKernel<<<blocks, threads, 0, cuda_stream_>>>(
          d_src, d_dst, height_, width_);
    }
  } else {
    throw std::runtime_error(
        "Для асинхронной версии поддерживаются только ядра Собеля и Превитта");
  }
  checkCudaErrors(cudaGetLastError());
}

void TIFFImage::SetKernelSobelSepCudaAsyncSlot(const bool shared_memory,
                                               int slot) const {
  uint16_t* d_src = cuda_mem_manager_.GetDeviceSrcSlot(slot);
  int* d_g_x = cuda_mem_manager_.GetDeviceSepGx();
  int* d_g_y = cuda_mem_manager_.GetDeviceSepGy();
  uint16_t* d_dst = cuda_mem_manager_.GetDeviceDstSlot(slot);
  dim3 threads = shared_memory ? dim3(kBlockSize, kBlockSize) : dim3(1024);
  dim3 blocks = shared_memory ? dim3((width_ + kBlockSize - 1) / kBlockSize,
                                     (height_ + kBlockSize - 1) / kBlockSize)
                              : dim3((width_ + 1023) / 1024, height_);
  if (shared_memory) {
    CudaSetSobelKernelSmoothShared<<<blocks, threads, 0, cuda_stream_>>>(
        d_src, d_g_x, d_g_y, height_, width_);
    CudaSepKernelDiffShared<<<blocks, threads, 0, cuda_stream_>>>(
        d_g_x, d_g_y, d_dst, height_, width_);
  } else {
    CudaSetSobelKernelSmooth<<<blocks, threads, 0, cuda_stream_>>>(
        d_src, d_g_x, d_g_y, height_, width_);
    CudaSepKernelDiff<<<blocks, threads, 0, cuda_stream_>>>(d_g_x, d_g_y, d_dst,
                                                            height_, width_);
  }
  checkCudaErrors(cudaGetLastError());
}

void TIFFImage::SetKernelPrewittSepCudaAsyncSlot(const bool shared_memory,
                                                 int slot) const {
  uint16_t* d_src = cuda_mem_manager_.GetDeviceSrcSlot(slot);
  int* d_g_x = cuda_mem_manager_.GetDeviceSepGx();
  int* d_g_y = cuda_mem_manager_.GetDeviceSepGy();
  uint16_t* d_dst = cuda_mem_manager_.GetDeviceDstSlot(slot);
  dim3 threads = shared_memory ? dim3(kBlockSize, kBlockSize) : dim3(1024);
  dim3 blocks = shared_memory ? dim3((width_ + kBlockSize - 1) / kBlockSize,
                                     (height_ + kBlockSize - 1) / kBlockSize)
                              : dim3((width_ + 1023) / 1024, height_);
  if (shared_memory) {
    CudaSetPrewittKernelAverageShared<<<blocks, threads, 0, cuda_stream_>>>(
        d_src, d_g_x, d_g_y, height_, width_);
    CudaSepKernelDiffShared<<<blocks, threads, 0, cuda_stream_>>>(
        d_g_x, d_g_y, d_dst, height_, width_);
  } else {
    CudaSetPrewittKernelAverage<<<blocks, threads, 0, cuda_stream_>>>(
        d_src, d_g_x, d_g_y, height_, width_);
    CudaSepKernelDiff<<<blocks, threads, 0, cuda_stream_>>>(d_g_x, d_g_y, d_dst,
                                                            height_, width_);
  }
  checkCudaErrors(cudaGetLastError());
}