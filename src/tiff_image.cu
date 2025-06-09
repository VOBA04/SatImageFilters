#include <cmath>
#include <cstddef>
#include <cstdint>
#include <stdexcept>
#include "image_operation.h"
#include "kernel.h"
#include "tiff_image.h"
#include <vector_types.h>
#include <cuda_runtime.h>
#include <cuda_runtime_api.h>
#include <driver_types.h>
#include "check_cuda_errors.h"

__constant__ int d_kernel_sobel[9] = {-1, 0, 1, -2, 0, 2, -1, 0, 1};
__constant__ int d_kernel_prewitt[9] = {-1, 0, 1, -1, 0, 1, -1, 0, 1};
__constant__ int d_kernel_sobel_sep[3] = {1, 2, 1};
__constant__ int d_kernel_prewitt_sep[3] = {1, 1, 1};
__constant__ int d_kernel_gradient[3] = {-1, 0, 1};

bool CheckFreeMem(size_t required_memory) {
  size_t free_memory, total_memory;
  checkCudaErrors(cudaMemGetInfo(&free_memory, &total_memory));
  return free_memory > required_memory;
}

__device__ int Clamp(const int val, const int min, const int max) {
  if (val < min) {
    return min;
  }
  if (val > max) {
    return max;
  }
  return val;
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
      dst[i * width + j] = Clamp(abs(g_x) + abs(g_y), 0, 65535);
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
    for (int k = 0; k < 3; k++) {
      for (int l = 0; l < 3; l++) {
        int x = j + l - 1;
        int y = i + k - 1;
        x = Clamp(x, 0, width - 1);
        y = Clamp(y, 0, height - 1);
        g_x += src[y * width + x] * d_kernel_sobel[k * 3 + l];
        g_y += src[y * width + x] * d_kernel_sobel[(3 - 1 - l) * 3 + k];
      }
    }
    dst[i * width + j] = Clamp(abs(g_x) + abs(g_y), 0, 65535);
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
        x = Clamp(x, 0, width - 1);
        y = Clamp(y, 0, height - 1);
        g_x += src[y * width + x] * d_kernel_prewitt[k * 3 + l];
        g_y += src[y * width + x] * d_kernel_prewitt[(3 - 1 - l) * 3 + k];
      }
    }
    dst[i * width + j] = Clamp(abs(g_x) + abs(g_y), 0, 65535);
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
 * @param dst Указатель на промежуточное изображение (результат горизонтального
 * размытия).
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
 * @param src Указатель на промежуточное изображение (результат горизонтального
 * размытия).
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

void TIFFImage::ImageToDeviceMemory(ImageOperation operation,
                                    size_t gaussian_kernel_size,
                                    float gaussian_sigma) {
  uint8_t operation_num = static_cast<uint8_t>(operation);
  bool sobel =
      (static_cast<uint8_t>(ImageOperation::Sobel) & operation_num) != 0;
  bool prewitt =
      (static_cast<uint8_t>(ImageOperation::Prewitt) & operation_num) != 0;
  bool gaussian_blur =
      (static_cast<uint8_t>(ImageOperation::GaussianBlur) & operation_num) != 0;
  bool gaussian_blur_sep =
      (static_cast<uint8_t>(ImageOperation::GaussianBlurSep) & operation_num) !=
      0;
  bool separated =
      (static_cast<uint8_t>(ImageOperation::Separated) & operation_num) != 0;
  if (operation == ImageOperation::None) {
    throw std::runtime_error("Операция не задана");
  }
  if (gaussian_blur && gaussian_blur_sep) {
    throw std::runtime_error(
        "Нельзя одновременно использовать размытие по "
        "Гауссу и раздельное размытие по Гауссу");
  }
  if ((gaussian_blur || gaussian_blur_sep) &&
      ((gaussian_kernel_size % 2) == 0u)) {
    throw std::runtime_error("Ядро фильтр ядра Гаусса должен быть нечетным");
  }
  size_t image_size = height_ * width_ * sizeof(uint16_t);
  if (sobel || prewitt) {
    if (separated) {
      if (!CheckFreeMem(image_size * 2 + image_size * 2 * 4)) {
        throw std::runtime_error("Изображение слишком большое");
      }
    } else if (!CheckFreeMem(image_size * 2)) {
      throw std::runtime_error("Изображение слишком большое");
    }
  }
  if (gaussian_blur) {
    if (!CheckFreeMem(image_size + gaussian_kernel_size * gaussian_kernel_size *
                                       sizeof(float))) {
      throw std::runtime_error(
          "Изображение или ядро фильтра Гаусса слишком большие");
    }
  } else if (gaussian_blur_sep) {
    if (!CheckFreeMem(image_size * 2 + height_ * width_ * sizeof(float) +
                      gaussian_kernel_size * sizeof(float))) {
      throw std::runtime_error(
          "Изображение или ядро разделенного фильтра Гаусса слишком большие");
    }
  }
  checkCudaErrors(cudaMalloc(&d_src_, image_size));
  checkCudaErrors(
      cudaMemcpy(d_src_, image_, image_size, cudaMemcpyHostToDevice));
  checkCudaErrors(cudaMalloc(&d_dst_, image_size));
  if ((sobel || prewitt) && separated) {
    checkCudaErrors(cudaMalloc(&d_sep_g_x_, image_size * 2));
    checkCudaErrors(cudaMalloc(&d_sep_g_y_, image_size * 2));
    checkCudaErrors(cudaMalloc(&d_sep_result_x_, image_size * 2));
    checkCudaErrors(cudaMalloc(&d_sep_result_y_, image_size * 2));
  }
  if (gaussian_blur) {
    gaussian_kernel_size_ = gaussian_kernel_size;
    gaussian_sigma_ = gaussian_sigma;
    checkCudaErrors(cudaMalloc(
        &d_gaussian_kernel_,
        gaussian_kernel_size * gaussian_kernel_size * sizeof(float)));
    Kernel<float> kernel =
        Kernel<float>::GetGaussianKernel(gaussian_kernel_size, gaussian_sigma);
    float* h_kernel;
    kernel.CopyKernelTo(&h_kernel);
    checkCudaErrors(
        cudaMemcpy(d_gaussian_kernel_, h_kernel,
                   gaussian_kernel_size * gaussian_kernel_size * sizeof(float),
                   cudaMemcpyHostToDevice));
    delete[] h_kernel;
  } else if (gaussian_blur_sep) {
    gaussian_kernel_size_ = gaussian_kernel_size;
    gaussian_sigma_ = gaussian_sigma;
    checkCudaErrors(
        cudaMalloc(&d_gaussian_sep_temp_, height_ * width_ * sizeof(float)));
    checkCudaErrors(
        cudaMalloc(&d_gaussian_kernel_, gaussian_kernel_size * sizeof(float)));
    Kernel<float> kernel = Kernel<float>::GetGaussianKernelSep(
        gaussian_kernel_size, gaussian_sigma);
    float* h_kernel;
    kernel.CopyKernelTo(&h_kernel);
    checkCudaErrors(cudaMemcpy(d_gaussian_kernel_, h_kernel,
                               gaussian_kernel_size * sizeof(float),
                               cudaMemcpyHostToDevice));
    delete[] h_kernel;
  }
  d_mem_allocaded_ = true;
}

void TIFFImage::FreeDeviceMemory() {
  if (!d_mem_allocaded_) {
    return;
  }
  if (d_src_ != nullptr) {
    checkCudaErrors(cudaFree(d_src_));
    d_src_ = nullptr;
  }
  if (d_dst_ != nullptr) {
    checkCudaErrors(cudaFree(d_dst_));
    d_dst_ = nullptr;
  }
  if (d_gaussian_sep_temp_ != nullptr) {
    checkCudaErrors(cudaFree(d_gaussian_sep_temp_));
    d_gaussian_sep_temp_ = nullptr;
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
  d_mem_allocaded_ = false;
}

void TIFFImage::CopyImageToDevice() {
  if (!d_mem_allocaded_) {
    throw std::runtime_error("Память на устройстве не выделена");
  }
  size_t image_size = height_ * width_ * sizeof(uint16_t);
  checkCudaErrors(
      cudaMemcpy(d_src_, image_, image_size, cudaMemcpyHostToDevice));
}

TIFFImage TIFFImage::SetKernelCuda(const Kernel<int>& kernel,
                                   const bool rotate) const {
  uint16_t* h_src = image_;
  uint16_t* d_src;
  uint16_t* h_dst = new uint16_t[width_ * height_];
  uint16_t* d_dst;
  size_t image_size = width_ * height_ * sizeof(uint16_t);
  if (!d_mem_allocaded_) {
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
    d_src = d_src_;
    d_dst = d_dst_;
  }
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
  if (!d_mem_allocaded_) {
    checkCudaErrors(cudaFree(d_src));
    checkCudaErrors(cudaFree(d_dst));
  }
  TIFFImage result(*this);
  std::memcpy(result.image_, h_dst, image_size);
  delete[] h_dst;
  return result;
}

TIFFImage TIFFImage::SetKernelSobelSepCuda() const {
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
  if (!d_mem_allocaded_) {
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
    d_src = d_src_;
    d_g_x = d_sep_g_x_;
    d_g_y = d_sep_g_y_;
    d_result_x = d_sep_result_x_;
    d_result_y = d_sep_result_y_;
    d_dst = d_dst_;
  }
  // dim3 threads(32, 32);  // 2D-блоки по 32x32
  // dim3 blocks((width_ + 31) / 32, (height_ + 31) / 32);
  dim3 threads(1024);
  dim3 blocks((width_ + 1023) / 1024, height_);
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
  if (!d_mem_allocaded_) {
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

TIFFImage TIFFImage::SetKernelPrewittSepCuda() const {
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
  if (!d_mem_allocaded_) {
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
    d_src = d_src_;
    d_g_x = d_sep_g_x_;
    d_g_y = d_sep_g_y_;
    d_result_x = d_sep_result_x_;
    d_result_y = d_sep_result_y_;
    d_dst = d_dst_;
  }
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
  if (!d_mem_allocaded_) {
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

TIFFImage TIFFImage::GaussianBlurCuda(const size_t size, const float sigma) {
  uint16_t* h_src = image_;
  uint16_t* d_src;
  uint16_t* h_dst = new uint16_t[width_ * height_];
  uint16_t* d_dst;
  float* d_kernel;
  size_t image_size = width_ * height_ * sizeof(uint16_t);
  if (!d_mem_allocaded_) {
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
    d_src = d_src_;
    d_dst = d_dst_;
    if (size != gaussian_kernel_size_ || sigma != gaussian_sigma_) {
      gaussian_kernel_size_ = size;
      gaussian_sigma_ = sigma;
      Kernel<float> kernel = Kernel<float>::GetGaussianKernel(size, sigma);
      float* h_kernel;
      size_t kernel_size =
          kernel.GetHeight() * kernel.GetWidth() * sizeof(float);
      kernel.CopyKernelTo(&h_kernel);
      checkCudaErrors(cudaFree(d_gaussian_kernel_));
      checkCudaErrors(cudaMalloc(&d_gaussian_kernel_, kernel_size));
      checkCudaErrors(cudaMemcpy(d_gaussian_kernel_, h_kernel, kernel_size,
                                 cudaMemcpyHostToDevice));
      d_kernel = d_gaussian_kernel_;
      delete[] h_kernel;
    } else {
      d_kernel = d_gaussian_kernel_;
    }
  }
  // dim3 threads(32, 32);  // 2D-блоки по 32x32
  // dim3 blocks((width_ + 31) / 32, (height_ + 31) / 32);
  dim3 threads(1024);
  dim3 blocks((width_ + 1023) / 1024, height_);
  CudaGaussianBlur<<<blocks, threads>>>(d_src, d_dst, height_, width_, d_kernel,
                                        size);
  checkCudaErrors(cudaDeviceSynchronize());
  checkCudaErrors(cudaMemcpy(h_dst, d_dst, image_size, cudaMemcpyDeviceToHost));
  if (!d_mem_allocaded_) {
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
  if (!d_mem_allocaded_) {
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
    d_src = d_src_;
    d_temp = d_gaussian_sep_temp_;
    d_dst = d_dst_;
    if (size != gaussian_kernel_size_ || sigma != gaussian_sigma_) {
      gaussian_kernel_size_ = size;
      gaussian_sigma_ = sigma;
      Kernel<float> kernel = Kernel<float>::GetGaussianKernelSep(size, sigma);
      float* h_kernel;
      size_t kernel_size =
          kernel.GetHeight() * kernel.GetWidth() * sizeof(float);
      kernel.CopyKernelTo(&h_kernel);
      checkCudaErrors(cudaFree(d_gaussian_kernel_));
      checkCudaErrors(cudaMalloc(&d_gaussian_kernel_, kernel_size));
      checkCudaErrors(cudaMemcpy(d_gaussian_kernel_, h_kernel, kernel_size,
                                 cudaMemcpyHostToDevice));
      d_kernel = d_gaussian_kernel_;
      delete[] h_kernel;
    } else {
      d_kernel = d_gaussian_kernel_;
    }
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
  if (!d_mem_allocaded_) {
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