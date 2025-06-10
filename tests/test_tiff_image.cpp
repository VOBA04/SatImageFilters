#include <gtest/gtest.h>
#include <opencv2/core/hal/interface.h>
#include <tiffio.h>
#include "image_operation.h"
#include "kernel.h"
#include "tiff_image.h"
#include <cstddef>
#include <cstdint>
#include <cstdlib>
#include <opencv2/core.hpp>
#include <opencv2/core/mat.hpp>
#include <opencv2/opencv.hpp>
#include <string>
#include <filesystem>
#include <cuda_runtime.h>

const std::string kTestImagePath =
    std::string(PROJECT_SOURCE_DIR) + "/tests/test_image.tiff";

namespace fs = std::filesystem;

const size_t kTestImagesCount = 9;

void CreateTestImage(int width, int height, uint8_t image_type = 0) {
  cv::Mat img(height, width, CV_16U, cv::Scalar(0));
  switch (image_type) {
    case 0:
    default:
      // Левая половина чёрная, правая половина белая (вертикальное разделение)
      for (int y = 0; y < height; ++y) {
        for (int x = width / 2; x < width; ++x) {
          img.at<uint16_t>(y, x) = 65535;
        }
      }
      break;
    case 1:
      // Верхняя половина чёрная, нижняя половина белая (горизонтальное
      // разделение)
      for (int y = height / 2; y < height; ++y) {
        for (int x = 0; x < width; ++x) {
          img.at<uint16_t>(y, x) = 65535;
        }
      }
      break;
    case 2:
      // Горизонтальный градиент от чёрного к белому (слева направо)
      for (int y = 0; y < height; ++y) {
        for (int x = 0; x < width; ++x) {
          img.at<uint16_t>(y, x) =
              static_cast<uint16_t>((x / (float)width) * 65535);
        }
      }
      break;
    case 3:
      // Вертикальный градиент от чёрного к белому (сверху вниз)
      for (int y = 0; y < height; ++y) {
        for (int x = 0; x < width; ++x) {
          img.at<uint16_t>(y, x) =
              static_cast<uint16_t>((y / (float)height) * 65535);
        }
      }
      break;
    case 4:
      // Случайное изображение (равномерное распределение)
      cv::theRNG().state = 12345;
      cv::randu(img, 0, 65535);
      break;
    case 5:
      // Случайное изображение (нормальное распределение, среднее 32768,
      // σ=10000)
      cv::randn(img, 32768, 10000);
      break;
    case 6:
      // Белый круг в центре на чёрном фоне
      cv::circle(img, cv::Point(width / 2, height / 2),
                 std::min(width, height) / 4, cv::Scalar(65535), -1);
      break;
    case 7:
      // Один белый пиксель в центре, остальное чёрное
      img.at<uint16_t>(height / 2, width / 2) = 65535;
      break;
    case 8:
      // Шахматный узор (чёрно-белый)
      for (int y = 0; y < height; ++y) {
        for (int x = 0; x < width; ++x) {
          img.at<uint16_t>(y, x) = (((x / 10 + y / 10) % 2) != 0) ? 65535 : 0;
        }
      }
      break;
  }
  cv::imwrite(kTestImagePath, img);
}

inline bool IsCudaAvailable(std::string* error_message = nullptr) {
  int device_count = 0;
  cudaError_t error = cudaGetDeviceCount(&device_count);
  if (error != cudaSuccess) {
    if (error_message != nullptr) {
      *error_message = "CUDA error: " + std::string(cudaGetErrorString(error));
    }
    return false;
  }
  if (device_count == 0) {
    if (error_message != nullptr) {
      *error_message = "No CUDA-capable devices found.";
    }
    return false;
  }
  return true;
}

TEST(TIFFImageTest, LoadAndSave) {
  CreateTestImage(100, 100);
  TIFFImage img(kTestImagePath);
  EXPECT_EQ(img.GetWidth(), 100);
  EXPECT_EQ(img.GetHeight(), 100);
  EXPECT_NO_THROW(img.Save(std::string(PROJECT_SOURCE_DIR) +
                           "/tests/test_image_copy.tiff"));
  EXPECT_THROW(
      img.Save(std::string(PROJECT_SOURCE_DIR) + "tests/test_image_copy.tiff"),
      std::runtime_error);
  TIFFImage img_copy(std::string(PROJECT_SOURCE_DIR) +
                     "/tests/test_image_copy.tiff");
  EXPECT_TRUE(img == img_copy);
  fs::remove(kTestImagePath);
  fs::remove(std::string(PROJECT_SOURCE_DIR) + "/tests/test_image_copy.tiff");
}

TEST(TIFFImageTest, CopyConstructor) {
  CreateTestImage(100, 100);
  TIFFImage img(kTestImagePath);
  TIFFImage img_copy(img);
  EXPECT_TRUE(img == img_copy);
  fs::remove(kTestImagePath);
}

TEST(TIFFImageTest, CopyAssignment) {
  CreateTestImage(100, 100);
  TIFFImage img(kTestImagePath);
  TIFFImage img_copy;
  img_copy = img;
  EXPECT_TRUE(img == img_copy);
  img = img_copy;
  EXPECT_TRUE(img == img_copy);
  EXPECT_NO_FATAL_FAILURE(img_copy.CopyFields(img));
  fs::remove(kTestImagePath);
}

TEST(TIFFImageTest, NotEqual) {
  CreateTestImage(100, 100);
  TIFFImage img(kTestImagePath);
  TIFFImage img2(200, 200);
  TIFFImage img3(img);
  img3.Set(0, 0, 1);
  EXPECT_FALSE(img == img2);
  EXPECT_FALSE(img == img3);
  fs::remove(kTestImagePath);
}

TEST(TIFFImageTest, GetSetPixel) {
  TIFFImage void_img;
  EXPECT_THROW(void_img.Get(0, 0), std::runtime_error);
  EXPECT_THROW(void_img.Set(0, 0, 65535), std::runtime_error);
  TIFFImage img(100, 100);
  img.Set(50, 50, 65535);
  img.Set(0, 0, 65535);
  img.Set(99, 99, 65535);
  EXPECT_EQ(img.Get(50, 50), 65535);
  EXPECT_THROW(img.Set(100, 50, 65535), std::runtime_error);
  EXPECT_EQ(img.Get(-1, -1), 65535);
  EXPECT_EQ(img.Get(100, 100), 65535);
}

TEST(TIFFImageTest, Clrear) {
  TIFFImage img(100, 100);
  img.Set(50, 50, 65535);
  img.Clear();
  EXPECT_THROW(img.Get(50, 50), std::runtime_error);
  EXPECT_NO_FATAL_FAILURE(img.Clear());
}

TEST(TIFFImageTest, GaussianBlurCPU) {
  for (size_t k = 0; k < kTestImagesCount; k++) {
    CreateTestImage(100, 100, k);
    TIFFImage img(kTestImagePath);
    TIFFImage blurred = img.GaussianBlur(3, 1.0);
    cv::Mat cv_img = cv::imread(kTestImagePath, cv::IMREAD_UNCHANGED);
    cv::Mat blurred_cv;
    cv::GaussianBlur(cv_img, blurred_cv, cv::Size(3, 3), 1.0, 0,
                     cv::BORDER_REPLICATE);
    bool failed = false;
    for (size_t i = 0; i < img.GetHeight() && !failed; i++) {
      for (size_t j = 0; j < img.GetWidth(); j++) {
        EXPECT_NEAR(blurred.Get(j, i), blurred_cv.at<uint16_t>(i, j), 1)
            << "Mismatch at pixel (" << j << ", " << i << ")" << " image " << k;
        if (HasFailure()) {
          failed = true;
          break;
        }
      }
    }
    fs::remove(kTestImagePath);
  }
}

TEST(TIFFImageTest, GaussianBlurGPU) {
  std::string cuda_error;
  if (!IsCudaAvailable(&cuda_error)) {
    GTEST_SKIP() << "Skipping CUDA tests: " << cuda_error;
  }
  for (size_t k = 0; k < kTestImagesCount; k++) {
    CreateTestImage(100, 100, k);
    TIFFImage img(kTestImagePath);
    TIFFImage blurred_cpu = img.GaussianBlur(3, 1.0);
    TIFFImage blurred_cpu_2 = img.GaussianBlur(3, 2.0);
    TIFFImage blurred_cuda = img.GaussianBlurCuda(3, 1.0);
    TIFFImage blurred_cuda_sep = img.GaussianBlurSepCuda(3, 1.0);
    TIFFImage blurred_cuda_2 = img.GaussianBlurCuda(3, 2.0);
    img.SetImagePatametersForDevice(ImageOperation::GaussianBlur, 3, 1.0);
    img.AllocateDeviceMemory();
    img.CopyImageToDevice();
    TIFFImage blurred_cuda_mem = img.GaussianBlurCuda(3, 1.0);
    img.SetImagePatametersForDevice(ImageOperation::GaussianBlurSep, 3, 1.0);
    img.CopyImageToDevice();
    TIFFImage blurred_cuda_sep_mem = img.GaussianBlurSepCuda(3, 1.0);
    TIFFImage blurred_cuda_sep_2 = img.GaussianBlurSepCuda(3, 2.0);
    bool failed = false;
    for (size_t i = 0; i < img.GetHeight() && !failed; i++) {
      for (size_t j = 0; j < img.GetWidth(); j++) {
        EXPECT_NEAR(blurred_cpu.Get(j, i), blurred_cuda.Get(j, i), 1)
            << "Mismatch at pixel (" << j << ", " << i << ")"
            << " image " << k;
        if (HasFailure()) {
          failed = true;
          break;
        }
        EXPECT_NEAR(blurred_cpu_2.Get(j, i), blurred_cuda_2.Get(j, i), 1)
            << "Mismatch at pixel (" << j << ", " << i << ")"
            << " image " << k;
        if (HasFailure()) {
          failed = true;
          break;
        }
        EXPECT_NEAR(blurred_cpu.Get(j, i), blurred_cuda_mem.Get(j, i), 1)
            << "Mismatch at pixel (" << j << ", " << i << ")"
            << " image " << k;
        if (HasFailure()) {
          failed = true;
          break;
        }
        EXPECT_NEAR(blurred_cpu.Get(j, i), blurred_cuda_sep_mem.Get(j, i), 1)
            << "Mismatch at pixel (" << j << ", " << i << ")"
            << " image " << k;
        if (HasFailure()) {
          failed = true;
          break;
        }
        EXPECT_NEAR(blurred_cpu_2.Get(j, i), blurred_cuda_sep_2.Get(j, i), 1)
            << "Mismatch at pixel (" << j << ", " << i << ")"
            << " image " << k;
        if (HasFailure()) {
          failed = true;
          break;
        }
      }
    }
    img.FreeDeviceMemory();
    fs::remove(kTestImagePath);
  }
}

TEST(TIFFImageTest, SobelFilterCPU) {
  for (size_t k = 0; k < kTestImagesCount; k++) {
    CreateTestImage(100, 100, k);
    TIFFImage img(kTestImagePath);
    TIFFImage sobel_x = img.SetKernel(kKernelSobel, false);
    TIFFImage sobel_y = img.SetKernel(
        kKernelSobel.Rotate(KernelRotationDegrees::DEGREES_90), false);
    TIFFImage sobel = img.SetKernel(kKernelSobel);
    cv::Mat cv_img = cv::imread(kTestImagePath, cv::IMREAD_UNCHANGED);
    cv::Mat sobel_x_cv, sobel_y_cv, sobel_cv;
    float sobel_kernel_x[3 * 3] = {-1, 0, 1, -2, 0, 2, -1, 0, 1};
    float sobel_kernel_y[3 * 3] = {-1, -2, -1, 0, 0, 0, 1, 2, 1};
    cv::Mat kernel_x = cv::Mat(3, 3, CV_32F, sobel_kernel_x);
    cv::Mat kernel_y = cv::Mat(3, 3, CV_32F, sobel_kernel_y);
    cv::filter2D(cv_img, sobel_x_cv, CV_32F, kernel_x, cv::Point(-1, -1), 0,
                 cv::BORDER_REPLICATE);
    cv::filter2D(cv_img, sobel_y_cv, CV_32F, kernel_y, cv::Point(-1, -1), 0,
                 cv::BORDER_REPLICATE);
    cv::Mat abs_sobel_x, abs_sobel_y;
    abs_sobel_x = cv::abs(sobel_x_cv);
    abs_sobel_y = cv::abs(sobel_y_cv);
    abs_sobel_x.convertTo(sobel_x_cv, CV_16U);
    abs_sobel_y.convertTo(sobel_y_cv, CV_16U);
    cv::addWeighted(sobel_x_cv, 1, sobel_y_cv, 1, 0, sobel_cv);
    bool failed = false;
    for (size_t i = 0; i < img.GetHeight() && !failed; i++) {
      for (size_t j = 0; j < img.GetWidth(); j++) {
        EXPECT_EQ(sobel_x.Get(j, i), sobel_x_cv.at<uint16_t>(i, j))
            << "Mismatch at pixel (" << j << ", " << i << ")"
            << " image " << k;
        if (HasFailure()) {
          failed = true;
          break;
        }
        EXPECT_EQ(sobel_y.Get(j, i), sobel_y_cv.at<uint16_t>(i, j))
            << "Mismatch at pixel (" << j << ", " << i << ")"
            << " image " << k;
        if (HasFailure()) {
          failed = true;
          break;
        }
        EXPECT_EQ(sobel.Get(j, i), sobel_cv.at<uint16_t>(i, j))
            << "Mismatch at pixel (" << j << ", " << i << ")"
            << " image " << k;
        if (HasFailure()) {
          failed = true;
          break;
        }
      }
    }
    TIFFImage sobel_sep = img.SetKernelSobelSep();
    EXPECT_TRUE(sobel == sobel_sep);
    fs::remove(kTestImagePath);
  }
}

TEST(TIFFImageTest, SobelFilterGPU) {
  std::string cuda_error;
  if (!IsCudaAvailable(&cuda_error)) {
    GTEST_SKIP() << "Skipping CUDA tests: " << cuda_error;
  }
  for (size_t k = 0; k < kTestImagesCount; k++) {
    CreateTestImage(100, 100, k);
    TIFFImage img(kTestImagePath);
    TIFFImage sobel = img.SetKernel(kKernelSobel);
    TIFFImage sobel_cuda = img.SetKernelCuda(kKernelSobel);
    EXPECT_TRUE(sobel == sobel_cuda);
    TIFFImage sobel_cuda_sep = img.SetKernelSobelSepCuda();
    EXPECT_TRUE(sobel == sobel_cuda_sep);
    img.SetImagePatametersForDevice(ImageOperation::Sobel);
    img.AllocateDeviceMemory();
    img.CopyImageToDevice();
    sobel_cuda = img.SetKernelCuda(kKernelSobel);
    EXPECT_TRUE(sobel == sobel_cuda);
    img.SetImagePatametersForDevice(ImageOperation::Sobel |
                                    ImageOperation::Separated);
    img.CopyImageToDevice();
    sobel_cuda_sep = img.SetKernelSobelSepCuda();
    EXPECT_TRUE(sobel == sobel_cuda_sep);
    img.FreeDeviceMemory();
    fs::remove(kTestImagePath);
  }
}

TEST(TIFFImageTest, PrewittFilterCPU) {
  for (size_t k = 0; k < kTestImagesCount; k++) {
    CreateTestImage(100, 100, k);
    TIFFImage img(kTestImagePath);
    TIFFImage prewitt_x = img.SetKernel(kKernelPrewitt, false);
    TIFFImage prewitt_y = img.SetKernel(
        kKernelPrewitt.Rotate(KernelRotationDegrees::DEGREES_90), false);
    TIFFImage prewitt = img.SetKernel(kKernelPrewitt);
    cv::Mat cv_img = cv::imread(kTestImagePath, cv::IMREAD_UNCHANGED);
    cv::Mat prewitt_x_cv, prewitt_y_cv, prewitt_cv;
    float prewitt_kernel_x[3 * 3] = {-1, 0, 1, -1, 0, 1, -1, 0, 1};
    float prewitt_kernel_y[3 * 3] = {-1, -1, -1, 0, 0, 0, 1, 1, 1};
    cv::Mat kernel_x = cv::Mat(3, 3, CV_32F, prewitt_kernel_x);
    cv::Mat kernel_y = cv::Mat(3, 3, CV_32F, prewitt_kernel_y);
    cv::filter2D(cv_img, prewitt_x_cv, CV_32F, kernel_x, cv::Point(-1, -1), 0,
                 cv::BORDER_REPLICATE);
    cv::filter2D(cv_img, prewitt_y_cv, CV_32F, kernel_y, cv::Point(-1, -1), 0,
                 cv::BORDER_REPLICATE);
    cv::Mat abs_prewitt_x_cv, abs_prewitt_y_cv;
    abs_prewitt_x_cv = cv::abs(prewitt_x_cv);
    abs_prewitt_y_cv = cv::abs(prewitt_y_cv);
    abs_prewitt_x_cv.convertTo(prewitt_x_cv, CV_16U);
    abs_prewitt_y_cv.convertTo(prewitt_y_cv, CV_16U);
    cv::addWeighted(prewitt_x_cv, 1, prewitt_y_cv, 1, 0, prewitt_cv);
    bool failed = false;
    for (size_t i = 0; i < img.GetHeight() && !failed; i++) {
      for (size_t j = 0; j < img.GetWidth(); j++) {
        EXPECT_EQ(prewitt_x.Get(j, i), prewitt_x_cv.at<uint16_t>(i, j));
        if (HasFailure()) {
          failed = true;
          break;
        }
        EXPECT_EQ(prewitt_y.Get(j, i), prewitt_y_cv.at<uint16_t>(i, j));
        if (HasFailure()) {
          failed = true;
          break;
        }
        EXPECT_EQ(prewitt.Get(j, i), prewitt_cv.at<uint16_t>(i, j));
        if (HasFailure()) {
          failed = true;
          break;
        }
      }
    }
    TIFFImage prewitt_sep = img.SetKernelPrewittSep();
    EXPECT_TRUE(prewitt == prewitt_sep);
    fs::remove(kTestImagePath);
  }
}

TEST(TIFFImageTest, PrewittFilterGPU) {
  std::string cuda_error;
  if (!IsCudaAvailable(&cuda_error)) {
    GTEST_SKIP() << "Skipping CUDA tests: " << cuda_error;
  }
  for (size_t k = 0; k < kTestImagesCount; k++) {
    CreateTestImage(100, 100, k);
    TIFFImage img(kTestImagePath);
    TIFFImage prewitt = img.SetKernel(kKernelPrewitt);
    TIFFImage prewitt_cuda = img.SetKernelCuda(kKernelPrewitt);
    EXPECT_TRUE(prewitt == prewitt_cuda);
    TIFFImage prewitt_cuda_sep = img.SetKernelPrewittSepCuda();
    EXPECT_TRUE(prewitt == prewitt_cuda_sep);
    img.SetImagePatametersForDevice(ImageOperation::Prewitt);
    img.AllocateDeviceMemory();
    img.CopyImageToDevice();
    prewitt_cuda = img.SetKernelCuda(kKernelPrewitt);
    EXPECT_TRUE(prewitt == prewitt_cuda);
    img.SetImagePatametersForDevice(ImageOperation::Prewitt |
                                    ImageOperation::Separated);
    img.CopyImageToDevice();
    prewitt_cuda_sep = img.SetKernelPrewittSepCuda();
    EXPECT_TRUE(prewitt == prewitt_cuda_sep);
    img.FreeDeviceMemory();
    fs::remove(kTestImagePath);
  }
}

TEST(TIFFImageTest, InvalidFile) {
  EXPECT_THROW(TIFFImage("non_existent.tif"), std::runtime_error);
}

TEST(TIFFImageTest, LargeImage) {
  std::string cuda_error;
  if (!IsCudaAvailable(&cuda_error)) {
    GTEST_SKIP() << "Skipping CUDA tests: " << cuda_error;
  }
  TIFFImage img(10000, 10000);
  img.SetImagePatametersForDevice(ImageOperation::GaussianBlur, 3, 1.0);
  img.AllocateDeviceMemory();
  img.CopyImageToDevice();
  EXPECT_NO_THROW(img.GaussianBlurCuda(3, 1.0));
  img.FreeDeviceMemory();
}

TEST(TIFFImageTest, CudaMemoryManagement) {
  std::string cuda_error;
  if (!IsCudaAvailable(&cuda_error)) {
    GTEST_SKIP() << "Skipping CUDA tests: " << cuda_error;
  }
  CreateTestImage(100, 100);
  TIFFImage img(kTestImagePath);
  fs::remove(kTestImagePath);
}
