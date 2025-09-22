#include <cuda_runtime.h>
#include <gtest/gtest.h>
#include <opencv2/core/hal/interface.h>
#include <tiffio.h>

#include <cstddef>
#include <cstdint>
#include <cstdlib>
#include <filesystem>
#include <opencv2/core.hpp>
#include <opencv2/core/mat.hpp>
#include <opencv2/opencv.hpp>
#include <random>
#include <string>

#include "image_operation.h"
#include "kernel.h"
#include "tiff_image.h"

const std::string kTestImage("test_image.tiff");

namespace fs = std::filesystem;

const size_t kTestImagesCount = 9;

void CreateTestImage(fs::path temp_dir, int width, int height,
                     uint8_t image_type = 0) {
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
  cv::imwrite((temp_dir / kTestImage).generic_string(), img);
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

inline fs::path GetTempDir() {
  std::random_device rd;
  std::mt19937 gen(rd());
  std::uniform_int_distribution<> dis(1000, 9999);
  fs::path temp_dir =
      fs::temp_directory_path() / ("test_temp_" + std::to_string(dis(gen)));
  if (!fs::create_directory(temp_dir)) {
    std::cerr << "Failed to create a temporary directory: " << temp_dir;
    exit(1);
  }
  return temp_dir;
}

inline void DeleteTempDir(fs::path temp_dir) {
  if (fs::exists(temp_dir)) {
    fs::remove_all(temp_dir);
  }
}

TEST(TIFFImageTest, LoadAndSave) {
  fs::path temp_dir = GetTempDir();
  CreateTestImage(temp_dir, 100, 100);
  TIFFImage img(temp_dir / kTestImage);
  EXPECT_EQ(img.GetWidth(), 100);
  EXPECT_EQ(img.GetHeight(), 100);
  EXPECT_NO_THROW(img.Save(temp_dir / "test_image_copy.tiff"));
  EXPECT_THROW(img.Save(temp_dir / "tests/test_image_copy.tiff"),
               std::runtime_error);
  TIFFImage img_copy(temp_dir / "test_image_copy.tiff");
  EXPECT_TRUE(img == img_copy);
  img.Close();
  img_copy.Close();
  DeleteTempDir(temp_dir);
}

TEST(TIFFImageTest, CopyConstructor) {
  fs::path temp_dir = GetTempDir();
  CreateTestImage(temp_dir, 100, 100);
  TIFFImage img(temp_dir / kTestImage);
  TIFFImage img_copy(img);
  EXPECT_TRUE(img == img_copy);
  img.Close();
  DeleteTempDir(temp_dir);
}

TEST(TIFFImageTest, CopyAssignment) {
  fs::path temp_dir = GetTempDir();
  CreateTestImage(temp_dir, 100, 100);
  TIFFImage img(temp_dir / kTestImage);
  TIFFImage img_copy;
  img_copy = img;
  EXPECT_TRUE(img == img_copy);
  img = img_copy;
  EXPECT_TRUE(img == img_copy);
  EXPECT_NO_FATAL_FAILURE(img_copy.CopyFields(img));
  img.Close();
  DeleteTempDir(temp_dir);
}

TEST(TIFFImageTest, NotEqual) {
  fs::path temp_dir = GetTempDir();
  CreateTestImage(temp_dir, 100, 100);
  TIFFImage img(temp_dir / kTestImage);
  TIFFImage img2(200, 200);
  TIFFImage img3(img);
  img3.Set(0, 0, 1);
  EXPECT_FALSE(img == img2);
  EXPECT_FALSE(img == img3);
  img.Close();
  DeleteTempDir(temp_dir);
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
  fs::path temp_dir = GetTempDir();
  for (size_t k = 0; k < kTestImagesCount; k++) {
    CreateTestImage(temp_dir, 100, 100, k);
    TIFFImage img(temp_dir / kTestImage);
    TIFFImage blurred = img.GaussianBlur(3, 1.0);
    cv::Mat cv_img = cv::imread((temp_dir / kTestImage).generic_string(),
                                cv::IMREAD_UNCHANGED);
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
  }
  DeleteTempDir(temp_dir);
}

TEST(TIFFImageTest, GaussianBlurGPU) {
  std::string cuda_error;
  if (!IsCudaAvailable(&cuda_error)) {
    GTEST_SKIP() << "Skipping CUDA tests: " << cuda_error;
  }
  fs::path temp_dir = GetTempDir();
  for (size_t k = 0; k < kTestImagesCount; k++) {
    CreateTestImage(temp_dir, 100, 100, k);
    TIFFImage img(temp_dir / kTestImage);
    TIFFImage blurred_cpu = img.GaussianBlur(3, 1.0);
    TIFFImage blurred_cpu_2 = img.GaussianBlur(3, 2.0);
    TIFFImage blurred_cuda = img.GaussianBlurCuda(3, 1.0, false);
    TIFFImage blurred_cuda_shared = img.GaussianBlurCuda(3, 1.0, true);
    TIFFImage blurred_cuda_sep = img.GaussianBlurSepCuda(3, 1.0, false);
    TIFFImage blurred_cuda_sep_shared = img.GaussianBlurSepCuda(3, 1.0, true);
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
        EXPECT_NEAR(blurred_cpu.Get(j, i), blurred_cuda_shared.Get(j, i), 1)
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
        EXPECT_NEAR(blurred_cpu.Get(j, i), blurred_cuda_sep_shared.Get(j, i), 1)
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
  }
  DeleteTempDir(temp_dir);
}

TEST(TIFFImageTest, SobelFilterCPU) {
  fs::path temp_dir = GetTempDir();
  for (size_t k = 0; k < kTestImagesCount; k++) {
    CreateTestImage(temp_dir, 100, 100, k);
    TIFFImage img(temp_dir / kTestImage);
    TIFFImage sobel_x = img.SetKernel(kKernelSobel, false);
    TIFFImage sobel_y = img.SetKernel(
        kKernelSobel.Rotate(KernelRotationDegrees::DEGREES_90), false);
    TIFFImage sobel = img.SetKernel(kKernelSobel);
    cv::Mat cv_img = cv::imread((temp_dir / kTestImage).generic_string(),
                                cv::IMREAD_UNCHANGED);
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
    EXPECT_TRUE(sobel == sobel_sep) << "Image: " << k;
  }
  DeleteTempDir(temp_dir);
}

TEST(TIFFImageTest, SobelFilterGPU) {
  std::string cuda_error;
  if (!IsCudaAvailable(&cuda_error)) {
    GTEST_SKIP() << "Skipping CUDA tests: " << cuda_error;
  }
  fs::path temp_dir = GetTempDir();
  for (size_t k = 0; k < kTestImagesCount; k++) {
    CreateTestImage(temp_dir, 100, 100, k);
    TIFFImage img(temp_dir / kTestImage);
    TIFFImage sobel = img.SetKernel(kKernelSobel);
    TIFFImage sobel_cuda = img.SetKernelCuda(kKernelSobel, false);
    EXPECT_TRUE(sobel == sobel_cuda) << "Image: " << k << std::endl
                                     << "Sobel: " << std::endl
                                     << sobel << "Sobel CUDA: " << std::endl
                                     << sobel_cuda;
    TIFFImage sobel_cuda_shared = img.SetKernelCuda(kKernelSobel, true);
    EXPECT_TRUE(sobel == sobel_cuda_shared) << "Image: " << k;
    TIFFImage sobel_cuda_sep = img.SetKernelSobelSepCuda();
    EXPECT_TRUE(sobel == sobel_cuda_sep) << "Image: " << k;
    TIFFImage sobel_cuda_sep_shared = img.SetKernelSobelSepCuda(true);
    EXPECT_TRUE(sobel == sobel_cuda_sep_shared) << "Image: " << k;
    img.SetImagePatametersForDevice(ImageOperation::Sobel);
    img.AllocateDeviceMemory();
    img.CopyImageToDevice();
    sobel_cuda = img.SetKernelCuda(kKernelSobel);
    EXPECT_TRUE(sobel == sobel_cuda) << "Image: " << k;
    img.SetImagePatametersForDevice(ImageOperation::Sobel |
                                    ImageOperation::Separated);
    img.CopyImageToDevice();
    sobel_cuda_sep = img.SetKernelSobelSepCuda();
    EXPECT_TRUE(sobel == sobel_cuda_sep) << "Image: " << k;
    img.FreeDeviceMemory();
  }
  DeleteTempDir(temp_dir);
}

TEST(TIFFImageTest, PrewittFilterCPU) {
  fs::path temp_dir = GetTempDir();
  for (size_t k = 0; k < kTestImagesCount; k++) {
    CreateTestImage(temp_dir, 100, 100, k);
    TIFFImage img(temp_dir / kTestImage);
    TIFFImage prewitt_x = img.SetKernel(kKernelPrewitt, false);
    TIFFImage prewitt_y = img.SetKernel(
        kKernelPrewitt.Rotate(KernelRotationDegrees::DEGREES_90), false);
    TIFFImage prewitt = img.SetKernel(kKernelPrewitt);
    cv::Mat cv_img = cv::imread((temp_dir / kTestImage).generic_string(),
                                cv::IMREAD_UNCHANGED);
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
    EXPECT_TRUE(prewitt == prewitt_sep) << "Image: " << k;
  }
  DeleteTempDir(temp_dir);
}

TEST(TIFFImageTest, PrewittFilterGPU) {
  std::string cuda_error;
  if (!IsCudaAvailable(&cuda_error)) {
    GTEST_SKIP() << "Skipping CUDA tests: " << cuda_error;
  }
  fs::path temp_dir = GetTempDir();
  for (size_t k = 0; k < kTestImagesCount; k++) {
    CreateTestImage(temp_dir, 100, 100, k);
    TIFFImage img(temp_dir / kTestImage);
    TIFFImage prewitt = img.SetKernel(kKernelPrewitt);
    TIFFImage prewitt_cuda = img.SetKernelCuda(kKernelPrewitt);
    EXPECT_TRUE(prewitt == prewitt_cuda) << "Image: " << k;
    TIFFImage prewitt_cuda_shared = img.SetKernelCuda(kKernelPrewitt, true);
    EXPECT_TRUE(prewitt == prewitt_cuda_shared) << "Image: " << k;
    TIFFImage prewitt_cuda_sep = img.SetKernelPrewittSepCuda();
    EXPECT_TRUE(prewitt == prewitt_cuda_sep) << "Image: " << k;
    TIFFImage prewitt_cuda_sep_shared = img.SetKernelPrewittSepCuda(true);
    EXPECT_TRUE(prewitt == prewitt_cuda_sep_shared) << "Image: " << k;
    img.SetImagePatametersForDevice(ImageOperation::Prewitt);
    img.AllocateDeviceMemory();
    img.CopyImageToDevice();
    prewitt_cuda = img.SetKernelCuda(kKernelPrewitt);
    EXPECT_TRUE(prewitt == prewitt_cuda) << "Image: " << k;
    img.SetImagePatametersForDevice(ImageOperation::Prewitt |
                                    ImageOperation::Separated);
    img.CopyImageToDevice();
    prewitt_cuda_sep = img.SetKernelPrewittSepCuda();
    EXPECT_TRUE(prewitt == prewitt_cuda_sep) << "Image: " << k;
    img.FreeDeviceMemory();
  }
  DeleteTempDir(temp_dir);
}

TEST(TIFFImageTest, SetKernelArbitraryCPU) {
  fs::path temp_dir = GetTempDir();
  Kernel<int> box3(3, 3, {{1, 1, 1}, {1, 1, 1}, {1, 1, 1}}, false);
  Kernel<int> sharpen3(3, 3, {{0, -1, 0}, {-1, 5, -1}, {0, -1, 0}}, false);
  Kernel<int> emboss3(3, 3, {{-2, -1, 0}, {-1, 1, 1}, {0, 1, 2}}, true);
  for (size_t k = 0; k < kTestImagesCount; ++k) {
    CreateTestImage(temp_dir, 100, 100, k);
    TIFFImage img(temp_dir / kTestImage);
    auto cv_ref_single = [&](const Kernel<int>& ker) {
      cv::Mat in = cv::imread((temp_dir / kTestImage).generic_string(),
                              cv::IMREAD_UNCHANGED);
      cv::Mat kcv(ker.GetHeight(), ker.GetWidth(), CV_32F);
      for (int r = 0; r < (int)ker.GetHeight(); ++r) {
        for (int c = 0; c < (int)ker.GetWidth(); ++c) {
          kcv.at<float>(r, c) = static_cast<float>(ker.Get(c, r));
        }
      }
      cv::Mat tmp32f;
      cv::filter2D(in, tmp32f, CV_32F, kcv, cv::Point(-1, -1), 0,
                   cv::BORDER_REPLICATE);
      cv::Mat abs32f;
      cv::absdiff(tmp32f, cv::Scalar(0), abs32f);
      cv::Mat clipped32f;
      cv::min(abs32f, 65535, clipped32f);
      cv::Mat out16u;
      clipped32f.convertTo(out16u, CV_16U);
      return out16u;
    };
    auto cv_ref_rotate = [&](const Kernel<int>& ker) {
      cv::Mat in = cv::imread((temp_dir / kTestImage).generic_string(),
                              cv::IMREAD_UNCHANGED);
      cv::Mat kcv(ker.GetHeight(), ker.GetWidth(), CV_32F);
      for (int r = 0; r < (int)ker.GetHeight(); ++r) {
        for (int c = 0; c < (int)ker.GetWidth(); ++c) {
          kcv.at<float>(r, c) = static_cast<float>(ker.Get(c, r));
        }
      }
      Kernel<int> ker_rot = ker.Rotate(KernelRotationDegrees::DEGREES_90);
      cv::Mat kcv_rot(ker_rot.GetHeight(), ker_rot.GetWidth(), CV_32F);
      for (int r = 0; r < (int)ker_rot.GetHeight(); ++r) {
        for (int c = 0; c < (int)ker_rot.GetWidth(); ++c) {
          kcv_rot.at<float>(r, c) = static_cast<float>(ker_rot.Get(c, r));
        }
      }
      cv::Mat gx32f, gy32f;
      cv::filter2D(in, gx32f, CV_32F, kcv, cv::Point(-1, -1), 0,
                   cv::BORDER_REPLICATE);
      cv::filter2D(in, gy32f, CV_32F, kcv_rot, cv::Point(-1, -1), 0,
                   cv::BORDER_REPLICATE);
      cv::Mat absx, absy;
      cv::absdiff(gx32f, cv::Scalar(0), absx);
      cv::absdiff(gy32f, cv::Scalar(0), absy);
      cv::Mat sum32f = absx + absy;
      cv::Mat clipped32f;
      cv::min(sum32f, 65535, clipped32f);
      cv::Mat out16u;
      clipped32f.convertTo(out16u, CV_16U);
      return out16u;
    };
    {
      TIFFImage out = img.SetKernel(box3, false);
      cv::Mat cv_ref = cv_ref_single(box3);
      bool failed = false;
      for (size_t i = 0; i < img.GetHeight() && !failed; ++i) {
        for (size_t j = 0; j < img.GetWidth(); ++j) {
          EXPECT_EQ(out.Get(j, i), cv_ref.at<uint16_t>(i, j))
              << "Mismatch at (" << j << ", " << i << ") image " << k;
          if (HasFailure()) {
            failed = true;
          }
        }
      }
    }
    {
      TIFFImage out = img.SetKernel(sharpen3, false);
      cv::Mat cv_ref = cv_ref_single(sharpen3);
      bool failed = false;
      for (size_t i = 0; i < img.GetHeight() && !failed; ++i) {
        for (size_t j = 0; j < img.GetWidth(); ++j) {
          EXPECT_EQ(out.Get(j, i), cv_ref.at<uint16_t>(i, j))
              << "Mismatch at (" << j << ", " << i << ") image " << k;
          if (HasFailure()) {
            failed = true;
          }
        }
      }
    }
    {
      TIFFImage out = img.SetKernel(emboss3, true);
      cv::Mat cv_ref = cv_ref_rotate(emboss3);
      bool failed = false;
      for (size_t i = 0; i < img.GetHeight() && !failed; ++i) {
        for (size_t j = 0; j < img.GetWidth(); ++j) {
          EXPECT_EQ(out.Get(j, i), cv_ref.at<uint16_t>(i, j))
              << "Mismatch at (" << j << ", " << i << ") image " << k;
          if (HasFailure()) {
            failed = true;
          }
        }
      }
    }
  }
  DeleteTempDir(temp_dir);
}

TEST(TIFFImageTest, SetKernelArbitraryGPU) {
  std::string cuda_error;
  if (!IsCudaAvailable(&cuda_error)) {
    GTEST_SKIP() << "Skipping CUDA tests: " << cuda_error;
  }
  fs::path temp_dir = GetTempDir();
  Kernel<int> sharpen3(3, 3, {{0, -1, 0}, {-1, 5, -1}, {0, -1, 0}}, false);
  Kernel<int> emboss3(3, 3, {{-2, -1, 0}, {-1, 1, 1}, {0, 1, 2}}, true);
  for (size_t k = 0; k < kTestImagesCount; ++k) {
    CreateTestImage(temp_dir, 100, 100, k);
    TIFFImage img(temp_dir / kTestImage);
    TIFFImage ref_sharpen = img.SetKernel(sharpen3, false);
    TIFFImage ref_emboss_rot = img.SetKernel(emboss3, true);
    TIFFImage gpu_sharpen_global = img.SetKernelCuda(sharpen3, false, false);
    EXPECT_TRUE(ref_sharpen == gpu_sharpen_global) << "Image: " << k;
    TIFFImage gpu_sharpen_shared = img.SetKernelCuda(sharpen3, true, false);
    EXPECT_TRUE(ref_sharpen == gpu_sharpen_shared) << "Image: " << k;
    TIFFImage gpu_emboss_rot_shared = img.SetKernelCuda(emboss3, true, true);
    EXPECT_TRUE(ref_emboss_rot == gpu_emboss_rot_shared) << "Image: " << k;
  }
  DeleteTempDir(temp_dir);
}

TEST(TIFFImageTest, InvalidFile) {
  EXPECT_THROW(TIFFImage("non_existent.tiff"), std::runtime_error);
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
  fs::path temp_dir = GetTempDir();
  CreateTestImage(temp_dir, 100, 100);
  TIFFImage img(temp_dir / kTestImage);
  img.SetImagePatametersForDevice(ImageOperation::GaussianBlur, 3, 1.0);
  img.AllocateDeviceMemory();
  img.CopyImageToDevice();
  TIFFImage blurred = img.GaussianBlurCuda(3, 1.0);
  EXPECT_EQ(blurred.GetWidth(), img.GetWidth());
  EXPECT_EQ(blurred.GetHeight(), img.GetHeight());
  img.FreeDeviceMemory();
  size_t free_memory = 0, total_memory = 0;
  cudaError_t error = cudaMemGetInfo(&free_memory, &total_memory);
  EXPECT_EQ(error, cudaSuccess)
      << "Failed to get CUDA memory info: " << cudaGetErrorString(error);
  EXPECT_GT(free_memory, 0) << "Free memory should be greater than zero.";
  img.Close();
  DeleteTempDir(temp_dir);
}

TEST(CudaMemManagerTest, BasicMemoryManagement) {
  std::string cuda_error;
  if (!IsCudaAvailable(&cuda_error)) {
    GTEST_SKIP() << "Skipping CUDA tests: " << cuda_error;
  }
  CudaMemManager mgr;
  size_t width = 32, height = 32;
  mgr.SetImageSize(width, height);
  mgr.SetImageOperations(ImageOperation::GaussianBlur);
  mgr.SetGaussianParameters(3, 1.0f);
  EXPECT_NO_THROW(mgr.AllocateMemory());
  EXPECT_TRUE(mgr.IsAllocated());
  std::vector<uint16_t> src(width * height, 123);
  std::vector<uint16_t> dst(width * height, 0);
  EXPECT_NO_THROW(mgr.CopyImageToDevice(src.data()));
  EXPECT_NO_THROW(mgr.CopyImageFromDevice(dst.data()));
  EXPECT_NO_THROW(mgr.FreeMemory());
  EXPECT_FALSE(mgr.IsAllocated());
  EXPECT_NO_THROW(mgr.AllocateMemory());
  EXPECT_TRUE(mgr.IsAllocated());
  EXPECT_NO_THROW(mgr.FreeMemory());
}
