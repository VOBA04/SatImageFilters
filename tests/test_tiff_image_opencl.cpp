#include <gtest/gtest.h>
#include <CL/cl.h>
#include <opencv2/core.hpp>
#include <opencv2/opencv.hpp>

#include <cstdint>
#include <filesystem>
#include <random>
#include <string>
#include <vector>

#include "image_operation.h"
#include "kernel.h"
#include "tiff_image.h"

namespace fs = std::filesystem;

static const std::string kTestImage("test_image.tiff");
static const size_t kTestImagesCount = 9;

// --- Helpers (mirrors CUDA test helpers) ---
static inline bool IsOpenCLAvailable(std::string* error_message = nullptr) {
  cl_uint num_platforms = 0;
  cl_int err = clGetPlatformIDs(0, nullptr, &num_platforms);
  if (err != CL_SUCCESS || num_platforms == 0) {
    if (error_message != nullptr) {
      *error_message =
          "No OpenCL platforms found (err=" + std::to_string(err) + ")";
    }
    return false;
  }
  std::vector<cl_platform_id> platforms(num_platforms);
  clGetPlatformIDs(num_platforms, platforms.data(), nullptr);
  for (auto pid : platforms) {
    cl_uint num_devices = 0;
    if (clGetDeviceIDs(pid, CL_DEVICE_TYPE_GPU, 0, nullptr, &num_devices) ==
            CL_SUCCESS &&
        num_devices > 0) {
      return true;
    }
  }
  // fallback to CPU
  for (auto pid : platforms) {
    cl_uint num_devices = 0;
    if (clGetDeviceIDs(pid, CL_DEVICE_TYPE_CPU, 0, nullptr, &num_devices) ==
            CL_SUCCESS &&
        num_devices > 0) {
      return true;
    }
  }
  if (error_message != nullptr) {
    *error_message = "No suitable OpenCL GPU or CPU devices found";
  }
  return false;
}

static inline fs::path GetTempDir() {
  std::random_device rd;
  std::mt19937 gen(rd());
  std::uniform_int_distribution<> dis(1000, 9999);
  fs::path temp_dir =
      fs::temp_directory_path() / ("test_temp_ocl_" + std::to_string(dis(gen)));
  if (!fs::create_directory(temp_dir)) {
    throw std::runtime_error("Failed to create temp dir: " + temp_dir.string());
  }
  return temp_dir;
}

static inline void DeleteTempDir(const fs::path& temp_dir) {
  if (fs::exists(temp_dir)) {
    fs::remove_all(temp_dir);
  }
}

static void CreateTestImage(fs::path temp_dir, int width, int height,
                            uint8_t image_type = 0) {
  cv::Mat img(height, width, CV_16U, cv::Scalar(0));
  switch (image_type) {
    case 0:
    default:
      for (int y = 0; y < height; ++y) {
        for (int x = width / 2; x < width; ++x) {
          img.at<uint16_t>(y, x) = 65535;
        }
      }
      break;
    case 1:
      for (int y = height / 2; y < height; ++y) {
        for (int x = 0; x < width; ++x) {
          img.at<uint16_t>(y, x) = 65535;
        }
      }
      break;
    case 2:
      for (int y = 0; y < height; ++y) {
        for (int x = 0; x < width; ++x) {
          img.at<uint16_t>(y, x) =
              static_cast<uint16_t>((x / (float)width) * 65535);
        }
      }
      break;
    case 3:
      for (int y = 0; y < height; ++y) {
        for (int x = 0; x < width; ++x) {
          img.at<uint16_t>(y, x) =
              static_cast<uint16_t>((y / (float)height) * 65535);
        }
      }
      break;
    case 4:
      cv::theRNG().state = 12345;
      cv::randu(img, 0, 65535);
      break;
    case 5:
      cv::randn(img, 32768, 10000);
      break;
    case 6:
      cv::circle(img, cv::Point(width / 2, height / 2),
                 std::min(width, height) / 4, cv::Scalar(65535), -1);
      break;
    case 7:
      img.at<uint16_t>(height / 2, width / 2) = 65535;
      break;
    case 8:
      for (int y = 0; y < height; ++y) {
        for (int x = 0; x < width; ++x) {
          img.at<uint16_t>(y, x) = (((x / 10 + y / 10) % 2) != 0) ? 65535 : 0;
        }
      }
      break;
  }
  cv::imwrite((temp_dir / kTestImage).generic_string(), img);
}

// --- Tests ---

TEST(OpenCLImageTest, GaussianBlurOpenCL) {
  std::string ocl_error;
  if (!IsOpenCLAvailable(&ocl_error)) {
    GTEST_SKIP() << "Skipping OpenCL tests: " << ocl_error;
  }
  fs::path temp_dir = GetTempDir();
  for (size_t k = 0; k < kTestImagesCount; k++) {
    CreateTestImage(temp_dir, 100, 100, static_cast<uint8_t>(k));
    TIFFImage img(temp_dir / kTestImage);
    TIFFImage blurred_cpu = img.GaussianBlur(3, 1.0);
    TIFFImage blurred_cpu_2 = img.GaussianBlur(3, 2.0);
    TIFFImage blurred_ocl = img.GaussianBlurOpenCL(3, 1.0);
    TIFFImage blurred_ocl_sep = img.GaussianBlurSepOpenCL(3, 1.0);
    TIFFImage blurred_ocl_2 = img.GaussianBlurOpenCL(3, 2.0);

    // Pre-allocated path
    img.SetImagePatametersForOpenCLOps(ImageOperation::GaussianBlur, 3, 1.0);
    img.AllocateOpenCLMemory();
    img.CopyImageToOpenCLDevice();
    TIFFImage blurred_ocl_mem = img.GaussianBlurOpenCL(3, 1.0);
    img.SetImagePatametersForOpenCLOps(ImageOperation::GaussianBlurSep, 3, 1.0);
    img.ReallocateOpenCLMemory();
    img.CopyImageToOpenCLDevice();
    TIFFImage blurred_ocl_sep_mem = img.GaussianBlurSepOpenCL(3, 1.0);
    TIFFImage blurred_ocl_sep_2 = img.GaussianBlurSepOpenCL(3, 2.0);

    bool failed = false;
    for (size_t i = 0; i < img.GetHeight() && !failed; i++) {
      for (size_t j = 0; j < img.GetWidth(); j++) {
        EXPECT_NEAR(blurred_cpu.Get(j, i), blurred_ocl.Get(j, i), 1)
            << "Mismatch at pixel (" << j << ", " << i << ") image " << k;
        if (HasFailure()) {
          failed = true;
          break;
        }
        EXPECT_NEAR(blurred_cpu_2.Get(j, i), blurred_ocl_2.Get(j, i), 1)
            << "Mismatch at pixel (" << j << ", " << i << ") image " << k;
        if (HasFailure()) {
          failed = true;
          break;
        }
        EXPECT_NEAR(blurred_cpu.Get(j, i), blurred_ocl_mem.Get(j, i), 1)
            << "Mismatch at pixel (" << j << ", " << i << ") image " << k;
        if (HasFailure()) {
          failed = true;
          break;
        }
        EXPECT_NEAR(blurred_cpu.Get(j, i), blurred_ocl_sep_mem.Get(j, i), 1)
            << "Mismatch at pixel (" << j << ", " << i << ") image " << k;
        if (HasFailure()) {
          failed = true;
          break;
        }
        EXPECT_NEAR(blurred_cpu_2.Get(j, i), blurred_ocl_sep_2.Get(j, i), 1)
            << "Mismatch at pixel (" << j << ", " << i << ") image " << k;
        if (HasFailure()) {
          failed = true;
          break;
        }
      }
    }
    img.FreeOpenCLMemory();
  }
  DeleteTempDir(temp_dir);
}

TEST(OpenCLImageTest, SobelFilterOpenCL) {
  std::string ocl_error;
  if (!IsOpenCLAvailable(&ocl_error)) {
    GTEST_SKIP() << "Skipping OpenCL tests: " << ocl_error;
  }
  fs::path temp_dir = GetTempDir();
  for (size_t k = 0; k < kTestImagesCount; k++) {
    CreateTestImage(temp_dir, 100, 100, static_cast<uint8_t>(k));
    TIFFImage img(temp_dir / kTestImage);
    TIFFImage sobel = img.SetKernel(kKernelSobel);
    TIFFImage sobel_ocl = img.SetKernelOpenCL(kKernelSobel, false);
    EXPECT_TRUE(sobel == sobel_ocl) << "Image: " << k;

    // Also test shared memory variant (uses local memory)
    TIFFImage sobel_ocl_shared = img.SetKernelOpenCL(kKernelSobel, true);
    EXPECT_TRUE(sobel == sobel_ocl_shared) << "Image: " << k;

    TIFFImage sobel_ocl_sep = img.SetKernelSobelSepOpenCL(false);
    EXPECT_TRUE(sobel == sobel_ocl_sep) << "Image: " << k;
    // Shared-memory separable path
    TIFFImage sobel_ocl_sep_shared = img.SetKernelSobelSepOpenCL(true);
    EXPECT_TRUE(sobel == sobel_ocl_sep_shared) << "Image: " << k;

    // Pre-allocated path
    img.SetImagePatametersForOpenCLOps(ImageOperation::Sobel);
    img.AllocateOpenCLMemory();
    img.CopyImageToOpenCLDevice();
    sobel_ocl = img.SetKernelOpenCL(kKernelSobel);
    EXPECT_TRUE(sobel == sobel_ocl) << "Image: " << k;
    img.SetImagePatametersForOpenCLOps(ImageOperation::Sobel |
                                       ImageOperation::Separated);
    img.ReallocateOpenCLMemory();
    img.CopyImageToOpenCLDevice();
    sobel_ocl_sep = img.SetKernelSobelSepOpenCL(false);
    EXPECT_TRUE(sobel == sobel_ocl_sep) << "Image: " << k;
    sobel_ocl_sep = img.SetKernelSobelSepOpenCL(true);
    EXPECT_TRUE(sobel == sobel_ocl_sep) << "Image: " << k;
    img.FreeOpenCLMemory();
  }
  DeleteTempDir(temp_dir);
}

TEST(OpenCLImageTest, PrewittFilterOpenCL) {
  std::string ocl_error;
  if (!IsOpenCLAvailable(&ocl_error)) {
    GTEST_SKIP() << "Skipping OpenCL tests: " << ocl_error;
  }
  fs::path temp_dir = GetTempDir();
  for (size_t k = 0; k < kTestImagesCount; k++) {
    CreateTestImage(temp_dir, 100, 100, static_cast<uint8_t>(k));
    TIFFImage img(temp_dir / kTestImage);
    TIFFImage prewitt = img.SetKernel(kKernelPrewitt);
    TIFFImage prewitt_ocl = img.SetKernelOpenCL(kKernelPrewitt);
    EXPECT_TRUE(prewitt == prewitt_ocl) << "Image: " << k;
    // Also test shared memory variant (uses local memory)
    TIFFImage prewitt_ocl_shared = img.SetKernelOpenCL(kKernelPrewitt, true);
    EXPECT_TRUE(prewitt == prewitt_ocl_shared) << "Image: " << k;
    TIFFImage prewitt_ocl_sep = img.SetKernelPrewittSepOpenCL(false);
    EXPECT_TRUE(prewitt == prewitt_ocl_sep) << "Image: " << k;
    TIFFImage prewitt_ocl_sep_shared = img.SetKernelPrewittSepOpenCL(true);
    EXPECT_TRUE(prewitt == prewitt_ocl_sep_shared) << "Image: " << k;

    // Pre-allocated path
    img.SetImagePatametersForOpenCLOps(ImageOperation::Prewitt);
    img.AllocateOpenCLMemory();
    img.CopyImageToOpenCLDevice();
    prewitt_ocl = img.SetKernelOpenCL(kKernelPrewitt);
    EXPECT_TRUE(prewitt == prewitt_ocl) << "Image: " << k;
    img.SetImagePatametersForOpenCLOps(ImageOperation::Prewitt |
                                       ImageOperation::Separated);
    img.ReallocateOpenCLMemory();
    img.CopyImageToOpenCLDevice();
    prewitt_ocl_sep = img.SetKernelPrewittSepOpenCL(false);
    EXPECT_TRUE(prewitt == prewitt_ocl_sep) << "Image: " << k;
    prewitt_ocl_sep = img.SetKernelPrewittSepOpenCL(true);
    EXPECT_TRUE(prewitt == prewitt_ocl_sep) << "Image: " << k;
    img.FreeOpenCLMemory();
  }
  DeleteTempDir(temp_dir);
}

TEST(OpenCLImageTest, Arbitrary3x3KernelSharedVsNonShared) {
  std::string ocl_error;
  if (!IsOpenCLAvailable(&ocl_error)) {
    GTEST_SKIP() << "Skipping OpenCL tests: " << ocl_error;
  }
  namespace fs = std::filesystem;
  fs::path temp_dir = GetTempDir();
  CreateTestImage(temp_dir, 64, 64, 4);  // random image
  TIFFImage img(temp_dir / kTestImage);

  // Simple 3x3 kernel (mean blur) to exercise generic path
  Kernel<int> mean3(3, 3, {{1, 1, 1}, {1, 1, 1}, {1, 1, 1}}, false);

  TIFFImage cpu = img.SetKernel(mean3, false);
  TIFFImage ocl_noshared = img.SetKernelOpenCL(mean3, false, false);
  TIFFImage ocl_shared = img.SetKernelOpenCL(mean3, true, false);

  EXPECT_TRUE(cpu == ocl_noshared);
  EXPECT_TRUE(cpu == ocl_shared);
  DeleteTempDir(temp_dir);
}

TEST(OpenCLImageTest, Arbitrary5x5KernelSharedVsNonShared) {
  std::string ocl_error;
  if (!IsOpenCLAvailable(&ocl_error)) {
    GTEST_SKIP() << "Skipping OpenCL tests: " << ocl_error;
  }
  namespace fs = std::filesystem;
  fs::path temp_dir = GetTempDir();
  CreateTestImage(temp_dir, 96, 96, 4);  // random image
  TIFFImage img(temp_dir / kTestImage);

  // 5x5 Gaussian-like integer kernel (unnormalized to preserve integer math)
  Kernel<int> k5(5, 5,
                 {{1, 4, 6, 4, 1},
                  {4, 16, 24, 16, 4},
                  {6, 24, 36, 24, 6},
                  {4, 16, 24, 16, 4},
                  {1, 4, 6, 4, 1}},
                 false);

  TIFFImage cpu = img.SetKernel(k5, false);
  TIFFImage ocl_noshared = img.SetKernelOpenCL(k5, false, false);
  TIFFImage ocl_shared = img.SetKernelOpenCL(k5, true, false);

  EXPECT_TRUE(cpu == ocl_noshared);
  EXPECT_TRUE(cpu == ocl_shared);
  DeleteTempDir(temp_dir);
}

TEST(OpenCLImageTest, LargeImageOpenCL) {
  std::string ocl_error;
  if (!IsOpenCLAvailable(&ocl_error)) {
    GTEST_SKIP() << "Skipping OpenCL tests: " << ocl_error;
  }
  TIFFImage img(4000, 4000);
  img.SetImagePatametersForOpenCLOps(ImageOperation::GaussianBlur, 3, 1.0);
  img.AllocateOpenCLMemory();
  // Image is uninitialized, just check no-throw path
  EXPECT_NO_THROW(img.GaussianBlurOpenCL(3, 1.0));
  img.FreeOpenCLMemory();
}

TEST(OpenCLImageTest, OpenCLMemoryManagement) {
  std::string ocl_error;
  if (!IsOpenCLAvailable(&ocl_error)) {
    GTEST_SKIP() << "Skipping OpenCL tests: " << ocl_error;
  }
  fs::path temp_dir = GetTempDir();
  CreateTestImage(temp_dir, 64, 64);
  TIFFImage img(temp_dir / kTestImage);
  img.SetImagePatametersForOpenCLOps(ImageOperation::GaussianBlur, 3, 1.0);
  EXPECT_NO_THROW(img.AllocateOpenCLMemory());
  EXPECT_NO_THROW(img.CopyImageToOpenCLDevice());
  TIFFImage blurred = img.GaussianBlurOpenCL(3, 1.0);
  EXPECT_EQ(blurred.GetWidth(), img.GetWidth());
  EXPECT_EQ(blurred.GetHeight(), img.GetHeight());
  EXPECT_NO_THROW(img.FreeOpenCLMemory());
  DeleteTempDir(temp_dir);
}
