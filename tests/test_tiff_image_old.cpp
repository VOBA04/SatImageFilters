#include <cuda_runtime.h>
#include <gtest/gtest.h>
#include <opencv2/core/hal/interface.h>
#include <tiffio.h>

#include <cstddef>
#include <cstdint>
#include <cstdlib>
#include <opencv2/core.hpp>
#include <opencv2/core/mat.hpp>
#include <opencv2/opencv.hpp>
#include <random>
#include <string>

#ifdef _WIN32
#include <windows.h>
#else
#include <dirent.h>
#include <sys/stat.h>
#include <unistd.h>
#endif

#include "image_operation.h"
#include "kernel.h"
#include "tiff_image.h"

#ifdef _WIN32
const std::string kTestImage("\\test_image.tiff");
#else
const std::string kTestImage("/test_image.tiff");
#endif

const size_t kTestImagesCount = 9;

void CreateTestImage(std::string temp_dir, int width, int height,
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
      cv::circle(img, cv::Point(width / 2, height / 2), min(width, height) / 4,
                 cv::Scalar(65535), -1);
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
  cv::imwrite(temp_dir + kTestImage, img);
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

inline std::string GetBaseTempPath() {
#ifdef _WIN32
  char temp_path[MAX_PATH];
  if (GetTempPathA(MAX_PATH, temp_path) == 0) {
    std::cerr << "Failed to get temporary directory path" << std::endl;
    exit(1);
  }
  return std::string(temp_path);
#else
  const char* tmpdir = getenv("TMPDIR");
  if (!tmpdir) {
    tmpdir = "/tmp";
  }
  return std::string(tmpdir);
#endif
}

inline std::string GetTempDir() {
  std::random_device rd;
  std::mt19937 gen(rd());
  std::uniform_int_distribution<> dis(1000, 9999);
  std::string temp_dir = GetBaseTempPath();
#ifdef _WIN32
  temp_dir += "\\test_temp_" + std::to_string(dis(gen));
  if (!CreateDirectoryA(temp_dir.c_str(), nullptr)) {
  std:
    std::cerr << "Failed to create temporary directory: " << temp_dir
              << std::endl;
    exit(1);
  }
#else
  temp_dir += "/test_temp_XXXXXX";
  char* temp_dir_cstr = new char[temp_dir.length() + 1];
  strcpy(temp_dir_cstr, temp_dir.c_str());
  if (mkdtemp(temp_dir_cstr) == nullptr) {
    cerr << "Failed to create temporary directory: " << temp_dir << std::endl;
    delete[] temp_dir_cstr;
    exit(1);
  }
  temp_dir = std::string(temp_dir_cstr);
  delete[] temp_dir_cstr;
#endif
  return temp_dir;
}

inline void RecursiveDelete(const std::string& path) {
#ifdef _WIN32
  WIN32_FIND_DATA find_data;
  HANDLE h_find = FindFirstFileA((path + "\\*").c_str(), &find_data);
  if (h_find == INVALID_HANDLE_VALUE) {
    return;
  }
  do {
    if (strcmp(find_data.cFileName, ".") == 0 ||
        strcmp(find_data.cFileName, "..") == 0) {
      continue;
    }
    std::string sub_path = path + "\\" + find_data.cFileName;
    if (find_data.dwFileAttributes & FILE_ATTRIBUTE_DIRECTORY) {
      RecursiveDelete(sub_path);
    } else {
      if (!DeleteFileA(sub_path.c_str())) {
        std::cerr << "Failed to delete file: " << sub_path << std::endl;
      }
    }
  } while (FindNextFileA(h_find, &find_data) != 0);
  FindClose(h_find);
  if (!RemoveDirectoryA(path.c_str())) {
    std::cerr << "Failed to delete directory: " << path << std::endl;
  }
#else
  DIR* dir = opendir(path.c_str());
  if (!dir) {
    return;
  }
  struct dirent* entry;
  while ((entry = readdir(dir)) != nullptr) {
    if (strcmp(entry->d_name, ".") == 0 || strcmp(entry->d_name, "..") == 0) {
      continue;
    }
    std::string sub_path = path + "/" + entry->d_name;
    struct stat statbuf;
    if (stat(sub_path.c_str(), &statbuf) == 0) {
      if (S_ISDIR(statbuf.st_mode)) {
        RecursiveDelete(sub_path);
      } else {
        if (unlink(sub_path.c_str()) != 0) {
          std::cerr << "Failed to delete file: " << sub_path << std::endl;
        }
      }
    }
  }
  closedir(dir);
  if (rmdir(path.c_str()) != 0) {
    std::cerr << "Failed to delete directory: " << path << std::endl;
  }
#endif
}

inline void DeleteTempDir(const std::string& temp_dir) {
  RecursiveDelete(temp_dir);
}

TEST(TIFFImageTest, LoadAndSave) {
  std::string temp_dir = GetTempDir();
#ifdef _WIN32
  std::string test_image_copy("\\test_image_copy.tiff");
#else
  std::string test_image_copy("/test_image_copy.tiff");
#endif
  CreateTestImage(temp_dir, 100, 100);
  TIFFImage img(temp_dir + kTestImage);
  EXPECT_EQ(img.GetWidth(), 100);
  EXPECT_EQ(img.GetHeight(), 100);
  EXPECT_NO_THROW(img.Save(temp_dir + test_image_copy));
  EXPECT_THROW(img.Save(temp_dir + "tests" + test_image_copy),
               std::runtime_error);
  TIFFImage img_copy(temp_dir + test_image_copy);
  EXPECT_TRUE(img == img_copy);
  img.Close();
  img_copy.Close();
  DeleteTempDir(temp_dir);
}

TEST(TIFFImageTest, CopyConstructor) {
  std::string temp_dir = GetTempDir();
  CreateTestImage(temp_dir, 100, 100);
  TIFFImage img(temp_dir + kTestImage);
  TIFFImage img_copy(img);
  EXPECT_TRUE(img == img_copy);
  img.Close();
  DeleteTempDir(temp_dir);
}

TEST(TIFFImageTest, CopyAssignment) {
  std::string temp_dir = GetTempDir();
  CreateTestImage(temp_dir, 100, 100);
  TIFFImage img(temp_dir + kTestImage);
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
  std::string temp_dir = GetTempDir();
  CreateTestImage(temp_dir, 100, 100);
  TIFFImage img(temp_dir + kTestImage);
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
  std::string temp_dir = GetTempDir();
  for (size_t k = 0; k < kTestImagesCount; k++) {
    CreateTestImage(temp_dir, 100, 100, k);
    TIFFImage img(temp_dir + kTestImage);
    TIFFImage blurred = img.GaussianBlur(3, 1.0);
    cv::Mat cv_img = cv::imread(temp_dir + kTestImage, cv::IMREAD_UNCHANGED);
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
  std::string temp_dir = GetTempDir();
  for (size_t k = 0; k < kTestImagesCount; k++) {
    CreateTestImage(temp_dir, 100, 100, k);
    TIFFImage img(temp_dir + kTestImage);
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
  }
  DeleteTempDir(temp_dir);
}

TEST(TIFFImageTest, SobelFilterCPU) {
  std::string temp_dir = GetTempDir();
  for (size_t k = 0; k < kTestImagesCount; k++) {
    CreateTestImage(temp_dir, 100, 100, k);
    TIFFImage img(temp_dir + kTestImage);
    TIFFImage sobel_x = img.SetKernel(kKernelSobel, false);
    TIFFImage sobel_y = img.SetKernel(
        kKernelSobel.Rotate(KernelRotationDegrees::DEGREES_90), false);
    TIFFImage sobel = img.SetKernel(kKernelSobel);
    cv::Mat cv_img = cv::imread(temp_dir + kTestImage, cv::IMREAD_UNCHANGED);
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
  std::string temp_dir = GetTempDir();
  for (size_t k = 0; k < kTestImagesCount; k++) {
    CreateTestImage(temp_dir, 100, 100, k);
    TIFFImage img(temp_dir + kTestImage);
    TIFFImage sobel = img.SetKernel(kKernelSobel);
    TIFFImage sobel_cuda = img.SetKernelCuda(kKernelSobel);
    EXPECT_TRUE(sobel == sobel_cuda) << "Image: " << k << std::endl
                                     << "Sobel: " << std::endl
                                     << sobel << "Sobel CUDA: " << std::endl
                                     << sobel_cuda;
    TIFFImage sobel_cuda_sep = img.SetKernelSobelSepCuda();
    EXPECT_TRUE(sobel == sobel_cuda_sep) << "Image: " << k;
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
  std::string temp_dir = GetTempDir();
  for (size_t k = 0; k < kTestImagesCount; k++) {
    CreateTestImage(temp_dir, 100, 100, k);
    TIFFImage img(temp_dir + kTestImage);
    TIFFImage prewitt_x = img.SetKernel(kKernelPrewitt, false);
    TIFFImage prewitt_y = img.SetKernel(
        kKernelPrewitt.Rotate(KernelRotationDegrees::DEGREES_90), false);
    TIFFImage prewitt = img.SetKernel(kKernelPrewitt);
    cv::Mat cv_img = cv::imread(temp_dir + kTestImage, cv::IMREAD_UNCHANGED);
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
  std::string temp_dir = GetTempDir();
  for (size_t k = 0; k < kTestImagesCount; k++) {
    CreateTestImage(temp_dir, 100, 100, k);
    TIFFImage img(temp_dir + kTestImage);
    TIFFImage prewitt = img.SetKernel(kKernelPrewitt);
    TIFFImage prewitt_cuda = img.SetKernelCuda(kKernelPrewitt);
    EXPECT_TRUE(prewitt == prewitt_cuda) << "Image: " << k;
    TIFFImage prewitt_cuda_sep = img.SetKernelPrewittSepCuda();
    EXPECT_TRUE(prewitt == prewitt_cuda_sep) << "Image: " << k;
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
  std::string temp_dir = GetTempDir();
  CreateTestImage(temp_dir, 100, 100);
  TIFFImage img(temp_dir + kTestImage);
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
