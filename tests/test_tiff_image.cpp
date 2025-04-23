#include <gtest/gtest.h>
#include <opencv2/core/hal/interface.h>
#include <tiffio.h>
#include "kernel.h"
#include "tiff_image.h"
#include <cstdlib>
#include <opencv2/core.hpp>
#include <opencv2/core/mat.hpp>
#include <opencv2/opencv.hpp>
#include <string>
#include <filesystem>

const std::string kTestImagePath =
    std::string(PROJECT_SOURCE_DIR) + "/tests/test_image.tiff";

namespace fs = std::filesystem;

void CreateTestImage(int width, int height) {
  cv::Mat img(height, width, CV_16U, cv::Scalar(0));
  for (int y = 0; y < height; ++y) {
    for (int x = width / 2; x < width; ++x) {
      img.at<uint16_t>(y, x) = 65535;
    }
  }
  cv::imwrite(kTestImagePath, img);
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

TEST(TIFFImageTest, GaussianBlur) {
  CreateTestImage(100, 100);
  TIFFImage img(kTestImagePath);
  TIFFImage blurred_cpu = img.GaussianBlur(3, 1.0);
  cv::Mat cv_img = cv::imread(kTestImagePath, cv::IMREAD_UNCHANGED);
  cv::Mat blurred_cv;
  cv::GaussianBlur(cv_img, blurred_cv, cv::Size(3, 3), 1.0);
  for (size_t i = 0; i < img.GetHeight(); ++i) {
    for (size_t j = 0; j < img.GetWidth(); ++j) {
      EXPECT_NEAR(blurred_cpu.Get(j, i), blurred_cv.at<uint16_t>(i, j), 1);
    }
  }
  TIFFImage blurred_cpu_sep = img.GaussianBlurSep(3, 1.0);
  EXPECT_TRUE(blurred_cpu == blurred_cpu_sep);
  TIFFImage blurred_cuda = img.GaussianBlurCuda(3, 1.0);
  EXPECT_TRUE(blurred_cpu == blurred_cuda);
  TIFFImage blurred_cuda_sep = img.GaussianBlurSepCuda(3, 1.0);
  EXPECT_TRUE(blurred_cpu == blurred_cuda_sep);
  fs::remove(kTestImagePath);
}

TEST(TIFFImageTest, SobelFilter) {
  CreateTestImage(100, 100);
  TIFFImage img(kTestImagePath);
  TIFFImage sobel_x = img.SetKernel(kKernelSobel, false);
  TIFFImage sobel_y = img.SetKernel(
      kKernelSobel.Rotate(KernelRotationDegrees::DEGREES_90), false);
  TIFFImage sobel = img.SetKernel(kKernelSobel);
  cv::Mat cv_img = cv::imread(kTestImagePath, cv::IMREAD_UNCHANGED);
  cv::Mat sobel_x_cv, sobel_y_cv, sobel_cv;
  cv::Sobel(cv_img, sobel_x_cv, CV_16U, 1, 0);
  cv::Sobel(cv_img, sobel_y_cv, CV_16U, 0, 1);
  cv::addWeighted(sobel_x_cv, 1, sobel_y_cv, 1, 0, sobel_cv);
  for (size_t i = 0; i < img.GetHeight(); ++i) {
    for (size_t j = 0; j < img.GetWidth(); ++j) {
      EXPECT_EQ(sobel_x.Get(j, i), sobel_x_cv.at<uint16_t>(i, j));
      EXPECT_EQ(sobel_y.Get(j, i), sobel_y_cv.at<uint16_t>(i, j));
      EXPECT_EQ(sobel.Get(j, i), sobel_cv.at<uint16_t>(i, j));
    }
  }
  TIFFImage sobel_sep = img.SetKernelSobelSep();
  EXPECT_TRUE(sobel == sobel_sep);
  TIFFImage sobel_cuda = img.SetKernelCuda(kKernelSobel);
  EXPECT_TRUE(sobel == sobel_cuda);
  TIFFImage sobel_cuda_sep = img.SetKernelSobelSepCuda();
  EXPECT_TRUE(sobel == sobel_cuda_sep);
  fs::remove(kTestImagePath);
}

TEST(TIFFImageTest, PrewittFilter) {
  CreateTestImage(100, 100);
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
  cv::filter2D(cv_img, prewitt_x_cv, CV_32F, kernel_x);
  cv::filter2D(cv_img, prewitt_y_cv, CV_32F, kernel_y);
  cv::Mat abs_prewitt_x_cv, abs_prewitt_y_cv;
  abs_prewitt_x_cv = cv::abs(prewitt_x_cv);
  abs_prewitt_y_cv = cv::abs(prewitt_y_cv);
  abs_prewitt_x_cv.convertTo(prewitt_x_cv, CV_16U);
  abs_prewitt_y_cv.convertTo(prewitt_y_cv, CV_16U);
  cv::addWeighted(prewitt_x_cv, 1, prewitt_y_cv, 1, 0, prewitt_cv);
  for (size_t i = 0; i < img.GetHeight(); ++i) {
    for (size_t j = 0; j < img.GetWidth(); ++j) {
      EXPECT_EQ(prewitt_x.Get(j, i), prewitt_x_cv.at<uint16_t>(i, j));
      EXPECT_EQ(prewitt_y.Get(j, i), prewitt_y_cv.at<uint16_t>(i, j));
      EXPECT_EQ(prewitt.Get(j, i), prewitt_cv.at<uint16_t>(i, j));
    }
  }
  TIFFImage prewitt_sep = img.SetKernelPrewittSep();
  EXPECT_TRUE(prewitt == prewitt_sep);
  TIFFImage prewitt_cuda = img.SetKernelCuda(kKernelPrewitt);
  EXPECT_TRUE(prewitt == prewitt_cuda);
  TIFFImage prewitt_cuda_sep = img.SetKernelPrewittSepCuda();
  EXPECT_TRUE(prewitt == prewitt_cuda_sep);
  fs::remove(kTestImagePath);
}

TEST(TIFFImageTest, InvalidFile) {
  EXPECT_THROW(TIFFImage("non_existent.tif"), std::runtime_error);
}

TEST(TIFFImageTest, LargeImage) {
  TIFFImage img(10000, 10000);
  img.ImageToDeviceMemory(ImageOperation::GaussianBlur, 3, 1.0);
  EXPECT_NO_THROW(img.GaussianBlurCuda(3, 1.0));
  img.FreeDeviceMemory();
}

TEST(TIFFImageTest, CudaMemoryManagement) {
  CreateTestImage(100, 100);
  TIFFImage img(kTestImagePath);
  EXPECT_NO_THROW(
      img.ImageToDeviceMemory(ImageOperation::GaussianBlur, 3, 1.0));
  TIFFImage img_copy = img;
  img_copy.CopyImageToDevice();
  TIFFImage blurred = img_copy.GaussianBlurCuda(3, 1.0);
  EXPECT_NO_THROW(img_copy.FreeDeviceMemory());
  EXPECT_NO_THROW(
      img_copy.ImageToDeviceMemory(ImageOperation::GaussianBlur, 3, 1.0));
  img_copy.FreeDeviceMemory();
  fs::remove(kTestImagePath);
}
