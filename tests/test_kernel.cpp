#include <gtest/gtest.h>
#include <cstddef>
#include "kernel.h"
#include <filesystem>

namespace fs = std::filesystem;

TEST(KernelTest, KernelException) {
  KernelException e("Test exception");
  EXPECT_STREQ(e.what(), "Test exception");
}

TEST(KernelTest, KernelConstructor) {
  Kernel<int> kernel(3, 3, true);
  EXPECT_EQ(kernel.GetHeight(), 3);
  EXPECT_EQ(kernel.GetWidth(), 3);
  EXPECT_TRUE(kernel.IsRotatable());
  int** kernel_data = nullptr;
  EXPECT_THROW(Kernel<int> kernel_invalid(2, 2, true), KernelException);
  EXPECT_THROW(Kernel<int> kernel_invalid(3, 3, kernel_data, false),
               KernelException);
  int** kernel_data2 = new int*[3]{nullptr, nullptr, nullptr};
  EXPECT_THROW(Kernel<int> kernel_invalid(3, 3, kernel_data2, false),
               KernelException);
  int** kernel_data3 = new int*[3]{
      new int[3]{1, 2, 3},
      new int[3]{4, 5, 6},
      new int[3]{7, 8, 9},
  };
  EXPECT_NO_THROW(Kernel<int> kernel_valid(3, 3, kernel_data3, false));
  int* kernel_row = nullptr;
  EXPECT_THROW(Kernel<int> kernel_invalid(2, 2, kernel_row, false),
               KernelException);
  EXPECT_THROW(Kernel<int> kernel_invalid(3, 3, kernel_row, false),
               KernelException);
  int* kernel_row2 = new int[3]{1, 2, 3};
  EXPECT_NO_THROW(Kernel<int> kernel_invalid(3, 3, kernel_row2, false));
  EXPECT_THROW(Kernel<int> kernel_invalid(2, 2, {{}}, false), KernelException);
  EXPECT_THROW(Kernel<int> kernel_invalid(3, 3, {{}}, false), KernelException);
  EXPECT_THROW(Kernel<int> kernel_invalid(3, 3, {{}, {}, {}}, false),
               KernelException);
  delete[] kernel_row2;
  delete[] kernel_data2;
  delete[] kernel_data3[0];
  delete[] kernel_data3[1];
  delete[] kernel_data3[2];
  delete[] kernel_data3;
}

TEST(KernelTest, KernelCopyConstructor) {
  Kernel<int> kernel(3, 3, true);
  Kernel<int> kernel_copy(kernel);
  EXPECT_EQ(kernel.GetHeight(), kernel_copy.GetHeight());
  EXPECT_EQ(kernel.GetWidth(), kernel_copy.GetWidth());
  EXPECT_TRUE(kernel.IsRotatable());
  EXPECT_TRUE(kernel == kernel_copy);
}

TEST(KernelTest, KernelCopyAssignment) {
  Kernel<int> kernel(3, 3, true);
  Kernel<int> kernel_copy;
  kernel_copy = kernel;
  EXPECT_EQ(kernel.GetHeight(), kernel_copy.GetHeight());
  EXPECT_EQ(kernel.GetWidth(), kernel_copy.GetWidth());
  EXPECT_TRUE(kernel.IsRotatable());
  EXPECT_TRUE(kernel == kernel_copy);
}

TEST(KernelTest, KernelRotate) {
  Kernel<int> kernel(1, 3, {{1, 2, 3}}, true);
  Kernel<int> rotated_kernel = kernel.Rotate(KernelRotationDegrees::DEGREES_90);
  EXPECT_EQ(rotated_kernel.GetHeight(), kernel.GetWidth());
  EXPECT_EQ(rotated_kernel.GetWidth(), kernel.GetHeight());
  EXPECT_TRUE(rotated_kernel.IsRotatable());
  EXPECT_TRUE(kernel != rotated_kernel);
  for (size_t x = 0; x < kernel.GetHeight(); x++) {
    for (size_t y = 0; y < kernel.GetWidth(); y++) {
      EXPECT_EQ(rotated_kernel.Get(x, y), kernel.Get(y, x));
    }
  }
  rotated_kernel = kernel.Rotate(KernelRotationDegrees::DEGREES_180);
  EXPECT_EQ(rotated_kernel.GetHeight(), kernel.GetHeight());
  EXPECT_EQ(rotated_kernel.GetWidth(), kernel.GetWidth());
  EXPECT_TRUE(rotated_kernel.IsRotatable());
  EXPECT_TRUE(kernel != rotated_kernel);
  for (size_t y = 0; y < kernel.GetHeight(); y++) {
    for (size_t x = 0; x < kernel.GetWidth(); x++) {
      EXPECT_EQ(
          rotated_kernel.Get(x, y),
          kernel.Get(kernel.GetWidth() - 1 - x, kernel.GetHeight() - 1 - y));
    }
  }
  rotated_kernel = kernel.Rotate(KernelRotationDegrees::DEGREES_270);
  EXPECT_EQ(rotated_kernel.GetHeight(), kernel.GetWidth());
  EXPECT_EQ(rotated_kernel.GetWidth(), kernel.GetHeight());
  EXPECT_TRUE(rotated_kernel.IsRotatable());
  EXPECT_TRUE(kernel != rotated_kernel);
  for (size_t x = 0; x < kernel.GetHeight(); x++) {
    for (size_t y = 0; y < kernel.GetWidth(); y++) {
      EXPECT_EQ(rotated_kernel.Get(x, y),
                kernel.Get(kernel.GetWidth() - 1 - y, x));
    }
  }
}

TEST(KernelTest, KernelGet) {
  Kernel<int> kernel(3, 3, {{1, 2, 3}, {4, 5, 6}, {7, 8, 9}}, true);
  EXPECT_EQ(kernel.Get(0, 0), 1);
  EXPECT_EQ(kernel.Get(1, 1), 5);
  EXPECT_EQ(kernel.Get(2, 2), 9);
}

TEST(KernelTest, KernelSet) {
  Kernel<int> kernel(3, 3, {{1, 2, 3}, {4, 5, 6}, {7, 8, 9}}, true);
  int* new_kernel1 = nullptr;
  EXPECT_THROW(kernel.Set(2, 2, new_kernel1, true), KernelException);
  EXPECT_THROW(kernel.Set(3, 3, new_kernel1, true), KernelException);
  new_kernel1 = new int[3]{1, 2, 3};
  EXPECT_NO_THROW(kernel.Set(1, 3, new_kernel1, true));
  EXPECT_EQ(kernel.GetHeight(), 1);
  EXPECT_EQ(kernel.GetWidth(), 3);
  int** new_kernel2 = nullptr;
  EXPECT_THROW(kernel.Set(2, 2, new_kernel2, true), KernelException);
  EXPECT_THROW(kernel.Set(3, 3, new_kernel2, true), KernelException);
  new_kernel2 = new int*[3];
  new_kernel2[0] = new int[3]{1, 2, 3};
  new_kernel2[1] = new int[3]{4, 5, 6};
  new_kernel2[2] = new int[3]{7, 8, 9};
  EXPECT_NO_THROW(kernel.Set(3, 3, new_kernel2, true));
  EXPECT_EQ(kernel.GetHeight(), 3);
  EXPECT_EQ(kernel.GetWidth(), 3);
}

TEST(KernelTest, KernelCopyKernelTo) {
  Kernel<int> kernel(3, 3, {{1, 2, 3}, {4, 5, 6}, {7, 8, 9}}, true);
  int* kernel_data = nullptr;
  kernel.CopyKernelTo(&kernel_data);
  EXPECT_EQ(kernel_data[0], 1);
  EXPECT_EQ(kernel_data[1], 2);
  EXPECT_EQ(kernel_data[2], 3);
  EXPECT_EQ(kernel_data[3], 4);
  EXPECT_EQ(kernel_data[4], 5);
  EXPECT_EQ(kernel_data[5], 6);
  EXPECT_EQ(kernel_data[6], 7);
  EXPECT_EQ(kernel_data[7], 8);
  EXPECT_EQ(kernel_data[8], 9);
  delete[] kernel_data;
}

TEST(KernelTest, KernelDestructor) {
  Kernel<int> kernel(3, 3, {{1, 2, 3}, {4, 5, 6}, {7, 8, 9}}, true);
  EXPECT_NO_THROW(kernel.~Kernel());
}

TEST(KernelTest, KernelEquality) {
  Kernel<int> kernel1(3, 3, {{1, 2, 3}, {4, 5, 6}, {7, 8, 9}}, true);
  Kernel<int> kernel2(3, 3, {{1, 2, 3}, {4, 5, 6}, {7, 8, 9}}, true);
  EXPECT_TRUE(kernel1 == kernel2);
  EXPECT_FALSE(kernel1 != kernel2);
  Kernel<int> kernel3(1, 3, {{1, 2, 3}}, true);
  EXPECT_FALSE(kernel1 == kernel3);
  EXPECT_TRUE(kernel1 != kernel3);
  Kernel<int> kernel4(3, 3, {{1, 2, 3}, {4, 5, 4}, {3, 2, 1}}, true);
  EXPECT_FALSE(kernel1 == kernel4);
  EXPECT_TRUE(kernel1 != kernel4);
}

TEST(KernelTest, KernelGaussianKernel) {
  Kernel<float> kernel = Kernel<float>::GetGaussianKernel(3, 1.0);
  EXPECT_EQ(kernel.GetHeight(), 3);
  EXPECT_EQ(kernel.GetWidth(), 3);
  EXPECT_FALSE(kernel.IsRotatable());
  EXPECT_NEAR(kernel.Get(0, 0), 0.075113, 5e-5);
  EXPECT_NEAR(kernel.Get(1, 1), 0.204173, 5e-5);
  EXPECT_NEAR(kernel.Get(2, 2), 0.075113, 5e-5);
}

TEST(KernelTest, KernelGaussianKernelSep) {
  Kernel<float> kernel = Kernel<float>::GetGaussianKernelSep(3, 1.0);
  EXPECT_EQ(kernel.GetHeight(), 3);
  EXPECT_EQ(kernel.GetWidth(), 1);
  EXPECT_TRUE(kernel.IsRotatable());
  EXPECT_NEAR(kernel.Get(0, 0), 0.274079, 5e-5);
  EXPECT_NEAR(kernel.Get(1, 0), 0.451842, 5e-5);
  EXPECT_NEAR(kernel.Get(2, 0), 0.274079, 5e-5);
}

TEST(KernelTest, KernelGaussianKernelException) {
  EXPECT_THROW(Kernel<int>::GetGaussianKernel(2, 1.0), KernelException);
  EXPECT_THROW(Kernel<int>::GetGaussianKernelSep(2, 1.0), KernelException);
}

TEST(KernelTest, SetFromFile) {
  fs::path files_path = fs::path(PROJECT_SOURCE_DIR) / "tests";
  Kernel<int> kernel;
  EXPECT_THROW(
      kernel.SetFromFile((files_path / "non_existent_file.txt").string()),
      KernelException);
  EXPECT_NO_THROW(
      kernel.SetFromFile((files_path / "test_kernel1.txt").string()));
  EXPECT_EQ(kernel.GetHeight(), 3);
  EXPECT_EQ(kernel.GetWidth(), 3);
  EXPECT_FALSE(kernel.IsRotatable());
  EXPECT_EQ(kernel.Get(0, 0), 0);
  EXPECT_EQ(kernel.Get(1, 1), 1);
  EXPECT_THROW(kernel.SetFromFile((files_path / "test_kernel2.txt").string()),
               KernelException);
  EXPECT_THROW(kernel.SetFromFile((files_path / "test_kernel3.txt").string()),
               KernelException);
  EXPECT_THROW(kernel.SetFromFile((files_path / "test_kernel4.txt").string()),
               KernelException);
}