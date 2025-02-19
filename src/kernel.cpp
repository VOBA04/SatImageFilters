#include "kernel.h"

const Kernel kKernelSobel(3, {{-1, 0, 1}, {-2, 0, 2}, {-1, 0, 1}});
const Kernel kKernelPrewitt(3, {{-1, 0, 1}, {-1, 0, 1}, {-1, 0, 1}});

Kernel::Kernel()
    : size_(0),
      kernel_(nullptr) {
}

Kernel::Kernel(const size_t size, const int** kernel)
    : size_{size} {
  if ((size % 2) == 0u) {
    throw KernelException("Неверный размер ядра. Размер должен быть нечетным");
  }
  kernel_ = new int*[size];
  for (size_t i = 0; i < size; i++) {
    kernel_[i] = new int[size];
    for (size_t j = 0; j < size; j++) {
      kernel_[i][j] = kernel[i][j];
    }
  }
}

Kernel::Kernel(const size_t size,
               const std::initializer_list<std::initializer_list<int>> kernel)
    : size_{size} {
  if ((size % 2) == 0u) {
    throw KernelException("Неверный размер ядра. Размер должен быть нечетным");
  }
  if (kernel.size() != size) {
    throw KernelException(
        "Неверный размер ядра. Размер ядра не совпадает с заданным");
  }
  kernel_ = new int*[size];
  size_t kernel_i = 0;
  for (auto i : kernel) {
    if (i.size() != size) {
      throw KernelException(
          "Неверный размер ядра. Размер ядра не совпадает с заданным");
    }
    kernel_[kernel_i] = new int[size];
    size_t kernel_j = 0;
    for (auto j : i) {
      kernel_[kernel_i][kernel_j] = j;
      kernel_j++;
    }
    kernel_i++;
  }
}

Kernel::Kernel(const Kernel& other)
    : size_(other.size_) {
  kernel_ = new int*[size_];
  for (size_t i = 0; i < size_; i++) {
    kernel_[i] = new int[size_];
    for (size_t j = 0; j < size_; j++) {
      kernel_[i][j] = other.kernel_[i][j];
    }
  }
}

Kernel::~Kernel() {
  for (size_t i = 0; i < size_; i++) {
    delete[] kernel_[i];
  }
  delete[] kernel_;
}

void Kernel::Set(const size_t size, const int** kernel) {
  if ((size % 2) == 0u) {
    throw KernelException("Неверный размер ядра. Размер должен быть нечетным");
  }
  for (size_t i = 0; i < size_; i++) {
    delete[] kernel_[i];
  }
  delete[] kernel_;
  size_ = size;
  kernel_ = new int*[size];
  for (size_t i = 0; i < size; i++) {
    kernel_[i] = new int[size];
    for (size_t j = 0; j < size; j++) {
      kernel_[i][j] = kernel[i][j];
    }
  }
}

Kernel& Kernel::operator=(const Kernel& other) {
  if (this == &other) {
    return *this;
  }
  size_ = other.size_;
  for (size_t i = 0; i < size_; i++) {
    delete[] kernel_[i];
  }
  delete[] kernel_;
  kernel_ = new int*[size_];
  for (size_t i = 0; i < size_; i++) {
    kernel_[i] = new int[size_];
    for (size_t j = 0; j < size_; j++) {
      kernel_[i][j] = other.kernel_[i][j];
    }
  }
  return *this;
}

Kernel Kernel::Rotate(const KernelRotationDegrees degrees) const {
  Kernel rotated(*this);
  switch (degrees) {
    case KernelRotationDegrees::DEGREES_90: {
      for (size_t i = 0; i < size_; i++) {
        for (size_t j = 0; j < size_; j++) {
          rotated.kernel_[j][size_ - 1 - i] = kernel_[i][j];
        }
      }
      break;
    }
    case KernelRotationDegrees::DEGREES_180: {
      for (size_t i = 0; i < size_; i++) {
        for (size_t j = 0; j < size_; j++) {
          rotated.kernel_[size_ - 1 - i][size_ - 1 - j] = kernel_[i][j];
        }
      }
      break;
    }
    case KernelRotationDegrees::DEGREES_270: {
      for (size_t i = 0; i < size_; i++) {
        for (size_t j = 0; j < size_; j++) {
          rotated.kernel_[size_ - 1 - j][i] = kernel_[i][j];
        }
      }
      break;
    }
  }
  return rotated;
}

size_t Kernel::GetSize() const {
  return size_;
}

int Kernel::Get(const size_t x, const size_t y) const {
  return kernel_[x][y];
}