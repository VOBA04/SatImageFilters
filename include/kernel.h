/**
 * @file kernel.h
 * @brief Заголовочный файл, содержащий определение шаблонного класса Kernel,
 *        обработку исключений и операции вращения ядра.
 *
 * Этот файл содержит шаблонный класс `Kernel` для работы с матрицами свертки,
 * их создание, копирование, вращение и обработку ошибок.
 */

#pragma once

#include <cstddef>
#include <exception>
#include <initializer_list>
#include <string>
#include <cmath>

/**
 * @brief Класс исключений, связанных с @ref Kernel "Kernel".
 *
 */
class KernelException : public std::exception {
 private:
  std::string message_;  ///< Сообщение об ошибке

 public:
  /**
   * @brief Конструктор исключения KernelException.
   *
   * @param message Сообщение, описывающее ошибку.
   */
  explicit KernelException(const std::string& message)
      : message_{message} {
  }

  /**
   * @brief Получить сообщение об ошибке.
   *
   * @return Указатель на строку с описанием ошибки.
   */
  const char* what() const noexcept override {
    return message_.c_str();
  }
};

/**
 * @brief Перечисление углов поворота ядра.
 *
 * Используется в методе @ref Kernel::Rotate для указания угла вращения.
 */
enum class KernelRotationDegrees {
  DEGREES_90,  ///< Поворот на 90 градусов по часовой стрелке
  DEGREES_180,  ///< Поворот на 180 градусов
  DEGREES_270  ///< Поворот на 270 градусов по часовой стрелке
};

/**
 * @brief Шаблонный класс для представления ядра оператора.
 *
 * Используется для задания матрицы свертки, применяемой к изображениям.
 *
 * @tparam T Тип данных ядра (например, int, float).
 */
template <typename T>
class Kernel {
 private:
  size_t size_ = 0;  ///< Размер ядра
  T** kernel_ = nullptr;  ///< Двумерный массив, представляющий ядро

 public:
  /**
   * @brief Конструктор по умолчанию.
   *
   * Создает пустое ядро.
   */
  Kernel();

  /**
   * @brief Конструктор с параметрами.
   *
   * Создает квадратное ядро размером size x size и инициализирует его
   * значениями из переданного массива.
   *
   * @param size Размер ядра (должен быть нечетным).
   * @param kernel Двумерный массив, содержащий значения ядра.
   * @throws KernelException Если передан четный размер.
   */
  Kernel(const size_t size, const T** kernel);

  /**
   * @brief Конструктор с инициализатором списка.
   *
   * Позволяет создавать ядро, используя список инициализации.
   *
   * @param size Размер ядра (должен быть нечетным).
   * @param kernel Список инициализации с значениями ядра.
   * @throws KernelException Если размер ядра четный или не совпадает с
   * заданным.
   */
  Kernel(const size_t size,
         const std::initializer_list<std::initializer_list<T>> kernel);

  /**
   * @brief Конструктор копирования.
   *
   * Создает новый объект Kernel, копируя содержимое другого объекта.
   *
   * @param other Исходный объект Kernel для копирования.
   */
  Kernel(const Kernel& other);

  explicit Kernel(const size_t size);

  /**
   * @brief Деструктор ядра.
   *
   * Освобождает выделенную память.
   */
  ~Kernel();

  /**
   * @brief Устанавливает новое ядро.
   *
   * Освобождает предыдущее ядро и выделяет память под новое.
   *
   * @param size Новый размер ядра (должен быть нечетным).
   * @param kernel Двумерный массив, содержащий новые значения ядра.
   * @throws KernelException Если передан четный размер.
   */
  void Set(const size_t size, const T** kernel);

  /**
   * @brief Оператор присваивания.
   *
   * Освобождает текущие данные и копирует содержимое другого объекта.
   *
   * @param other Исходный объект Kernel для копирования.
   * @return Ссылка на текущий объект.
   */
  Kernel& operator=(const Kernel& other);

  /**
   * @brief Поворачивает ядро на заданный угол.
   *
   * Создает новый объект Kernel, который является повернутой версией текущего
   * ядра.
   *
   * @param degrees Угол поворота (90, 180 или 270 градусов).
   * @return Повернутое ядро.
   */
  Kernel Rotate(const KernelRotationDegrees degrees) const;

  /**
   * @brief Возвращает размер ядра.
   *
   * @return Размер ядра.
   */
  size_t GetSize() const;

  /**
   * @brief Возвращает значение ядра в заданной позиции.
   *
   * @param x Координата x.
   * @param y Координата y.
   * @return Значение ядра в позиции (x, y).
   */
  T Get(const size_t x, const size_t y) const;

  static Kernel GetGaussianKernel(const size_t size);
};

template <typename T>
Kernel<T>::Kernel()
    : size_(0),
      kernel_(nullptr) {
}

template <typename T>
Kernel<T>::Kernel(const size_t size, const T** kernel)
    : size_{size} {
  if ((size % 2) == 0u) {
    throw KernelException("Неверный размер ядра. Размер должен быть нечетным");
  }
  kernel_ = new T*[size];
  for (size_t i = 0; i < size; i++) {
    kernel_[i] = new T[size];
    for (size_t j = 0; j < size; j++) {
      kernel_[i][j] = kernel[i][j];
    }
  }
}

template <typename T>
Kernel<T>::Kernel(const size_t size,
                  const std::initializer_list<std::initializer_list<T>> kernel)
    : size_{size} {
  if ((size % 2) == 0u) {
    throw KernelException("Неверный размер ядра. Размер должен быть нечетным");
  }
  if (kernel.size() != size) {
    throw KernelException(
        "Неверный размер ядра. Размер ядра не совпадает с заданным");
  }
  kernel_ = new T*[size];
  size_t kernel_i = 0;
  for (auto i : kernel) {
    if (i.size() != size) {
      throw KernelException(
          "Неверный размер ядра. Размер ядра не совпадает с заданным");
    }
    kernel_[kernel_i] = new T[size];
    size_t kernel_j = 0;
    for (auto j : i) {
      kernel_[kernel_i][kernel_j] = j;
      kernel_j++;
    }
    kernel_i++;
  }
}

template <typename T>
Kernel<T>::Kernel(const Kernel& other)
    : size_(other.size_) {
  kernel_ = new T*[size_];
  for (size_t i = 0; i < size_; i++) {
    kernel_[i] = new T[size_];
    for (size_t j = 0; j < size_; j++) {
      kernel_[i][j] = other.kernel_[i][j];
    }
  }
}

template <typename T>
Kernel<T>::Kernel(const size_t size)
    : size_{size} {
  if ((size % 2) == 0u) {
    throw KernelException("Неверный размер ядра. Размер должен быть нечетным");
  }
  kernel_ = new T*[size];
  for (size_t i = 0; i < size; i++) {
    kernel_[i] = new T[size];
    for (size_t j = 0; j < size; j++) {
      kernel_[i][j] = 0;
    }
  }
}

template <typename T>
Kernel<T>::~Kernel() {
  for (size_t i = 0; i < size_; i++) {
    delete[] kernel_[i];
  }
  delete[] kernel_;
}

template <typename T>
void Kernel<T>::Set(const size_t size, const T** kernel) {
  if ((size % 2) == 0u) {
    throw KernelException("Неверный размер ядра. Размер должен быть нечетным");
  }
  for (size_t i = 0; i < size_; i++) {
    delete[] kernel_[i];
  }
  delete[] kernel_;
  size_ = size;
  kernel_ = new T*[size];
  for (size_t i = 0; i < size; i++) {
    kernel_[i] = new T[size];
    for (size_t j = 0; j < size; j++) {
      kernel_[i][j] = kernel[i][j];
    }
  }
}

template <typename T>
Kernel<T>& Kernel<T>::operator=(const Kernel& other) {
  if (this == &other) {
    return *this;
  }
  size_ = other.size_;
  for (size_t i = 0; i < size_; i++) {
    delete[] kernel_[i];
  }
  delete[] kernel_;
  kernel_ = new T*[size_];
  for (size_t i = 0; i < size_; i++) {
    kernel_[i] = new T[size_];
    for (size_t j = 0; j < size_; j++) {
      kernel_[i][j] = other.kernel_[i][j];
    }
  }
  return *this;
}

template <typename T>
Kernel<T> Kernel<T>::Rotate(const KernelRotationDegrees degrees) const {
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

template <typename T>
size_t Kernel<T>::GetSize() const {
  return size_;
}

template <typename T>
T Kernel<T>::Get(const size_t x, const size_t y) const {
  return kernel_[x][y];
}

template <typename T>
Kernel<T> Kernel<T>::GetGaussianKernel(const size_t size) {
  if ((size % 2) == 0u) {
    throw KernelException("Неверный размер ядра. Размер должен быть нечетным");
  }
  Kernel<T> gaussian_kernel(size);
  int half = size / 2;
  T sum = 0.0;
  T sigma = size / 6.0;
  for (int i = -half; i <= half; i++) {
    for (int j = -half; j <= half; j++) {
      T g = exp(-(i * i + j * j) / (2 * sigma * sigma)) /
            (2 * M_PI * sigma * sigma);
      gaussian_kernel.kernel_[i + half][j + half] = g;
      sum += g;
    }
  }
  for (int i = 0; i < size; i++) {
    for (int j = 0; j < size; j++) {
      gaussian_kernel.kernel_[i][j] /= sum;
    }
  }
  return gaussian_kernel;
}

const Kernel<int> kKernelSobel(3, {{-1, 0, 1}, {-2, 0, 2}, {-1, 0, 1}});
const Kernel<int> kKernelPrewitt(3, {{-1, 0, 1}, {-1, 0, 1}, {-1, 0, 1}});