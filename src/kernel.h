/**
 * @file kernel.h
 * @brief Заголовочный файл, содержащий определение класса Kernel,
 *        обработку исключений и операции вращения ядра.
 *
 * Этот файл содержит класс `Kernel` для работы с матрицами свертки,
 * их создание, копирование, вращение и обработку ошибок.
 */

#pragma once

#include <cstddef>
#include <exception>
#include <initializer_list>
#include <string>

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
 * @brief Класс для представления ядра оператора.
 *
 * Используется для задания матрицы свертки, применяемой к изображениям.
 */
class Kernel {
 private:
  size_t size_;  ///< Размер ядра
  int** kernel_ = nullptr;  ///< Двумерный массив, представляющий ядро

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
  Kernel(const size_t size, const int** kernel)
      : size_{size} {
    if ((size % 2) == 0u) {
      throw KernelException(
          "Неверный размер ядра. Размер должен быть нечетным");
    }
    kernel_ = new int*[size];
    for (int i = 0; i < size; i++) {
      kernel_[i] = new int[size];
      for (int j = 0; j < size; j++) {
        kernel_[i][j] = kernel[i][j];
      }
    }
  }

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
         const std::initializer_list<std::initializer_list<int>> kernel)
      : size_{size} {
    if ((size % 2) == 0u) {
      throw KernelException(
          "Неверный размер ядра. Размер должен быть нечетным");
    }
    if (kernel.size() != size) {
      throw KernelException(
          "Неверный размер ядра. Размер ядра не совпадает с заданным");
    }
    kernel_ = new int*[size];
    size_t kernel_i = 0, kernel_j = 0;
    for (auto i : kernel) {
      if (i.size() != size) {
        throw KernelException(
            "Неверный размер ядра. Размер ядра не совпадает с заданным");
      }
      kernel_[kernel_i] = new int[size];
      for (auto j : i) {
        kernel_[kernel_i][kernel_j] = j;
        kernel_j++;
      }
      kernel_i++;
    }
  }

  /**
   * @brief Конструктор копирования.
   *
   * Создает новый объект Kernel, копируя содержимое другого объекта.
   *
   * @param other Исходный объект Kernel для копирования.
   */
  Kernel(const Kernel& other)
      : size_(other.size_) {
    kernel_ = new int*[size_];
    for (size_t i = 0; i < size_; i++) {
      kernel_[i] = new int[size_];
      for (size_t j = 0; j < size_; j++) {
        kernel_[i][j] = other.kernel_[i][j];
      }
    }
  }

  /**
   * @brief Деструктор ядра.
   *
   * Освобождает выделенную память.
   */
  ~Kernel() {
    for (int i = 0; i < size_; i++) {
      delete[] kernel_[i];
    }
    delete[] kernel_;
  }

  /**
   * @brief Устанавливает новое ядро.
   *
   * Освобождает предыдущее ядро и выделяет память под новое.
   *
   * @param size Новый размер ядра (должен быть нечетным).
   * @param kernel Двумерный массив, содержащий новые значения ядра.
   * @throws KernelException Если передан четный размер.
   */
  void Set(const size_t size, const int** kernel) {
    if ((size % 2) == 0u) {
      throw KernelException(
          "Неверный размер ядра. Размер должен быть нечетным");
    }
    for (int i = 0; i < size_; i++) {
      delete[] kernel_[i];
    }
    delete[] kernel_;
    size_ = size;
    kernel_ = new int*[size];
    for (int i = 0; i < size; i++) {
      kernel_[i] = new int[size];
      for (int j = 0; j < size; j++) {
        kernel_[i][j] = kernel[i][j];
      }
    }
  }

  /**
   * @brief Оператор присваивания.
   *
   * Освобождает текущие данные и копирует содержимое другого объекта.
   *
   * @param other Исходный объект Kernel для копирования.
   * @return Ссылка на текущий объект.
   */
  Kernel& operator=(const Kernel& other) {
    size_ = other.size_;
    for (int i = 0; i < size_; i++) {
      delete[] kernel_[i];
    }
    delete[] kernel_;
    kernel_ = new int*[size_];
    for (int i = 0; i < size_; i++) {
      kernel_[i] = new int[size_];
      for (int j = 0; j < size_; j++) {
        kernel_[i][j] = other.kernel_[i][j];
      }
    }
    return *this;
  }

  /**
   * @brief Поворачивает ядро на заданный угол.
   *
   * Создает новый объект Kernel, который является повернутой версией текущего
   * ядра.
   *
   * @param degrees Угол поворота (90, 180 или 270 градусов).
   * @return Повернутое ядро.
   */
  Kernel Rotate(const KernelRotationDegrees degrees) const {
    Kernel rotated(*this);
    switch (degrees) {
      case KernelRotationDegrees::DEGREES_90: {
        for (int i = 0; i < size_; i++) {
          for (int j = 0; j < size_; j++) {
            rotated.kernel_[j][size_ - 1 - i] = kernel_[i][j];
          }
        }
        break;
      }
      case KernelRotationDegrees::DEGREES_180: {
        for (int i = 0; i < size_; i++) {
          for (int j = 0; j < size_; j++) {
            rotated.kernel_[size_ - 1 - i][size_ - 1 - j] = kernel_[i][j];
          }
        }
        break;
      }
      case KernelRotationDegrees::DEGREES_270: {
        for (int i = 0; i < size_; i++) {
          for (int j = 0; j < size_; j++) {
            rotated.kernel_[size_ - 1 - j][i] = kernel_[i][j];
          }
        }
        break;
      }
    }
    return rotated;
  }
};

/**
 * @brief Константа. Стандартное ядро свертки Собеля.
 *
 * Используется для выделения границ в изображениях.
 */
const Kernel kKernelSobel(3, {{-1, 0, 1}, {-2, 0, 2}, {-1, 0, 1}});

/**
 * @brief Константа. Стандартное ядро свертки Превитта.
 *
 * Используется для выделения границ в изображениях.
 */
const Kernel kKernelPrewitt(3, {{-1, 0, 1}, {-1, 0, 1}, {-1, 0, 1}});