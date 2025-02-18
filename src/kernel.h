#pragma once

#include <cstddef>
#include <exception>
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
 * @brief Класс для представления ядра оператора.
 *
 * Используется для задания матрицы свертки, применяемой к изображениям.
 */
class Kernel {
 private:
  size_t width_;   ///< Ширина ядра
  size_t height_;  ///< Высота ядра
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
      : width_{size},
        height_{size} {
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
   * @brief Деструктор ядра.
   *
   * Освобождает выделенную память.
   */
  ~Kernel() {
    for (int i = 0; i < height_; i++) {
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

    for (int i = 0; i < height_; i++) {
      delete[] kernel_[i];
    }
    delete[] kernel_;

    width_ = size;
    height_ = size;
    kernel_ = new int*[size];
    for (int i = 0; i < size; i++) {
      kernel_[i] = new int[size];
      for (int j = 0; j < size; j++) {
        kernel_[i][j] = kernel[i][j];
      }
    }
  }
};
