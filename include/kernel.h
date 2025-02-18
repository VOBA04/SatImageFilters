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
  size_t size_ = 0;  ///< Размер ядра
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
  Kernel(const size_t size, const int** kernel);

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
         const std::initializer_list<std::initializer_list<int>> kernel);

  /**
   * @brief Конструктор копирования.
   *
   * Создает новый объект Kernel, копируя содержимое другого объекта.
   *
   * @param other Исходный объект Kernel для копирования.
   */
  Kernel(const Kernel& other);

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
  void Set(const size_t size, const int** kernel);

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
  int Get(const size_t x, const size_t y) const;
};

/**
 * @brief Константа. Стандартное ядро свертки Собеля.
 *
 * Используется для выделения границ в изображениях.
 *
 * Ядро:
 * \code
 * -1  0  1
 * -2  0  2
 * -1  0  1
 * \endcode
 */
extern const Kernel kKernelSobel;

/**
 * @brief Константа. Стандартное ядро свертки Превитта.
 *
 * Используется для выделения границ в изображениях.
 *
 * Ядро:
 * \code
 * -1  0  1
 * -1  0  1
 * -1  0  1
 * \endcode
 */
extern const Kernel kKernelPrewitt;