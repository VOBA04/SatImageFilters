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
#include <cstring>
#include <exception>
#include <initializer_list>
#include <string>
#include <cmath>
#include <fstream>

/**
 * @brief Класс исключений, связанных с @ref Kernel "Kernel".
 *
 * Используется для обработки ошибок, возникающих при работе с ядрами.
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
  DEGREES_90,   ///< Поворот на 90 градусов по часовой стрелке
  DEGREES_180,  ///< Поворот на 180 градусов
  DEGREES_270   ///< Поворот на 270 градусов по часовой стрелке
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
  size_t height_ = 0;       ///< Высота ядра
  size_t width_ = 0;        ///< Ширина ядра
  T* kernel_ = nullptr;     ///< Одномерный массив, представляющий ядро
  bool rotatable_ = false;  ///< Флаг, указывающий, можно ли вращать ядро

 public:
  /**
   * @brief Конструктор по умолчанию.
   *
   * Создает пустое ядро с высотой и шириной, равными 0.
   */
  Kernel();

  /**
   * @brief Конструктор с параметрами.
   *
   * Создает ядро с заданной высотой и шириной и инициализирует его значениями
   * из переданного двумерного массива.
   *
   * @param height Высота ядра (должна быть нечетной).
   * @param width Ширина ядра (должна быть нечетной).
   * @param kernel Двумерный массив, содержащий значения ядра.
   * @param rotatable Флаг, указывающий, можно ли вращать ядро.
   * @throws KernelException Если высота или ширина четные, либо передан
   * нулевой указатель.
   */
  Kernel(const size_t height, const size_t width, T** kernel, bool rotatable);

  /**
   * @brief Конструктор с параметрами.
   *
   * Создает ядро с заданной высотой и шириной и инициализирует его значениями
   * из переданного одномерного массива.
   *
   * @param height Высота ядра (должна быть нечетной).
   * @param width Ширина ядра (должна быть нечетной).
   * @param kernel Одномерный массив, содержащий значения ядра.
   * @param rotatable Флаг, указывающий, можно ли вращать ядро.
   * @throws KernelException Если высота или ширина четные, либо передан
   * нулевой указатель.
   */
  Kernel(const size_t height, const size_t width, T* kernel, bool rotatable);

  /**
   * @brief Конструктор с инициализатором списка.
   *
   * Позволяет создавать ядро, используя список инициализации.
   *
   * @param height Высота ядра (должна быть нечетной).
   * @param width Ширина ядра (должна быть нечетной).
   * @param kernel Список инициализации с значениями ядра.
   * @param rotatable Флаг, указывающий, можно ли вращать ядро.
   * @throws KernelException Если высота или ширина четные, либо размеры
   * списка не совпадают с заданными.
   */
  Kernel(const size_t height, const size_t width,
         const std::initializer_list<std::initializer_list<T>> kernel,
         bool rotatable);

  /**
   * @brief Конструктор копирования.
   *
   * Создает новый объект Kernel, копируя содержимое другого объекта.
   *
   * @param other Исходный объект Kernel для копирования.
   */
  Kernel(const Kernel& other);

  /**
   * @brief Конструктор с параметрами высоты и ширины.
   *
   * Создает ядро с заданной высотой и шириной и инициализирует его нулями.
   *
   * @param height Высота ядра (должна быть нечетной).
   * @param width Ширина ядра (должна быть нечетной).
   * @param rotatable Флаг, указывающий, можно ли вращать ядро.
   * @throws KernelException Если высота или ширина четные.
   */
  explicit Kernel(const size_t height, const size_t width, bool rotatable);

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
   * @param height Новая высота ядра (должна быть нечетной).
   * @param width Новая ширина ядра (должна быть нечетной).
   * @param kernel Двумерный массив, содержащий новые значения ядра.
   * @param rotatable Флаг, указывающий, можно ли вращать ядро.
   * @throws KernelException Если высота или ширина четные, либо передан
   * нулевой указатель.
   */
  void Set(const size_t height, const size_t width, T** kernel, bool rotatable);

  /**
   * @brief Устанавливает новое ядро.
   *
   * Освобождает предыдущее ядро и выделяет память под новое.
   *
   * @param height Новая высота ядра (должна быть нечетной).
   * @param width Новая ширина ядра (должна быть нечетной).
   * @param kernel Одномерный массив, содержащий новые значения ядра.
   * @param rotatable Флаг, указывающий, можно ли вращать ядро.
   * @throws KernelException Если высота или ширина четные, либо передан
   * нулевой указатель.
   */
  void Set(const size_t height, const size_t width, T* kernel, bool rotatable);

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
   * @brief Оператор сравнения.
   *
   * Сравнивает два объекта Kernel на равенство.
   *
   * @param other Объект Kernel для сравнения.
   * @return true, если объекты равны; false в противном случае.
   */
  bool operator==(const Kernel& other) const;

  /**
   * @brief Оператор неравенства.
   *
   * Сравнивает два объекта Kernel на неравенство.
   *
   * @param other Объект Kernel для сравнения.
   * @return true, если объекты не равны; false в противном случае.
   *
   * Операторы сравнения проверяют размеры ядра (высоту и ширину), а также
   * значения всех элементов ядра. Если хотя бы одно из этих условий не
   * выполняется, объекты считаются неравными.
   */
  bool operator!=(const Kernel& other) const;

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
   * @brief Возвращает высоту ядра.
   *
   * @return Высота ядра.
   */
  size_t GetHeight() const;

  /**
   * @brief Возвращает ширину ядра.
   *
   * @return Ширина ядра.
   */
  size_t GetWidth() const;

  /**
   * @brief Проверяет, нужно ли вращать ядро.
   *
   * @return true, если ядро можно вращать; false в противном случае.
   */
  bool IsRotatable() const;

  /**
   * @brief Возвращает значение ядра в заданной позиции.
   *
   * @param x Координата x.
   * @param y Координата y.
   * @return Значение ядра в позиции (x, y).
   */
  T Get(const size_t x, const size_t y) const;

  /**
   * @brief Копирует ядро в указанный буфер памяти.
   *
   * Выделяет новую память и копирует в неё содержимое ядра.
   * Ответственность за освобождение памяти лежит на вызывающей стороне.
   *
   * @param dst Указатель на указатель, который будет установлен на новую
   * область памяти.
   */
  void CopyKernelTo(T** dst) const;

  /**
   * @brief Возвращает ядро фильтра Гаусса.
   *
   * Создает и возвращает ядро фильтра Гаусса заданного размера.
   *
   * Формула для вычисления значений ядра:
   * \f[
   * G(x, y) = e^{-\frac{x^2 + y^2}{2 \sigma^2}}
   * \f]
   *
   * Если \f$\sigma\f$ не задан, он вычисляется как \f$\sigma =
   * \frac{size}{6}\f$.
   *
   * @param size Размер ядра (должен быть нечетным).
   * @param sigma Стандартное отклонение (опционально).
   * @return Ядро фильтра Гаусса.
   * @throws KernelException Если передан четный размер.
   */
  static Kernel GetGaussianKernel(const size_t size, float sigma = 0.0);

  /**
   * @brief Возвращает одномерное ядро фильтра Гаусса.
   *
   * Создает и возвращает одномерное ядро фильтра Гаусса заданного размера.
   * Это ядро может быть использовано для свертки по одной из осей
   * (горизонтальной или вертикальной).
   *
   * Формула для вычисления значений ядра:
   * \f[
   * G(x) = e^{-\frac{x^2}{2 \sigma^2}}
   * \f]
   *
   * Если \f$\sigma\f$ не задан, он вычисляется как \f$\sigma =
   * \frac{size}{6}\f$.
   *
   * @param size Размер ядра (должен быть нечетным).
   * @param sigma Стандартное отклонение (опционально).
   * @return Одномерное ядро фильтра Гаусса.
   * @throws KernelException Если передан четный размер.
   */
  static Kernel GetGaussianKernelSep(const size_t size, float sigma = 0.0);

  /**
   * @brief Устанавливает ядро из файла.
   *
   * Читает высоту, ширину, флаг вращения и значения ядра из файла и
   * устанавливает их.
   *
   * @param filename Имя файла, содержащего параметры ядра.
   * @throws KernelException Если не удалось открыть файл или произошла ошибка
   * чтения.
   */
  void SetFromFile(const std::string& filename);
};

template <typename T>
Kernel<T>::Kernel()
    : height_(0),
      width_(0),
      kernel_(nullptr),
      rotatable_(false) {
}

template <typename T>
Kernel<T>::Kernel(const size_t height, const size_t width, T** kernel,
                  bool rotatable)
    : height_{height},
      width_{width},
      rotatable_(rotatable) {
  if ((height % 2) == 0u || (width % 2) == 0u) {
    throw KernelException(
        "Неверный размер ядра. Высота и ширина должны быть нечетными");
  }
  if (kernel == nullptr) {
    throw KernelException("Передан нулевой указатель на ядро");
  }
  for (size_t i = 0; i < height; i++) {
    for (size_t j = 0; j < width; j++) {
      if (kernel[i] == nullptr) {
        throw KernelException("Передан нулевой указатель на строку ядра");
      }
    }
  }
  kernel_ = new T[height * width];
  for (size_t i = 0; i < height; i++) {
    for (size_t j = 0; j < width; j++) {
      kernel_[i * width + j] = kernel[i][j];
    }
  }
}

template <typename T>
Kernel<T>::Kernel(const size_t height, const size_t width, T* kernel,
                  bool rotatable)
    : height_{height},
      width_{width},
      rotatable_(rotatable) {
  if ((height % 2) == 0u || (width % 2) == 0u) {
    throw KernelException(
        "Неверный размер ядра. Высота и ширина должны быть нечетными");
  }
  if (kernel == nullptr) {
    throw KernelException("Передан нулевой указатель на ядро");
  }
  kernel_ = new T[height * width];
  std::memcpy(kernel_, kernel, height * width * sizeof(T));
}

template <typename T>
Kernel<T>::Kernel(const size_t height, const size_t width,
                  const std::initializer_list<std::initializer_list<T>> kernel,
                  bool rotatable)
    : height_{height},
      width_{width},
      rotatable_(rotatable) {
  if ((height % 2) == 0u || (width % 2) == 0u) {
    throw KernelException(
        "Неверный размер ядра. Высота и ширина должны быть нечетными");
  }
  if (kernel.size() != height) {
    throw KernelException(
        "Неверный размер ядра. Высота ядра не совпадает с заданной");
  }
  kernel_ = new T[height * width];
  size_t i = 0;
  for (const auto& row : kernel) {
    if (row.size() != width) {
      throw KernelException(
          "Неверный размер ядра. Ширина ядра не совпадает с заданной");
    }
    size_t j = 0;
    for (const auto& value : row) {
      kernel_[i * width + j] = value;
      j++;
    }
    i++;
  }
}

template <typename T>
Kernel<T>::Kernel(const Kernel& other)
    : height_(other.height_),
      width_(other.width_),
      rotatable_(other.rotatable_) {
  kernel_ = new T[height_ * width_];
  std::memcpy(kernel_, other.kernel_, height_ * width_ * sizeof(T));
}

template <typename T>
Kernel<T>::Kernel(const size_t height, const size_t width, bool rotatable)
    : height_{height},
      width_{width},
      rotatable_(rotatable) {
  if ((height % 2) == 0u || (width % 2) == 0u) {
    throw KernelException(
        "Неверный размер ядра. Высота и ширина должны быть нечетными");
  }
  kernel_ = new T[height * width]();
}

template <typename T>
Kernel<T>::~Kernel() {
  if (kernel_ != nullptr) {
    delete[] kernel_;
    kernel_ = nullptr;
  }
}

template <typename T>
void Kernel<T>::Set(const size_t height, const size_t width, T** kernel,
                    bool rotatable) {
  if ((height % 2) == 0u || (width % 2) == 0u) {
    throw KernelException(
        "Неверный размер ядра. Высота и ширина должны быть нечетными");
  }
  if (kernel == nullptr) {
    throw KernelException("Передан нулевой указатель на ядро");
  }
  delete[] kernel_;
  height_ = height;
  width_ = width;
  rotatable_ = rotatable;
  kernel_ = new T[height * width];
  for (size_t i = 0; i < height; i++) {
    for (size_t j = 0; j < width; j++) {
      if (kernel[i] == nullptr) {
        throw KernelException("Передан нулевой указатель на строку ядра");
      }
      kernel_[i * width + j] = kernel[i][j];
    }
  }
}

template <typename T>
void Kernel<T>::Set(const size_t height, const size_t width, T* kernel,
                    bool rotatable) {
  if ((height % 2) == 0u || (width % 2) == 0u) {
    throw KernelException(
        "Неверный размер ядра. Высота и ширина должны быть нечетными");
  }
  if (kernel == nullptr) {
    throw KernelException("Передан нулевой указатель на ядро");
  }
  delete[] kernel_;
  height_ = height;
  width_ = width;
  rotatable_ = rotatable;
  kernel_ = new T[height * width];
  std::memcpy(kernel_, kernel, height * width * sizeof(T));
}

template <typename T>
Kernel<T>& Kernel<T>::operator=(const Kernel& other) {
  if (this == &other) {
    return *this;
  }
  delete[] kernel_;
  height_ = other.height_;
  width_ = other.width_;
  rotatable_ = other.rotatable_;
  kernel_ = new T[height_ * width_];
  std::memcpy(kernel_, other.kernel_, height_ * width_ * sizeof(T));
  return *this;
}

template <typename T>
bool Kernel<T>::operator==(const Kernel& other) const {
  if (height_ != other.height_ || width_ != other.width_) {
    return false;
  }
  for (size_t i = 0; i < height_ * width_; i++) {
    if (kernel_[i] != other.kernel_[i]) {
      return false;
    }
  }
  return true;
}

template <typename T>
bool Kernel<T>::operator!=(const Kernel& other) const {
  return !(*this == other);
}

template <typename T>
Kernel<T> Kernel<T>::Rotate(const KernelRotationDegrees degrees) const {
  Kernel rotated(height_, width_, rotatable_);
  switch (degrees) {
    case KernelRotationDegrees::DEGREES_90: {
      rotated.height_ = width_;
      rotated.width_ = height_;
      for (size_t i = 0; i < height_; i++) {
        for (size_t j = 0; j < width_; j++) {
          rotated.kernel_[j * rotated.width_ + (rotated.width_ - 1 - i)] =
              kernel_[i * width_ + j];
        }
      }
      break;
    }
    case KernelRotationDegrees::DEGREES_180: {
      rotated.height_ = height_;
      rotated.width_ = width_;
      for (size_t i = 0; i < height_; i++) {
        for (size_t j = 0; j < width_; j++) {
          rotated.kernel_[(height_ - 1 - i) * width_ + (width_ - 1 - j)] =
              kernel_[i * width_ + j];
        }
      }
      break;
    }
    case KernelRotationDegrees::DEGREES_270: {
      rotated.height_ = width_;
      rotated.width_ = height_;
      for (size_t i = 0; i < height_; i++) {
        for (size_t j = 0; j < width_; j++) {
          rotated.kernel_[(width_ - 1 - j) * rotated.width_ + i] =
              kernel_[i * width_ + j];
        }
      }
      break;
    }
  }
  return rotated;
}

template <typename T>
size_t Kernel<T>::GetHeight() const {
  return height_;
}

template <typename T>
size_t Kernel<T>::GetWidth() const {
  return width_;
}

template <typename T>
bool Kernel<T>::IsRotatable() const {
  return rotatable_;
}

template <typename T>
T Kernel<T>::Get(const size_t x, const size_t y) const {
  return kernel_[y * width_ + x];
}

template <typename T>
void Kernel<T>::CopyKernelTo(T** dst) const {
  *dst = new T[height_ * width_];
  std::memcpy(*dst, kernel_, height_ * width_ * sizeof(T));
}

template <typename T>
Kernel<T> Kernel<T>::GetGaussianKernel(const size_t size, float sigma) {
  if ((size % 2) == 0u) {
    throw KernelException("Неверный размер ядра. Размер должен быть нечетным");
  }
  Kernel<T> gaussian_kernel(size, size, false);
  int half = size / 2;
  T sum = 0.0;
  if (sigma == 0) {
    sigma = size / 6.0;
    // T sigma = size * 0.15 + 0.35;
  }
  for (int i = -half; i <= half; i++) {
    for (int j = -half; j <= half; j++) {
      // T g = exp(-(i * i + j * j) / (2 * sigma * sigma)) /
      //       (2 * M_PI * sigma * sigma);
      T g = exp(-(i * i + j * j) / (2 * sigma * sigma));
      gaussian_kernel.kernel_[(i + half) * size + (j + half)] = g;
      sum += g;
    }
  }
  for (size_t i = 0; i < size; i++) {
    for (size_t j = 0; j < size; j++) {
      gaussian_kernel.kernel_[i * size + j] /= sum;
    }
  }
  return gaussian_kernel;
}

template <typename T>
Kernel<T> Kernel<T>::GetGaussianKernelSep(const size_t size, float sigma) {
  if ((size % 2) == 0u) {
    throw KernelException("Неверный размер ядра. Размер должен быть нечетным");
  }
  Kernel<T> gaussian_kernel(size, 1, true);
  int half = size / 2;
  T sum = 0.0;
  if (sigma == 0) {
    sigma = size / 6.0;
    // T sigma = size * 0.15 + 0.35;
  }
  for (int i = -half; i <= half; i++) {
    T g = exp(-i * i / (2 * sigma * sigma));
    gaussian_kernel.kernel_[i + half] = g;
    sum += g;
  }
  for (size_t i = 0; i < size; i++) {
    gaussian_kernel.kernel_[i] /= sum;
  }
  return gaussian_kernel;
}

template <typename T>
void Kernel<T>::SetFromFile(const std::string& filename) {
  std::ifstream file(filename);
  if (!file.is_open()) {
    throw KernelException("Не удалось открыть файл");
  }
  size_t height, width;
  bool rotate;
  if (!(file >> height >> width >> rotate)) {
    throw KernelException(
        "Ошибка чтения размера ядра или флага вращения из файла");
  }
  if (height % 2 == 0 || width % 2 == 0) {
    throw KernelException(
        "Неверный размер ядра. Высота и ширина должны быть нечетными");
  }
  if (kernel_ != nullptr) {
    delete[] kernel_;
  }
  height_ = height;
  width_ = width;
  rotatable_ = rotate;
  kernel_ = new T[height * width];
  for (size_t i = 0; i < height; i++) {
    for (size_t j = 0; j < width; j++) {
      if (!(file >> kernel_[i * width + j])) {
        throw KernelException("Ошибка чтения значения ядра из файла");
      }
    }
  }
  file.close();
}

/**
 * @brief Ядро оператора Собеля.
 *
 * Используется для выделения границ в изображении.
 *
 * Матрица ядра:
 * \f[
 * \begin{bmatrix}
 * -1 & 0 & 1 \\
 * -2 & 0 & 2 \\
 * -1 & 0 & 1
 * \end{bmatrix}
 * \f]
 */
const Kernel<int> kKernelSobel(3, 3, {{-1, 0, 1}, {-2, 0, 2}, {-1, 0, 1}},
                               true);

/**
 * @brief Ядро оператора Превитта.
 *
 * Используется для выделения границ в изображении.
 *
 * Матрица ядра:
 * \f[
 * \begin{bmatrix}
 * -1 & 0 & 1 \\
 * -1 & 0 & 1 \\
 * -1 & 0 & 1
 * \end{bmatrix}
 * \f]
 */
const Kernel<int> kKernelPrewitt(3, 3, {{-1, 0, 1}, {-1, 0, 1}, {-1, 0, 1}},
                                 true);
