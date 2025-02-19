/**
 * @file tiff_image.h
 * @brief Заголовочный файл, содержащий определение класса TIFFImage для работы
 * с TIFF изображениями.
 *
 * Этот файл содержит класс `TIFFImage` для работы с TIFF изображениями, включая
 * их создание, копирование, сохранение и обработку.
 */

#pragma once
#include "kernel.h"
#include <cstddef>
#include <cstdint>
#include <stdexcept>
#include <string>
#include <tiff.h>
#include <tiffio.h>

/**
 * @brief Класс для работы с TIFF изображением.
 *
 * @tparam T Тип данных изображения (8 или 16 бит).
 */
template <typename T>
class TIFFImage {
 private:
  TIFF* tif_ = nullptr;  ///< Указатель на объект TIFF.
  size_t width_ = 0, height_ = 0;  ///< Ширина и высота изображения.
  uint16_t samples_per_pixel_, bits_per_sample_, photo_metric_,
      resolution_unit_, config_;  ///< Параметры изображения.
  bool photo_metric_enabled_ = true,
       resolution_unit_enabled_ = true;  ///< Флаги включения параметров.
  float resolution_x_, resolution_y_;  ///< Разрешение по осям X и Y.
  T** image_ = nullptr;  ///< Двумерный массив, представляющий изображение.

  /**
   * @brief Меняет порядок байтов в массиве.
   *
   * @param array Массив данных.
   */
  void SwapBytes(T* array);

 public:
  /**
   * @brief Конструктор по умолчанию.
   *
   * Создает пустой объект TIFFImage.
   */
  TIFFImage();

  /**
   * @brief Конструктор с параметрами.
   *
   * Создает объект TIFFImage и открывает файл с изображением.
   *
   * @param name Имя файла.
   * @throws std::runtime_error Если не удается открыть файл.
   */
  explicit TIFFImage(const char* name) noexcept(false);

  /**
   * @brief Конструктор с параметрами.
   *
   * Создает объект TIFFImage и открывает файл с изображением.
   *
   * @param name Имя файла.
   * @throws std::runtime_error Если не удается открыть файл.
   */
  explicit TIFFImage(std::string name) noexcept(false);

  /**
   * @brief Конструктор копирования.
   *
   * Создает новый объект TIFFImage, копируя содержимое другого объекта.
   *
   * @param other Исходный объект TIFFImage для копирования.
   */
  TIFFImage(const TIFFImage<T>& other);

  /**
   * @brief Деструктор.
   *
   * Уничтожает объект TIFFImage и освобождает выделенную память.
   */
  ~TIFFImage();

  /**
   * @brief Открывает файл с изображением.
   *
   * @param name Имя файла.
   * @throws std::runtime_error Если не удается открыть файл.
   */
  void Open(const char* name) noexcept(false);

  /**
   * @brief Открывает файл с изображением.
   *
   * @param name Имя файла.
   * @throws std::runtime_error Если не удается открыть файл.
   */
  void Open(const std::string name) noexcept(false);

  /**
   * @brief Закрывает файл с изображением.
   *
   * Закрывает файл с изображением.
   */
  void Close();

  /**
   * @brief Сохраняет изображение в файл.
   *
   * @param name Имя файла.
   * @throws std::runtime_error Если не удается создать файл.
   */
  void Save(const char* name);

  /**
   * @brief Сохраняет изображение в файл.
   *
   * @param name Имя файла.
   * @throws std::runtime_error Если не удается создать файл.
   */
  void Save(const std::string name);

  /**
   * @brief Очищает изображение.
   *
   * Освобождает память и сбрасывает параметры изображения.
   */
  void Clear();

  /**
   * @brief Возвращает значение пикселя.
   *
   * @param x Координата x.
   * @param y Координата y.
   * @return Значение пикселя.
   * @throws std::runtime_error Если изображение не загружено.
   */
  T Get(const int x, const int y) const noexcept(false);

  /**
   * @brief Устанавливает значение пикселя.
   *
   * @param x Координата x.
   * @param y Координата y.
   * @param value Значение пикселя.
   * @throws std::runtime_error Если изображение не загружено или координаты
   * выходят за границы.
   */
  void Set(const size_t x, const size_t y, const T value) noexcept(false);

  /**
   * @brief Копирует поля из другого объекта TIFFImage.
   *
   * @param other Исходный объект TIFFImage для копирования.
   */
  void CopyFields(const TIFFImage<T>& other);

  /**
   * @brief Оператор сравнения.
   *
   * Сравнивает два объекта TIFFImage.
   *
   * @param other Объект для сравнения.
   * @return true, если объекты равны, иначе false.
   */
  bool operator==(const TIFFImage<T>& other) const;

  /**
   * @brief Оператор присваивания.
   *
   * Копирует содержимое другого объекта TIFFImage.
   *
   * @param other Исходный объект TIFFImage для копирования.
   * @return Ссылка на текущий объект.
   */
  TIFFImage<T>& operator=(const TIFFImage<T>& other);

  /**
   * @brief Применяет ядро к изображению.
   *
   * Создает новое изображение, применяя ядро свертки.
   *
   * @param kernel Ядро свертки.
   * @param rotate Флаг, указывающий, нужно ли поворачивать ядро.
   * @return Новое изображение с примененным ядром.
   */
  TIFFImage<T> SetKernel(const Kernel& kernel, bool rotate = true) const;
};

template <typename T>
void TIFFImage<T>::SwapBytes(T* array) {
  if (sizeof(T) != 2) {
    return;
  }
  for (size_t i = 0; i < width_; i++) {
    array[i] = (array[i] >> 8) | (array[i] << 8);
  }
}

template <typename T>
TIFFImage<T>::TIFFImage() {
}

template <typename T>
TIFFImage<T>::TIFFImage(const char* name) noexcept(false) {
  Open(name);
}

template <typename T>
TIFFImage<T>::TIFFImage(const std::string name) noexcept(false) {
  Open(name);
}

template <typename T>
TIFFImage<T>::TIFFImage(const TIFFImage<T>& other) {
  CopyFields(other);
  for (size_t i = 0; i < height_; i++) {
    for (size_t j = 0; j < width_; j++) {
      image_[i][j] = other.image_[i][j];
    }
  }
}

template <typename T>
TIFFImage<T>::~TIFFImage() {
  Close();
  if (image_ != nullptr) {
    for (size_t i = 0; i < height_; i++) {
      delete[] image_[i];
    }
    delete[] image_;
  }
}

template <typename T>
void TIFFImage<T>::Open(const char* name) noexcept(false) {
  tif_ = TIFFOpen(name, "r");
  if (tif_ == nullptr) {
    throw std::runtime_error("Невозможно открыть файл");
  }
  TIFFGetField(tif_, TIFFTAG_IMAGEWIDTH, &width_);
  TIFFGetField(tif_, TIFFTAG_IMAGELENGTH, &height_);
  TIFFGetField(tif_, TIFFTAG_SAMPLESPERPIXEL, &samples_per_pixel_);
  TIFFGetField(tif_, TIFFTAG_BITSPERSAMPLE, &bits_per_sample_);
  TIFFGetField(tif_, TIFFTAG_PLANARCONFIG, &config_);
  if (TIFFGetField(tif_, TIFFTAG_PHOTOMETRIC, &photo_metric_) != 1) {
    photo_metric_enabled_ = false;
  }
  if (TIFFGetField(tif_, TIFFTAG_RESOLUTIONUNIT, &resolution_unit_) != 1) {
    resolution_unit_enabled_ = false;
  }
  if (TIFFGetField(tif_, TIFFTAG_XRESOLUTION, &resolution_x_) == 0) {
    resolution_x_ = -1;
  }
  if (TIFFGetField(tif_, TIFFTAG_YRESOLUTION, &resolution_y_) == 0) {
    resolution_y_ = -1;
  }
  image_ = new T*[height_];
  for (size_t i = 0; i < height_; i++) {
    image_[i] = new T[width_];
    TIFFReadScanline(tif_, image_[i], i);
    // SwapBytes(image_[i]);
  }
}

template <typename T>
void TIFFImage<T>::Open(const std::string name) noexcept(false) {
  Open(name.c_str());
}

template <typename T>
void TIFFImage<T>::Close() {
  if (tif_ != nullptr) {
    TIFFClose(tif_);
    tif_ = nullptr;
  }
}

template <typename T>
void TIFFImage<T>::Save(const char* name) {
  tif_ = TIFFOpen(name, "w");
  if (tif_ == nullptr) {
    throw std::runtime_error("Невозможно создать файл");
  }
  TIFFSetField(tif_, TIFFTAG_IMAGEWIDTH, width_);
  TIFFSetField(tif_, TIFFTAG_IMAGELENGTH, height_);
  TIFFSetField(tif_, TIFFTAG_SAMPLESPERPIXEL, samples_per_pixel_);
  TIFFSetField(tif_, TIFFTAG_BITSPERSAMPLE, bits_per_sample_);
  TIFFSetField(tif_, TIFFTAG_PLANARCONFIG, config_);
  if (photo_metric_enabled_) {
    TIFFSetField(tif_, TIFFTAG_PHOTOMETRIC, photo_metric_);
  }
  if (resolution_unit_enabled_) {
    TIFFSetField(tif_, TIFFTAG_RESOLUTIONUNIT, resolution_unit_);
  }
  if (resolution_x_ != -1) {
    TIFFSetField(tif_, TIFFTAG_XRESOLUTION, resolution_x_);
  }
  if (resolution_y_ != -1) {
    TIFFSetField(tif_, TIFFTAG_YRESOLUTION, resolution_y_);
  }
  for (size_t i = 0; i < height_; i++) {
    // SwapBytes(image_[i]);
    TIFFWriteScanline(tif_, image_[i], i);
  }
  TIFFClose(tif_);
  tif_ = nullptr;
}

template <typename T>
void TIFFImage<T>::Save(const std::string name) {
  Save(name.c_str());
}

template <typename T>
void TIFFImage<T>::Clear() {
  if (image_ != nullptr) {
    for (size_t i = 0; i < height_; i++) {
      delete[] image_[i];
    }
    delete[] image_;
    image_ = nullptr;
  }
  width_ = 0;
  height_ = 0;
  samples_per_pixel_ = 0;
  bits_per_sample_ = 0;
  photo_metric_ = 0;
  resolution_unit_ = 0;
  config_ = 0;
  photo_metric_enabled_ = true;
  resolution_unit_enabled_ = true;
  resolution_x_ = 0;
  resolution_y_ = 0;
}

template <typename T>
T TIFFImage<T>::Get(const int x, const int y) const noexcept(false) {
  if ((width_ != 0u) && (height_ != 0u) && image_) {
    if (x < 0 || x >= static_cast<int>(width_) || y < 0 ||
        y >= static_cast<int>(height_)) {
      return 0;
    }
    return image_[y][x];
  } else {
    throw std::runtime_error("Изображение не загружено");
  }
}

template <typename T>
void TIFFImage<T>::Set(const size_t x, const size_t y,
                       const T value) noexcept(false) {
  if ((width_ != 0u) && (height_ != 0u) && image_) {
    if (x < width_ || x >= width_ || y < height_ || y >= height_) {
      throw std::runtime_error("Выход за границы изображения");
    }
    image_[y][x] = value;
  } else {
    throw std::runtime_error("Изображение не загружено");
  }
}

template <typename T>
void TIFFImage<T>::CopyFields(const TIFFImage<T>& other) {
  Clear();
  width_ = other.width_;
  height_ = other.height_;
  samples_per_pixel_ = other.samples_per_pixel_;
  bits_per_sample_ = other.bits_per_sample_;
  photo_metric_ = other.photo_metric_;
  resolution_unit_ = other.resolution_unit_;
  config_ = other.config_;
  photo_metric_enabled_ = other.photo_metric_enabled_;
  resolution_unit_enabled_ = other.resolution_unit_enabled_;
  resolution_x_ = other.resolution_x_;
  resolution_y_ = other.resolution_y_;
  image_ = new T*[height_];
  for (size_t i = 0; i < height_; i++) {
    image_[i] = new T[width_];
    for (size_t j = 0; j < width_; j++) {
      image_[i][j] = other.image_[i][j];
    }
  }
}

template <typename T>
bool TIFFImage<T>::operator==(const TIFFImage<T>& other) const {
  if (width_ != other.width_ || height_ != other.height_ ||
      samples_per_pixel_ != other.samples_per_pixel_ ||
      bits_per_sample_ != other.bits_per_sample_ ||
      photo_metric_ != other.photo_metric_ ||
      resolution_unit_ != other.resolution_unit_ || config_ != other.config_ ||
      photo_metric_enabled_ != other.photo_metric_enabled_ ||
      resolution_unit_enabled_ != other.resolution_unit_enabled_ ||
      resolution_x_ != other.resolution_x_ ||
      resolution_y_ != other.resolution_y_) {
    return false;
  }

  for (size_t i = 0; i < height_; i++) {
    for (size_t j = 0; j < width_; j++) {
      if (image_[i][j] != other.image_[i][j]) {
        return false;
      }
    }
  }

  return true;
}

template <typename T>
TIFFImage<T>& TIFFImage<T>::operator=(const TIFFImage<T>& other) {
  if (this == &other) {
    return *this;
  }
  CopyFields(other);
  for (size_t i = 0; i < height_; i++) {
    for (size_t j = 0; j < width_; j++) {
      image_[i][j] = other.image_[i][j];
    }
  }
  return *this;
}

template <typename T>
TIFFImage<T> TIFFImage<T>::SetKernel(const Kernel& kernel, bool rotate) const {
  TIFFImage<T> result(*this);
  if (rotate) {
    Kernel kernel_y(kernel);
    kernel_y.Rotate(KernelRotationDegrees::DEGREES_90);
    for (size_t i = 0; i < height_; i++) {
      for (size_t j = 0; j < width_; j++) {
        int g_x = 0, g_y = 0;
        int radius = kernel.GetSize() / 2;
        for (int k = -radius; k <= radius; k++) {
          for (int l = -radius; l <= radius; l++) {
            g_x += kernel.Get(k + radius, l + radius) * Get(i + k, j + l);
            g_y += kernel_y.Get(k + radius, l + radius) * Get(i + k, j + l);
          }
        }
        result.image_[i][j] = abs(g_x) + abs(g_y);
      }
    }
  } else {
    for (size_t i = 0; i < height_; i++) {
      for (size_t j = 0; j < width_; j++) {
        int g = 0;
        int radius = kernel.GetSize() / 2;
        for (int k = -radius; k <= radius; k++) {
          for (int l = -radius; l <= radius; l++) {
            g += kernel.Get(k + radius, l + radius) * Get(i + k, j + l);
          }
        }
        result.image_[i][j] = abs(g);
      }
    }
  }
  return result;
}