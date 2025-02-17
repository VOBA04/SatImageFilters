#pragma once
#include "kernel.h"
#include <cstddef>
#include <cstdint>
#include <string>
#include <tiff.h>
#include <tiffio.h>

/**
 * @brief Класс для работы с tiff изображением
 *
 * @tparam T Битность изображения (8 или 16)
 */
template <typename T> class TIFFImage {
private:
  TIFF *tif_;
  size_t width_, height_;
  uint16_t samples_per_pixel_, bits_per_sample_, photo_metric_,
      resolution_unit_;
  bool photo_metric_enabled_ = true, resolution_unit_enabled_ = true;
  float resolution_x_, resolution_y_;
  T **image_ = NULL;

  void ShiftArrayLeft(T *array);
  void ShiftArrayRight(T *array);

public:
  /**
   * @brief Создайте новый объект TIFFImage
   *
   */
  TIFFImage();
  /**
   * @brief Создайте новый объект TIFFImage
   *
   * @param name Имя файла
   */
  TIFFImage(const char *name) noexcept(false);
  /**
   * @brief Создайте новый объект TIFFImage
   *
   * @param name Имя файла
   */
  TIFFImage(std::string name) noexcept(false);
  /**
   * @brief Уничтожает объект TIFFImage
   *
   */
  ~TIFFImage();
  void Open(const char *name) noexcept(false);
  void Open(const std::string name) noexcept(false);
  void Close();
  void Save(const char *name) const;
  void Save(const std::string name) const;
  T Get(const size_t x, const size_t y) const;
  TIFFImage<T> &operator=(const TIFFImage<T> another);
  void SetKernel(const Kernel &kernel);
};