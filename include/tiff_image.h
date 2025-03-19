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
#include <string>
#include <tiff.h>
#include <tiffio.h>

/**
 * @brief Класс для работы с TIFF изображением.
 */
class TIFFImage {
 private:
  TIFF* tif_ = nullptr;  ///< Указатель на объект TIFF.
  size_t width_ = 0, height_ = 0;  ///< Ширина и высота изображения.
  uint16_t samples_per_pixel_, bits_per_sample_, photo_metric_,
      resolution_unit_, config_;  ///< Параметры изображения.
  bool photo_metric_enabled_ = true,
       resolution_unit_enabled_ = true;  ///< Флаги включения параметров.
  float resolution_x_, resolution_y_;  ///< Разрешение по осям X и Y.
  uint16_t** image_ =
      nullptr;  ///< Двумерный массив, представляющий изображение.

  /**
   * @brief Меняет порядок байтов в массиве.
   *
   * @param array Массив данных.
   */
  void SwapBytes(uint16_t* array);

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
  TIFFImage(const TIFFImage& other);

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
  void Open(const std::string& name) noexcept(false);

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
  void Save(const std::string& name);

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
  uint16_t Get(const int x, const int y) const noexcept(false);

  /**
   * @brief Устанавливает значение пикселя.
   *
   * @param x Координата x.
   * @param y Координата y.
   * @param value Значение пикселя.
   * @throws std::runtime_error Если изображение не загружено или координаты
   * выходят за границы.
   */
  void Set(const size_t x, const size_t y,
           const uint16_t value) noexcept(false);

  /**
   * @brief Копирует поля из другого объекта TIFFImage.
   *
   * @param other Исходный объект TIFFImage для копирования.
   */
  void CopyFields(const TIFFImage& other);

  /**
   * @brief Оператор сравнения.
   *
   * Сравнивает два объекта TIFFImage.
   *
   * @param other Объект для сравнения.
   * @return true, если объекты равны, иначе false.
   */
  bool operator==(const TIFFImage& other) const;

  /**
   * @brief Оператор присваивания.
   *
   * Копирует содержимое другого объекта TIFFImage.
   *
   * @param other Исходный объект TIFFImage для копирования.
   * @return Ссылка на текущий объект.
   */
  TIFFImage& operator=(const TIFFImage& other);

  /**
   * @brief Применяет ядро к изображению.
   *
   * Создает новое изображение, применяя ядро свертки.
   *
   * @param kernel Ядро свертки.
   * @param rotate Флаг, указывающий, нужно ли поворачивать ядро (по умолчанию
   * true).
   * @return Новое изображение с примененным ядром.
   */
  TIFFImage SetKernel(const Kernel<int>& kernel, bool rotate = true) const;

  /**
   * @brief Применяет фильтр Гаусса к изображению.
   *
   * Создает новое изображение, применяя фильтр Гаусса.
   *
   * Формула фильтра Гаусса:
   * \f[
   * G(x, y) = \frac{1}{2\pi\sigma^2} e^{-\frac{x^2 + y^2}{2\sigma^2}}
   * \f]
   *
   * Если \f$\sigma\f$ не задан, он вычисляется как \f$\sigma =
   * \frac{size}{6}\f$.
   *
   * @param size Размер фильтра (должен быть нечетным).
   * @param sigma Стандартное отклонение (опционально).
   * @return Новое изображение с примененным фильтром Гаусса.
   */
  TIFFImage GaussianBlur(const size_t size = 3, const float sigma = 0.0) const;
};
