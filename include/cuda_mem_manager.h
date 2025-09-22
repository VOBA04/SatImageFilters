/**
 * @file cuda_mem_manager.h
 * @brief Управление памятью и ресурсами CUDA для операций обработки
 * изображений.
 *
 * Класс CudaMemManager инкапсулирует выделение/освобождение памяти на
 * устройстве, копирование данных между хостом и устройством, а также хранение
 * параметров, необходимых для сверток (в т.ч. разделённых фильтров и
 * гауссова размытия).
 */

#pragma once
#include <cstddef>
#include <cstdint>
#include <vector>
#include "image_operation.h"

/**
 * @class CudaMemManager
 * @brief Менеджер памяти CUDA и связанных буферов для ускоренной обработки.
 *
 * Обеспечивает:
 * - Выделение/перевыделение/освобождение буферов на GPU.
 * - Копирование исходного/результирующего изображения между CPU и GPU.
 * - Подготовку и кеширование ядра Гаусса (плотного и разделённого вида).
 * - Хранение параметров и набора операций для последующего запуска ядер.
 */
class CudaMemManager {
  size_t image_size_ = 0;  ///< Размер изображения (ширина*высота)
  size_t gaussian_kernel_size_ = 0;  ///< Текущий размер ядра Гаусса
  float gaussian_sigma_ = 0.0f;  ///< Текущее значение сигмы для Гаусса
  uint16_t* d_src_ =
      nullptr;  ///< Указатель на исходное изображение на устройстве
  uint16_t* d_dst_ =
      nullptr;  ///< Указатель на выходное изображение на устройстве
  int* d_sep_g_x_ =
      nullptr;  ///< Временный буфер для разделённого фильтра (ось X)
  int* d_sep_g_y_ =
      nullptr;  ///< Временный буфер для разделённого фильтра (ось Y)
  int* d_sep_result_x_ =
      nullptr;  ///< Промежуточный результат после свертки по X
  int* d_sep_result_y_ =
      nullptr;  ///< Промежуточный результат после свертки по Y
  float* d_gaussian_kernel_ = nullptr;  ///< Буфер ядра Гаусса (плотного)
  float* d_gaussian_sep_temp_ = nullptr;  ///< Буфер для разделённого Гаусса
  std::vector<ImageOperation>
      image_operations_;  ///< Запрошенный набор операций
  bool is_allocated_ = false;  ///< Признак, что память на устройстве выделена

  /**
   * @brief Инициализация буфера ядра Гаусса (если параметры заданы).
   */
  void InitializeGaussianKernel();

  /**
   * @brief Перевыделение буфера ядра Гаусса при изменении параметров.
   */
  void ReallocateGaussianKernel();

 public:
  /**
   * @brief Конструктор по умолчанию.
   */
  CudaMemManager();

  /**
   * @brief Деструктор. Освобождает все выделенные GPU-ресурсы.
   */
  ~CudaMemManager();

  /**
   * @brief Выделяет память на устройстве под все необходимые буферы.
   *
   * Вызывает исключение при нехватке памяти на устройстве.
   */
  void AllocateMemory();

  /**
   * @brief Освобождает всю память на устройстве.
   */
  void FreeMemory();

  /**
   * @brief Перевыделяет память на устройстве с учётом текущих параметров.
   */
  void ReallocateMemory();

  /**
   * @brief Копирует изображение-хост на устройство (в d_src_).
   * @param src Указатель на исходные пиксели на CPU.
   */
  void CopyImageToDevice(const uint16_t* src);

  /**
   * @brief Копирует изображение с устройства (d_dst_) на хост.
   * @param dst Указатель на буфер-приёмник на CPU.
   */
  void CopyImageFromDevice(uint16_t* dst);

  /**
   * @brief Устанавливает размер изображения в пикселях.
   * @param width Ширина изображения.
   * @param height Высота изображения.
   */
  void SetImageSize(size_t width, size_t height);

  /**
   * @brief Устанавливает размер изображения, переданный как width*height.
   * @param image_size Число пикселей (width*height).
   */
  void SetImageSize(size_t image_size);

  /**
   * @brief Задаёт параметры гауссова фильтра.
   * @param kernel_size Размер ядра (нечётный).
   * @param sigma Значение сигмы (0.0 — авторасчёт).
   */
  void SetGaussianParameters(size_t kernel_size, float sigma = 0.0f);

  /**
   * @brief Проверяет соответствие текущего буфера ядра Гаусса заданным
   * параметрам.
   * @param kernel_size Размер ядра (нечётный).
   * @param sigma Значение сигмы (0.0 — авторасчёт).
   *
   * При несовпадении выполняется перевыделение и заполнение буфера ядра.
   */
  void CheckGaussianKernel(size_t kernel_size, float sigma = 0.0f);

  /**
   * @brief Устанавливает набор операций, который будет применён на GPU.
   * @param operations Комбинация значений ImageOperation.
   */
  void SetImageOperations(const ImageOperation operations);

  /**
   * @brief Проверяет наличие достаточного объёма свободной памяти на
   * устройстве.
   * @param required_memory Требуемый объём (в байтах).
   * @return true, если памяти достаточно, иначе false.
   */
  bool CheckFreeMemory(size_t required_memory) const;

  /** @brief Возвращает указатель на исходное изображение на устройстве. */
  uint16_t* GetDeviceSrc() const;
  /** @brief Возвращает указатель на выходное изображение на устройстве. */
  uint16_t* GetDeviceDst() const;
  /** @brief Вспомогательный буфер: свертка по X. */
  int* GetDeviceSepGx() const;
  /** @brief Вспомогательный буфер: свертка по Y. */
  int* GetDeviceSepGy() const;
  /** @brief Промежуточный результат после X. */
  int* GetDeviceSepResultX() const;
  /** @brief Промежуточный результат после Y. */
  int* GetDeviceSepResultY() const;
  /** @brief Буфер ядра Гаусса. */
  float* GetDeviceGaussianKernel() const;
  /** @brief Буфер для разделённого гауссова размытия. */
  float* GetDeviceGaussianSepTemp() const;

  /**
   * @brief Признак, что память на устройстве выделена.
   * @return true, если буферы выделены; иначе false.
   */
  bool IsAllocated() const;
};