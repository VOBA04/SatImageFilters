/**
 * @file tiff_image.h
 * @brief Заголовочный файл, содержащий определение класса TIFFImage для работы
 * с TIFF изображениями.
 *
 * Этот файл содержит класс `TIFFImage` для работы с TIFF изображениями, включая
 * их создание, копирование, сохранение и обработку.
 */

#pragma once
#include <tiff.h>
#include <tiffio.h>

#include <cstddef>
#include <cstdint>
#include <string>
#include <vector>

#if __cplusplus > 201402L
#include <filesystem>
#endif

#ifdef BUILD_WITH_CUDA
#include "cuda_mem_manager.h"
#endif
#include "image_operation.h"
#include "kernel.h"

#include <CL/cl.h>

#ifdef USE_QT
#include <qimage.h>

#include <QImage>
#endif

/**
 * @brief Класс для работы с TIFF изображением.
 */
class TIFFImage {
 private:
  TIFF* tif_ = nullptr;             ///< Указатель на объект TIFF.
  size_t width_ = 0;                ///< Ширина изображения.
  size_t height_ = 0;               ///< Высота изображения.
  uint16_t samples_per_pixel_ = 1;  ///< Количество каналов на пиксель.
  uint16_t bits_per_sample_ = 16;   ///< Бит на канал.
  uint16_t photo_metric_ =
      PHOTOMETRIC_MINISBLACK;                ///< Фотометрическая интерпретация.
  uint16_t resolution_unit_ = RESUNIT_NONE;  ///< Единица измерения разрешения.
  uint16_t config_ = PLANARCONFIG_CONTIG;    ///< Конфигурация плоскостей.
  bool photo_metric_enabled_ = true;         ///< Флаг включения фотометрии.
  bool resolution_unit_enabled_ = true;  ///< Флаг включения единицы разрешения.
  float resolution_x_ = 0.0f;            ///< Разрешение по оси X.
  float resolution_y_ = 0.0f;            ///< Разрешение по оси Y.
  uint16_t* image_ =
      nullptr;  ///< Одномерный массив, представляющий изображение.
#ifdef BUILD_WITH_CUDA
  CudaMemManager
      cuda_mem_manager_{};  ///< Менеджер памяти CUDA (если доступна CUDA).
#endif

  // -------- OpenCL state --------
  // Контекст/устройство/очередь OpenCL и ресурсы для повторного использования
  cl_context cl_context_ = nullptr;
  cl_device_id cl_device_ = nullptr;
  cl_command_queue cl_queue_ = nullptr;
  cl_program cl_program_ = nullptr;

  // Буферы устройства (при предварительном выделении)
  cl_mem cl_src_ = nullptr;
  cl_mem cl_dst_ = nullptr;
  cl_mem cl_sep_g_x_ = nullptr;
  cl_mem cl_sep_g_y_ = nullptr;
  cl_mem cl_sep_result_x_ = nullptr;
  cl_mem cl_sep_result_y_ = nullptr;
  cl_mem cl_gaussian_kernel_ = nullptr;
  cl_mem cl_gaussian_sep_temp_ = nullptr;

  // Параметры и флаги выделения для OpenCL
  size_t cl_image_size_ = 0;
  size_t cl_gaussian_kernel_size_ = 0;
  float cl_gaussian_sigma_ = 0.0f;
  std::vector<ImageOperation> cl_image_operations_{};
  bool cl_allocated_ = false;

  // Вспомогательные методы OpenCL (инициализация/компиляция/очистка)
  void EnsureOpenCLInitialized();
  void EnsureOpenCLProgramBuilt();
  void ReleaseOpenCLProgram();
  void ReleaseOpenCLContext();
  void EnsureGaussianKernelBuffer(size_t kernel_size, float sigma);

  /**
   * @brief Складывает абсолютные значения элементов двух матриц.
   *
   * Функция выполняет сложение абсолютных значений соответствующих элементов
   * двух матриц и возвращает результат в виде новой матрицы типа uint16_t*.
   *
   * @param mtx1 Указатель на первую матрицу типа int.
   * @param mtx2 Указатель на вторую матрицу типа int.
   * @param height Высота матриц.
   * @param width Ширина матриц.
   * @return Указатель на новую матрицу типа uint16_t*, содержащую сумму
   *         абсолютных значений соответствующих элементов входных матриц.
   */
  static uint16_t* AddAbsMtx(const int* mtx1, const int* mtx2, size_t height,
                             size_t width);

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
  explicit TIFFImage(const std::string& name) noexcept(false);

#if __cplusplus > 201402L
  explicit TIFFImage(const std::filesystem::path& name) noexcept(false);
#endif

  TIFFImage(size_t width, size_t height, uint16_t samples_per_pixel = 1,
            uint16_t bits_per_sample = 16,
            uint16_t photo_metric = PHOTOMETRIC_MINISBLACK,
            uint16_t resolution_unit = RESUNIT_NONE, float resolution_x = 0.0f,
            float resolution_y = 0.0f, uint16_t config = PLANARCONFIG_CONTIG);

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
   * Уничтожает объект TIFFImage и освобождает выделенную память, включая
   * освобождение памяти на устройстве (если она была выделена).
   */
  ~TIFFImage();

  /**
   * @brief Открывает файл с изображением.
   *
   * Загружает TIFF изображение из файла и инициализирует параметры изображения.
   *
   * @param name Имя файла.
   * @throws std::runtime_error Если не удается открыть файл или файл поврежден.
   */
  void Open(const char* name) noexcept(false);

  /**
   * @brief Открывает файл с изображением.
   *
   * Загружает TIFF изображение из файла и инициализирует параметры изображения.
   *
   * @param name Имя файла.
   * @throws std::runtime_error Если не удается открыть файл или файл поврежден.
   */
  void Open(const std::string& name) noexcept(false);

#if __cplusplus > 201402L
  void Open(const std::filesystem::path& name) noexcept(false);
#endif

  /**
   * @brief Закрывает файл с изображением.
   *
   * Освобождает ресурсы, связанные с открытым TIFF файлом, и сбрасывает
   * внутренние параметры объекта.
   */
  void Close();

  /**
   * @brief Сохраняет изображение в файл.
   *
   * Сохраняет текущее состояние изображения в указанный файл формата TIFF.
   *
   * @param name Имя файла.
   * @throws std::runtime_error Если не удается создать или записать файл.
   */
  void Save(const char* name);

  /**
   * @brief Сохраняет изображение в файл.
   *
   * Сохраняет текущее состояние изображения в указанный файл формата TIFF.
   *
   * @param name Имя файла.
   * @throws std::runtime_error Если не удается создать или записать файл.
   */
  void Save(const std::string& name);

  /**
   * @brief Сохраняет изображение в файл.
   *
   * Сохраняет текущее состояние изображения в указанный файл формата TIFF.
   *
   * @param name Имя файла.
   * @throws std::runtime_error Если не удается создать или записать файл.
   */
#if __cplusplus > 201402L
  void Save(const std::filesystem::path& name);
#endif

  /**
   * @brief Очищает изображение.
   *
   * Освобождает выделенную память для изображения и сбрасывает параметры
   * объекта, включая освобождение памяти на устройстве (если она была
   * выделена).
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
   * @brief Возвращает ширину изображения.
   *
   * @return Ширина изображения.
   */
  size_t GetWidth() const;

  /**
   * @brief Возвращает высоту изображения.
   *
   * @return Высота изображения.
   */
  size_t GetHeight() const;

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

  void SetImage(const size_t width, const size_t height,
                const uint16_t* image) noexcept(false);

  /**
   * @brief Копирует поля из другого объекта TIFFImage.
   *
   * @param other Исходный объект TIFFImage для копирования.
   */
  void CopyFields(const TIFFImage& other);

  // CUDA memory helpers — доступны только при сборке с CUDA
#ifdef BUILD_WITH_CUDA
  void SetImagePatametersForDevice(
      ImageOperation operations = ImageOperation::None,
      size_t gaussian_kernel_size = 0, float gaussian_sigma = 0);
  void AllocateDeviceMemory();
  void ReallocateDeviceMemory();
  void CopyImageToDevice();
  void FreeDeviceMemory();
#endif

  // ---------- OpenCL memory management (аналог CUDA) ----------
  void SetImagePatametersForOpenCLOps(
      ImageOperation operations = ImageOperation::None,
      size_t gaussian_kernel_size = 0, float gaussian_sigma = 0.0f);

  void AllocateOpenCLMemory();
  void ReallocateOpenCLMemory();
  void CopyImageToOpenCLDevice();
  void FreeOpenCLMemory();

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

  friend std::ostream& operator<<(std::ostream& out,
                                  const TIFFImage& tiff_image);

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
   * @brief Применяет ядро к изображению с использованием CUDA.
   *
   * Создает новое изображение, применяя ядро свертки с использованием CUDA.
   *
   * @param kernel Ядро свертки.
   * @param rotate Флаг, указывающий, нужно ли поворачивать ядро (по умолчанию
   * true).
   * @return Новое изображение с примененным ядром.
   */
  // CUDA kernels — только при сборке с CUDA
#ifdef BUILD_WITH_CUDA
  TIFFImage SetKernelCuda(const Kernel<int>& kernel,
                          const bool shared_memory = true,
                          const bool rotate = true) const;
#endif

  // --------- OpenCL compute (аналог CUDA) ----------
  TIFFImage SetKernelOpenCL(const Kernel<int>& kernel,
                            const bool shared_memory = true,
                            const bool rotate = true) const;

  /**
   * @brief Применяет разделенный оператор Собеля к изображению.
   *
   * Создает новое изображение, применяя фильтр Собеля с использованием
   * метода разделения ядра, что повышает эффективность вычислений.
   *
   * @return Новое изображение с примененным разделенным оператором Собеля.
   */
  TIFFImage SetKernelSobelSep() const;

  /**
   * @brief Применяет разделенный оператор Превитта к изображению.
   *
   * Создает новое изображение, применяя фильтр Превитта с использованием
   * метода разделения ядра, что повышает эффективность вычислений.
   *
   * @return Новое изображение с примененным разделенным оператором Прюитта.
   */
  TIFFImage SetKernelPrewittSep() const;

/**
 * @brief Применяет разделенный оператор Собеля к изображению с использованием
 * CUDA.
 *
 * Создает новое изображение, применяя фильтр Собеля с использованием
 * метода разделения ядра и вычислений на GPU через CUDA.
 * Этот метод обеспечивает ускорение обработки изображений большого размера.
 *
 * @return Новое изображение с примененным разделенным оператором Собеля.
 */
#ifdef BUILD_WITH_CUDA
  TIFFImage SetKernelSobelSepCuda(const bool shared_memory = true) const;
#endif

  TIFFImage SetKernelSobelSepOpenCL(const bool shared_memory = true) const;

/**
 * @brief Применяет разделенный оператор Превитта к изображению с
 * использованием CUDA.
 *
 * Создает новое изображение, применяя фильтр Превитта с использованием
 * метода разделения ядра и вычислений на GPU через CUDA.
 * Этот метод обеспечивает ускорение обработки изображений большого размера.
 *
 * @return Новое изображение с примененным разделенным оператором Прюитта.
 */
#ifdef BUILD_WITH_CUDA
  TIFFImage SetKernelPrewittSepCuda(const bool shared_memory = true) const;
#endif

  TIFFImage SetKernelPrewittSepOpenCL(const bool shared_memory = true) const;

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

  /**
   * @brief Применяет разделенный фильтр Гаусса к изображению.
   *
   * Создает новое изображение, применяя разделенный фильтр Гаусса.
   * Разделенный фильтр Гаусса выполняет свертку по одной оси (горизонтальной
   * или вертикальной) за один проход, что снижает вычислительную сложность.
   *
   * Формула фильтра Гаусса:
   * \f[
   * G(x) = e^{-\frac{x^2}{2\sigma^2}}
   * \f]
   *
   * Если \f$\sigma\f$ не задан, он вычисляется как \f$\sigma =
   * \frac{size}{6}\f$.
   *
   * @param size Размер фильтра (должен быть нечетным).
   * @param sigma Стандартное отклонение (опционально).
   * @return Новое изображение с примененным разделенным фильтром Гаусса.
   */
  TIFFImage GaussianBlurSep(const size_t size = 3,
                            const float sigma = 0.0) const;

/**
 * @brief Применяет фильтр Гаусса к изображению с использованием CUDA.
 *
 * Создает новое изображение, применяя фильтр Гаусса с использованием CUDA.
 *
 * @param size Размер фильтра (должен быть нечетным).
 * @param sigma Стандартное отклонение (опционально).
 * @return Новое изображение с примененным фильтром Гаусса.
 */
#ifdef BUILD_WITH_CUDA
  TIFFImage GaussianBlurCuda(const size_t size = 3, const float sigma = 0.0);
#endif

  TIFFImage GaussianBlurOpenCL(const size_t size = 3, const float sigma = 0.0);

/**
 * @brief Применяет разделенный фильтр Гаусса к изображению с использованием
 * CUDA.
 *
 * Создает новое изображение, применяя разделенный фильтр Гаусса с
 * использованием вычислений на GPU через CUDA. Разделенный фильтр Гаусса
 * выполняет свертку по одной оси (горизонтальной или вертикальной) за один
 * проход, что значительно снижает вычислительную сложность. Использование
 * CUDA дополнительно ускоряет обработку для изображений большого размера.
 *
 * @param size Размер фильтра (должен быть нечетным).
 * @param sigma Стандартное отклонение (опционально).
 * @return Новое изображение с примененным разделенным фильтром Гаусса.
 */
#ifdef BUILD_WITH_CUDA
  TIFFImage GaussianBlurSepCuda(const size_t size = 3, const float sigma = 0.0);
#endif

  TIFFImage GaussianBlurSepOpenCL(const size_t size = 3,
                                  const float sigma = 0.0);

#ifdef USE_QT
  /**
   * @brief Преобразует изображение в формат QImage.
   *
   * Эта функция создает объект QImage из текущего изображения, что позволяет
   * использовать его в приложениях с графическим интерфейсом на основе Qt.
   *
   * @return Объект QImage, представляющий текущее изображение.
   * @throws std::runtime_error Если изображение не загружено или преобразование
   * невозможно.
   */
  QImage ToQImage() const;
#endif
};
