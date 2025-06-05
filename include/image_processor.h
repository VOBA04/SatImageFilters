#ifndef IMAGE_PROCESSOR_H
#define IMAGE_PROCESSOR_H

#include <QQueue>
#include <QThread>
#include "tiff_image.h"
#include <QImage>
#include <cstddef>
#include "image_operation.h"
#include <QMutex>
#include <QWaitCondition>

/**
 * @file image_processor.h
 * @brief Заголовочный файл для класса ImageProcessor, который выполняет
 * обработку изображений в отдельном потоке.
 *
 * Этот файл содержит описание класса ImageProcessor, который управляет очередью
 * задач обработки изображений и выполняет их асинхронно. Поддерживаются
 * операции, такие как размытие по Гауссу, фильтры Собеля и Прюитта, а также их
 * комбинации.
 */

/**
 * @brief Структура, представляющая задачу для обработки изображений.
 *
 * Содержит операцию, которую нужно выполнить, и параметры для размытия по
 * Гауссу.
 */
struct ImageTask {
  ImageOperation operation;     ///< Операция обработки изображения.
  size_t gaussian_kernel_size;  ///< Размер ядра для размытия по Гауссу.
  float gaussian_sigma;         ///< Значение сигмы для размытия по Гауссу.
};

/**
 * @brief Класс для обработки задач обработки изображений в отдельном потоке.
 *
 * Класс ImageProcessor управляет очередью задач обработки изображений
 * и выполняет их асинхронно. Поддерживает операции, такие как размытие по
 * Гауссу, фильтры Собеля и Прюитта, а также их комбинации.
 */
class ImageProcessor : public QThread {
  Q_OBJECT
 private:
  TIFFImage image_;  ///< Текущее изображение, обрабатываемое в данный момент.
  QQueue<ImageTask> tasks_;  ///< Очередь задач для обработки.
  QMutex mutex_;  ///< Мьютекс для потокобезопасного доступа к общим ресурсам.
  QWaitCondition condition_;  ///< Условная переменная для синхронизации задач.
  bool stop_ = false;  ///< Флаг, указывающий, нужно ли остановить обработку.

 public:
  /**
   * @brief Конструктор класса ImageProcessor.
   *
   * @param parent Указатель на родительский объект QObject, если есть.
   */
  explicit ImageProcessor(QObject* parent = nullptr);

  /**
   * @brief Деструктор класса ImageProcessor.
   *
   * Останавливает поток обработки и освобождает ресурсы.
   */
  ~ImageProcessor() override;

  /**
   * @brief Добавляет новую задачу в очередь обработки.
   *
   * @param image Изображение для обработки.
   * @param task Задача, которую нужно выполнить над изображением.
   */
  void EnqueueTask(const TIFFImage& image, const ImageTask& task);

  /**
   * @brief Очищает все задачи из очереди обработки.
   */
  void ClearTasks();

  /**
   * @brief Останавливает поток обработки и очищает очередь задач.
   */
  void Stop();

 protected:
  /**
   * @brief Основной цикл обработки в потоке.
   *
   * Выполняет задачи из очереди до тех пор, пока не установлен флаг остановки.
   */
  void run() override;

 signals:
  /**
   * @brief Сигнал, испускаемый после завершения задачи.
   *
   * @param result_image Результирующее изображение после обработки.
   */
  void ResultReady(const TIFFImage& result_image);
};

#endif  // IMAGE_PROCESSOR_H
