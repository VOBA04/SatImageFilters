/**
 * @file main_window.h
 * @brief Заголовочный файл, содержащий описание класса MainWindow, который
 * реализует основное окно приложения для обработки изображений и видео.
 *
 * Этот файл включает в себя определения методов для загрузки, обработки
 * и сохранения изображений, а также работы с видеопотоком. Используются
 * библиотеки Qt и OpenCV.
 */

#ifndef MAIN_WINDOW_H
#define MAIN_WINDOW_H

#include <qcontainerfwd.h>
#include <qlist.h>
#include <qpixmap.h>
#include <qtimer.h>
#include <qtmetamacros.h>
#include <QMainWindow>
#include "tiff_image.h"
#include "image_processor.h"
#include <opencv2/opencv.hpp>
#include <QTimer>
#include <opencv2/videoio.hpp>

// NOLINTNEXTLINE(readability-identifier-naming)
namespace Ui {
/**
 * @namespace Ui
 * @brief Пространство имен, содержащее автоматически сгенерированный класс
 * MainWindow.
 */
class MainWindow;
}  // namespace Ui

/**
 * @class MainWindow
 * @brief Класс, представляющий главное окно приложения.
 *
 * Этот класс предоставляет функциональность для работы с изображениями и видео,
 * включая загрузку, обработку, отображение и сохранение результатов.
 */
class MainWindow : public QMainWindow {
  Q_OBJECT

 public:
  /**
   * @brief Конструктор класса MainWindow.
   * @param parent Указатель на родительский виджет (по умолчанию nullptr).
   */
  explicit MainWindow(QWidget* parent = nullptr);

  /**
   * @brief Деструктор класса MainWindow.
   */
  ~MainWindow();

 private slots:
  /**
   * @brief Слот для открытия изображения.
   */
  void OpenImage();

  /**
   * @brief Слот для обработки изображения.
   */
  void ProcessImage();

  /**
   * @brief Слот для обновления результата обработки изображения.
   * @param result_image Обработанное изображение.
   */
  void UpdateResult(const TIFFImage& result_image);

  /**
   * @brief Слот для сохранения обработанного изображения.
   */
  void SaveResult();

  /**
   * @brief Слот для запуска видеопотока.
   */
  void StartVideo();

  /**
   * @brief Слот для остановки видеопотока.
   */
  void StopVideo();

  /**
   * @brief Слот для обработки текущего кадра видеопотока.
   */
  void ProcessVideoFrame();

 private:
  Ui::MainWindow* ui_;        ///< Указатель на пользовательский интерфейс.
  TIFFImage image_;           ///< Исходное изображение.
  TIFFImage result_image_;    ///< Обработанное изображение.
  QPixmap original_pixmap_;   ///< Оригинальное изображение в формате QPixmap.
  QPixmap processed_pixmap_;  ///< Обработанное изображение в формате QPixmap.
  ImageProcessor*
      image_processor_;  ///< Указатель на объект обработки изображений.
  QTimer* timer_;        ///< Таймер для обработки видеопотока.
  cv::VideoCapture video_capture_;  ///< Объект для захвата видеопотока.

  /**
   * @brief Обновляет отображение изображений в пользовательском интерфейсе.
   */
  void UpdatePixmaps();

 protected:
  /**
   * @brief Переопределенный метод для обработки событий изменения размера окна.
   * @param event Указатель на событие изменения размера.
   */
  void resizeEvent(QResizeEvent* event) override;
};

#endif  // MAIN_WINDOW_H
