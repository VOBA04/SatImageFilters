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
class MainWindow;
}

class MainWindow : public QMainWindow {
  Q_OBJECT

 public:
  explicit MainWindow(QWidget* parent = nullptr);
  ~MainWindow();

 private slots:
  void OpenImage();
  void ProcessImage();
  void UpdateResult(const TIFFImage& result_image);
  void SaveResult();
  void StartVideo();
  void StopVideo();
  void ProcessVideoFrame();

 private:
  Ui::MainWindow* ui_;
  TIFFImage image_;
  TIFFImage result_image_;
  QPixmap original_pixmap_;
  QPixmap processed_pixmap_;
  ImageProcessor* image_processor_;
  QTimer* timer_;
  cv::VideoCapture video_capture_;

  void UpdatePixmaps();

 protected:
  void resizeEvent(QResizeEvent* event) override;
};

#endif  // MAIN_WINDOW_H
