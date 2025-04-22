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

struct ImageTask {
  ImageOperation operation;
  size_t gaussian_kernel_size;
  float gaussian_sigma;
};

class ImageProcessor : public QThread {
  Q_OBJECT
 private:
  TIFFImage image_;
  QQueue<ImageTask> tasks_;
  QMutex mutex_;
  QWaitCondition condition_;
  bool stop_ = false;

 public:
  explicit ImageProcessor(QObject* parent = nullptr);
  ~ImageProcessor() override;
  void EnqueueTask(const TIFFImage& image, const ImageTask& task);
  void ClearTasks();
  void Stop();

 protected:
  void run() override;

 signals:
  void ResultReady(const TIFFImage& result_image);
};

#endif  // IMAGE_PROCESSOR_H
