#include "image_processor.h"
#include <qdebug.h>
#include <qlogging.h>
#include "image_operation.h"
#include "kernel.h"
#include "tiff_image.h"

ImageProcessor::ImageProcessor(QObject* parent)
    : QThread(parent) {
}

ImageProcessor::~ImageProcessor() {
  Stop();
  wait();
}

void ImageProcessor::EnqueueTask(TIFFImage* image, const ImageTask& task) {
  QMutexLocker locker(&mutex_);
  image_ = image;
  tasks_.enqueue(task);
  condition_.wakeOne();
}

void ImageProcessor::ClearTasks() {
  QMutexLocker locker(&mutex_);
  tasks_.clear();
}

void ImageProcessor::Stop() {
  QMutexLocker locker(&mutex_);
  stop_ = true;
  tasks_.clear();
  condition_.wakeOne();
}

void ImageProcessor::run() {
  while (true) {
    ImageTask task;
    {
      QMutexLocker locker(&mutex_);
      if (stop_) {
        return;
      }
      if (tasks_.isEmpty()) {
        condition_.wait(&mutex_);
        if (stop_) {
          return;
        }
      }
      if (!tasks_.isEmpty()) {
        task = tasks_.dequeue();
      }
    }
    TIFFImage result_image;
    if (task.operation == ImageOperation::GaussianBlur) {
#ifdef BUILD_WITH_CUDA
      image_->CopyImageToDevice();
      result_image = image_->GaussianBlurCuda(task.gaussian_kernel_size,
                                              task.gaussian_sigma);
#else
      result_image =
          image_->GaussianBlur(task.gaussian_kernel_size, task.gaussian_sigma);
#endif
    } else if (task.operation == ImageOperation::Sobel) {
#ifdef BUILD_WITH_CUDA
      image_->CopyImageToDevice();
      result_image = image_->SetKernelCuda(kKernelSobel);
#else
      result_image = image_->SetKernel(kKernelSobel);
#endif
    } else if (task.operation == ImageOperation::Prewitt) {
#ifdef BUILD_WITH_CUDA
      image_->CopyImageToDevice();
      result_image = image_->SetKernelCuda(kKernelPrewitt);
#else
      result_image = image_->SetKernel(kKernelPrewitt);
#endif
    } else if (task.operation ==
               (ImageOperation::GaussianBlur | ImageOperation::Sobel)) {
#ifdef BUILD_WITH_CUDA
      image_->CopyImageToDevice();
      TIFFImage gaussian_blurred_image = image_->GaussianBlurCuda(
          task.gaussian_kernel_size, task.gaussian_sigma);
      gaussian_blurred_image.SetImagePatametersForDevice(ImageOperation::Sobel);
      gaussian_blurred_image.AllocateDeviceMemory();
      gaussian_blurred_image.CopyImageToDevice();
      result_image = gaussian_blurred_image.SetKernelCuda(kKernelSobel);
#else
      TIFFImage gaussian_blurred_image =
          image_->GaussianBlur(task.gaussian_kernel_size, task.gaussian_sigma);
      result_image = gaussian_blurred_image.SetKernel(kKernelSobel);
#endif
    } else if (task.operation ==
               (ImageOperation::GaussianBlur | ImageOperation::Prewitt)) {
#ifdef BUILD_WITH_CUDA
      image_->CopyImageToDevice();
      TIFFImage gaussian_blurred_image = image_->GaussianBlurCuda(
          task.gaussian_kernel_size, task.gaussian_sigma);
      gaussian_blurred_image.SetImagePatametersForDevice(
          ImageOperation::Prewitt);
      gaussian_blurred_image.AllocateDeviceMemory();
      gaussian_blurred_image.CopyImageToDevice();
      result_image = gaussian_blurred_image.SetKernelCuda(kKernelPrewitt);
#else
      TIFFImage gaussian_blurred_image =
          image_->GaussianBlur(task.gaussian_kernel_size, task.gaussian_sigma);
      result_image = gaussian_blurred_image.SetKernel(kKernelPrewitt);
#endif
    }
    emit ResultReady(result_image);
  }
}
