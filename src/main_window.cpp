#include "main_window.h"
#include <opencv2/core/hal/interface.h>
#include <qfileinfo.h>
#include <qimage.h>
#include <qnamespace.h>
#include <qpixmap.h>
#include <qvariant.h>
#include "image_operation.h"
#include "image_processor.h"
#include "tiff_image.h"
#include "ui_main_window.h"
#include <QFileDialog>
#include <QDir>
#include <QMessageBox>
#include <cstdint>
#include <opencv2/core/mat.hpp>
#include <opencv2/imgproc.hpp>

struct ComboBoxItem {
  QString display_text;
  ImageOperation operation;
};

static const ComboBoxItem kComboBoxItems[] = {
    {"Фильтр Гаусса", ImageOperation::GaussianBlur},
    {"Оператор Собеля", ImageOperation::Sobel},
    {"Оператор Превитта", ImageOperation::Prewitt},
    {"Фильтр Гаусса + Оператор Собеля",
     ImageOperation::GaussianBlur | ImageOperation::Sobel},
    {"Фильтр Гаусса + Оператор Превитта",
     ImageOperation::GaussianBlur | ImageOperation::Prewitt}};

MainWindow::MainWindow(QWidget* parent)
    : QMainWindow(parent),
      ui_(new Ui::MainWindow),
      image_processor_(new ImageProcessor(this)) {
  ui_->setupUi(this);
  ui_->horizontal_layout_2->setAlignment(Qt::AlignLeft);
  ui_->horizontal_layout_3->setAlignment(Qt::AlignLeft);
  for (const auto& item : kComboBoxItems) {
    ui_->combobox->addItem(item.display_text,
                           QVariant::fromValue(item.operation));
  }
  ui_->combobox->setCurrentIndex(0);
  ui_->action_photo->setEnabled(false);
  ui_->action_save->setEnabled(false);
  timer_ = new QTimer(this);
  video_capture_ = cv::VideoCapture();
  connect(ui_->spinbox, &QSpinBox::valueChanged, this, [this](int value) {
    if (value % 2 == 0) {
      ui_->spinbox->setValue(value + 1);
    }
  });
  connect(ui_->action_open, &QAction::triggered, this, &MainWindow::OpenImage);
  connect(ui_->combobox, &QComboBox::currentTextChanged, this,
          &MainWindow::ProcessImage);
  connect(ui_->spinbox, &QSpinBox::valueChanged, this,
          &MainWindow::ProcessImage);
  connect(ui_->double_spinbox, &QDoubleSpinBox::valueChanged, this,
          &MainWindow::ProcessImage);
  connect(image_processor_, &ImageProcessor::ResultReady, this,
          &MainWindow::UpdateResult);
  connect(ui_->action_save, &QAction::triggered, this, &MainWindow::SaveResult);
  connect(ui_->action_video, &QAction::triggered, this,
          &MainWindow::StartVideo);
  connect(ui_->action_photo, &QAction::triggered, this, &MainWindow::StopVideo);
  connect(timer_, &QTimer::timeout, this, &MainWindow::ProcessVideoFrame);
  image_processor_->start();
}

MainWindow::~MainWindow() {
  delete ui_;
  delete image_processor_;
  delete timer_;
  if (video_capture_.isOpened()) {
    video_capture_.release();
  }
  image_.Close();
}

void MainWindow::OpenImage() {
  QString filename = QFileDialog::getOpenFileName(
      this, tr("Открыть изображение"), "/mnt/E/cv/project/SatImageFilters/",
      tr("TIFF Files (*.tiff *.tif)"));
  if (!filename.isEmpty()) {
    ui_->action_save->setEnabled(true);
    image_processor_->ClearTasks();
    image_.Close();
    image_.Open(filename.toStdString().c_str());
    image_.ImageToDeviceMemory(ImageOperation::Sobel | ImageOperation::Prewitt |
                                   ImageOperation::GaussianBlur,
                               ui_->spinbox->value(),
                               ui_->double_spinbox->value());
    original_pixmap_ = QPixmap::fromImage(image_.ToQImage());
    ui_->label_image_orig->setPixmap(
        original_pixmap_.scaled(ui_->label_image_orig->size(),
                                Qt::KeepAspectRatio, Qt::SmoothTransformation));
    ProcessImage();
  }
}

void MainWindow::UpdatePixmaps() {
  if (original_pixmap_.isNull()) {
    return;
  }
  QPixmap scaled_pixmap =
      original_pixmap_.scaled(ui_->label_image_orig->size(),
                              Qt::KeepAspectRatio, Qt::SmoothTransformation);
  ui_->label_image_orig->setPixmap(scaled_pixmap);
  if (processed_pixmap_.isNull()) {
    return;
  }
  QPixmap scaled_processed_pixmap =
      processed_pixmap_.scaled(ui_->label_image_result->size(),
                               Qt::KeepAspectRatio, Qt::SmoothTransformation);
  ui_->label_image_result->setPixmap(scaled_processed_pixmap);
}

void MainWindow::resizeEvent(QResizeEvent* event) {
  UpdatePixmaps();
  QMainWindow::resizeEvent(event);
}

void MainWindow::ProcessImage() {
  if (original_pixmap_.isNull()) {
    return;
  }
  ImageTask task = {ui_->combobox->currentData().value<ImageOperation>(),
                    static_cast<size_t>(ui_->spinbox->value()),
                    static_cast<float>(ui_->double_spinbox->value())};
  image_processor_->EnqueueTask(image_, task);
}

void MainWindow::UpdateResult(const TIFFImage& result_image) {
  result_image_ = result_image;
  processed_pixmap_ = QPixmap::fromImage(result_image.ToQImage());
  QPixmap scaled_pixmap =
      processed_pixmap_.scaled(ui_->label_image_result->size(),
                               Qt::KeepAspectRatio, Qt::SmoothTransformation);
  ui_->label_image_result->setPixmap(scaled_pixmap);
}

void MainWindow::SaveResult() {
  QString filename = QFileDialog::getSaveFileName(
      this, tr("Сохранить изображение"), "/mnt/E/cv/project/SatImageFilters/",
      tr("TIFF Files (*.tiff *.tif)"));
  if (!filename.isEmpty()) {
    QFileInfo file_info(filename);
    if (file_info.suffix().isEmpty()) {
      filename += ".tiff";
    }
    QFileInfo file_info_result(filename);
    if (file_info_result.exists()) {
      int ret = QMessageBox::warning(
          this, tr("Файл уже существует"),
          tr("Файл %1 уже существует. Перезаписать?").arg(filename),
          QMessageBox::Yes | QMessageBox::No);
      if (ret == QMessageBox::No) {
        return;
      }
    }
    result_image_.Save(filename.toStdString().c_str());
  }
}

void MainWindow::StartVideo() {
  video_capture_.open(0);
  if (!video_capture_.isOpened()) {
    QMessageBox::warning(this, tr("Ошибка"), tr("Не удалось открыть камеру"));
    return;
  }
  ui_->action_photo->setEnabled(true);
  ui_->action_open->setEnabled(false);
  ui_->action_video->setEnabled(false);
  ui_->action_save->setEnabled(false);
  image_processor_->ClearTasks();
  image_.Close();
  cv::Mat frame;
  video_capture_ >> frame;
  if (frame.empty()) {
    return;
  }
  cv::cvtColor(frame, frame, cv::COLOR_BGR2GRAY);
  frame.convertTo(frame, CV_16U, 256);
  image_.SetImage(frame.cols, frame.rows, (uint16_t*)frame.data);
  image_.ImageToDeviceMemory(ImageOperation::Sobel | ImageOperation::Prewitt |
                                 ImageOperation::GaussianBlur,
                             ui_->spinbox->value(),
                             ui_->double_spinbox->value());
  original_pixmap_ = QPixmap::fromImage(image_.ToQImage());
  ui_->label_image_orig->setPixmap(
      original_pixmap_.scaled(ui_->label_image_orig->size(),
                              Qt::KeepAspectRatio, Qt::SmoothTransformation));
  timer_->start(33);
}

void MainWindow::StopVideo() {
  timer_->stop();
  image_processor_->ClearTasks();
  video_capture_.release();
  image_.Close();
  ui_->action_photo->setEnabled(false);
  ui_->action_video->setEnabled(true);
  ui_->action_save->setEnabled(true);
  ui_->action_open->setEnabled(true);
  original_pixmap_ = QPixmap();
  processed_pixmap_ = QPixmap();
  ui_->label_image_orig->clear();
  ui_->label_image_result->clear();
}

void MainWindow::ProcessVideoFrame() {
  cv::Mat frame;
  video_capture_ >> frame;
  if (frame.empty()) {
    return;
  }
  cv::cvtColor(frame, frame, cv::COLOR_BGR2GRAY);
  frame.convertTo(frame, CV_16U, 256);
  image_.SetImage(frame.cols, frame.rows, (uint16_t*)frame.data);
  original_pixmap_ = QPixmap::fromImage(image_.ToQImage());
  ui_->label_image_orig->setPixmap(
      original_pixmap_.scaled(ui_->label_image_orig->size(),
                              Qt::KeepAspectRatio, Qt::SmoothTransformation));
  image_processor_->EnqueueTask(
      image_, {ui_->combobox->currentData().value<ImageOperation>(),
               static_cast<size_t>(ui_->spinbox->value()),
               static_cast<float>(ui_->double_spinbox->value())});
}