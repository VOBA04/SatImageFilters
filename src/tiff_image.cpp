#include "tiff_image.h"
#include <tiffio.h>
#include <cstddef>
#include <cstdlib>
#include <stdexcept>
#include <string>

template <typename T>
void TIFFImage<T>::SwapBytes(T* array) {
  if (sizeof(T) != 2) {
    return;
  }
  for (size_t i = 0; i < width_; i++) {
    array[i] = (array[i] >> 8) | (array[i] << 8);
  }
}

template <typename T>
TIFFImage<T>::TIFFImage() {
}

template <typename T>
TIFFImage<T>::TIFFImage(const char* name) noexcept(false) {
  Open(name);
}

template <typename T>
TIFFImage<T>::TIFFImage(const std::string name) noexcept(false) {
  Open(name);
}

template <typename T>
TIFFImage<T>::~TIFFImage() {
  Close();
  if (image_ != nullptr) {
    for (size_t i = 0; i < height_; i++) {
      delete[] image_[i];
    }
    delete[] image_;
  }
}

template <typename T>
void TIFFImage<T>::Open(const char* name) noexcept(false) {
  tif_ = TIFFOpen(name, "r");
  if (tif_ == nullptr) {
    throw std::runtime_error("Невозможно открыть файл");
  }
  TIFFGetField(tif_, TIFFTAG_IMAGEWIDTH, &width_);
  TIFFGetField(tif_, TIFFTAG_IMAGELENGTH, &height_);
  TIFFGetField(tif_, TIFFTAG_SAMPLESPERPIXEL, &samples_per_pixel_);
  TIFFGetField(tif_, TIFFTAG_BITSPERSAMPLE, &bits_per_sample_);
  TIFFGetField(tif_, TIFFTAG_PLANARCONFIG, &config_);
  if (TIFFGetField(tif_, TIFFTAG_PHOTOMETRIC, &photo_metric_) != 1) {
    photo_metric_enabled_ = false;
  }
  if (TIFFGetField(tif_, TIFFTAG_RESOLUTIONUNIT, &resolution_unit_) != 1) {
    resolution_unit_enabled_ = false;
  }
  if (TIFFGetField(tif_, TIFFTAG_XRESOLUTION, &resolution_x_) == 0) {
    resolution_x_ = -1;
  }
  if (TIFFGetField(tif_, TIFFTAG_YRESOLUTION, &resolution_y_) == 0) {
    resolution_y_ = -1;
  }
  image_ = new T*[height_];
  for (size_t i = 0; i < height_; i++) {
    image_[i] = new T[width_];
    TIFFReadScanline(tif_, image_[i], i);
    SwapBytes(image_[i]);
  }
}

template <typename T>
void TIFFImage<T>::Open(const std::string name) noexcept(false) {
  Open(name.c_str());
}

template <typename T>
void TIFFImage<T>::Close() {
  if (tif_ != nullptr) {
    TIFFClose(tif_);
    tif_ = nullptr;
  }
}

template <typename T>
void TIFFImage<T>::Save(const char* name) {
  tif_ = TIFFOpen(name, "w");
  if (tif_ == nullptr) {
    throw std::runtime_error("Невозможно создать файл");
  }
  TIFFSetField(tif_, TIFFTAG_IMAGEWIDTH, width_);
  TIFFSetField(tif_, TIFFTAG_IMAGELENGTH, height_);
  TIFFSetField(tif_, TIFFTAG_SAMPLESPERPIXEL, samples_per_pixel_);
  TIFFSetField(tif_, TIFFTAG_BITSPERSAMPLE, bits_per_sample_);
  TIFFSetField(tif_, TIFFTAG_PLANARCONFIG, config_);
  if (photo_metric_enabled_) {
    TIFFSetField(tif_, TIFFTAG_PHOTOMETRIC, photo_metric_);
  }
  if (resolution_unit_enabled_) {
    TIFFSetField(tif_, TIFFTAG_RESOLUTIONUNIT, resolution_unit_);
  }
  if (resolution_x_ != -1) {
    TIFFSetField(tif_, TIFFTAG_XRESOLUTION, resolution_x_);
  }
  if (resolution_y_ != -1) {
    TIFFSetField(tif_, TIFFTAG_YRESOLUTION, resolution_y_);
  }
  for (size_t i = 0; i < height_; i++) {
    SwapBytes(image_[i]);
    TIFFWriteScanline(tif_, image_[i], i);
  }
  TIFFClose(tif_);
}

template <typename T>
void TIFFImage<T>::Save(const std::string name) {
  Save(name.c_str());
}

template <typename T>
void TIFFImage<T>::Clear() {
  if (image_ != nullptr) {
    for (size_t i = 0; i < height_; i++) {
      delete[] image_[i];
    }
    delete[] image_;
    image_ = nullptr;
  }
  width_ = 0;
  height_ = 0;
  samples_per_pixel_ = 0;
  bits_per_sample_ = 0;
  photo_metric_ = 0;
  resolution_unit_ = 0;
  config_ = 0;
  photo_metric_enabled_ = true;
  resolution_unit_enabled_ = true;
  resolution_x_ = 0;
  resolution_y_ = 0;
}

template <typename T>
T TIFFImage<T>::Get(const size_t x, const size_t y) const noexcept(false) {
  if ((width_ != 0u) && (height_ != 0u) && image_) {
    if (x < width_ || x >= width_ || y < height_ || y >= height_) {
      return 0;
    }
    return image_[y][x];
  } else {
    throw std::runtime_error("Изображение не загружено");
  }
}

template <typename T>
void TIFFImage<T>::Set(const size_t x, const size_t y,
                       const T value) noexcept(false) {
  if ((width_ != 0u) && (height_ != 0u) && image_) {
    if (x < width_ || x >= width_ || y < height_ || y >= height_) {
      throw std::runtime_error("Выход за границы изображения");
    }
    image_[y][x] = value;
  } else {
    throw std::runtime_error("Изображение не загружено");
  }
}

template <typename T>
void TIFFImage<T>::CopyFields(const TIFFImage<T>& other) {
  Clear();
  width_ = other.width_;
  height_ = other.height_;
  samples_per_pixel_ = other.samples_per_pixel_;
  bits_per_sample_ = other.bits_per_sample_;
  photo_metric_ = other.photo_metric_;
  resolution_unit_ = other.resolution_unit_;
  config_ = other.config_;
  photo_metric_enabled_ = other.photo_metric_enabled_;
  resolution_unit_enabled_ = other.resolution_unit_enabled_;
  resolution_x_ = other.resolution_x_;
  resolution_y_ = other.resolution_y_;
  TIFFSetField(tif_, TIFFTAG_IMAGEWIDTH, width_);
  TIFFSetField(tif_, TIFFTAG_IMAGELENGTH, height_);
  TIFFSetField(tif_, TIFFTAG_SAMPLESPERPIXEL, samples_per_pixel_);
  TIFFSetField(tif_, TIFFTAG_BITSPERSAMPLE, bits_per_sample_);
  TIFFSetField(tif_, TIFFTAG_PLANARCONFIG, config_);
  if (photo_metric_enabled_) {
    TIFFSetField(tif_, TIFFTAG_PHOTOMETRIC, photo_metric_);
  }
  if (resolution_unit_enabled_) {
    TIFFSetField(tif_, TIFFTAG_RESOLUTIONUNIT, resolution_unit_);
  }
  if (resolution_x_ != -1) {
    TIFFSetField(tif_, TIFFTAG_XRESOLUTION, resolution_x_);
  }
  if (resolution_y_ != -1) {
    TIFFSetField(tif_, TIFFTAG_YRESOLUTION, resolution_y_);
  }
  image_ = new T*[height_];
  for (size_t i = 0; i < height_; i++) {
    image_[i] = new T[width_];
    for (size_t j = 0; j < width_; j++) {
      image_[i][j] = other.image_[i][j];
    }
  }
}

template <typename T>
bool TIFFImage<T>::operator==(const TIFFImage<T>& other) const {
  if (width_ != other.width_ || height_ != other.height_ ||
      samples_per_pixel_ != other.samples_per_pixel_ ||
      bits_per_sample_ != other.bits_per_sample_ ||
      photo_metric_ != other.photo_metric_ ||
      resolution_unit_ != other.resolution_unit_ || config_ != other.config_ ||
      photo_metric_enabled_ != other.photo_metric_enabled_ ||
      resolution_unit_enabled_ != other.resolution_unit_enabled_ ||
      resolution_x_ != other.resolution_x_ ||
      resolution_y_ != other.resolution_y_) {
    return false;
  }

  for (size_t i = 0; i < height_; i++) {
    for (size_t j = 0; j < width_; j++) {
      if (image_[i][j] != other.image_[i][j]) {
        return false;
      }
    }
  }

  return true;
}

template <typename T>
TIFFImage<T>& TIFFImage<T>::operator=(const TIFFImage<T>& other) {
  if (this == &other) {
    return *this;
  }
  CopyFields(other);
  for (size_t i = 0; i < height_; i++) {
    for (size_t j = 0; j < width_; j++) {
      image_[i][j] = other.image_[i][j];
    }
  }
  return *this;
}

template <typename T>
TIFFImage<T> TIFFImage<T>::SetKernel(const Kernel& kernel, bool rotate) const {
  TIFFImage<T> result(*this);
  if (rotate) {
    Kernel kernel_y(kernel);
    kernel_y.Rotate(KernelRotationDegrees::DEGREES_90);
    for (size_t i = 0; i < height_; i++) {
      for (size_t j = 0; j < width_; j++) {
        int g_x = 0, g_y = 0;
        int radius = kernel.GetSize() / 2;
        for (int k = -radius; k <= radius; k++) {
          for (int l = -radius; l <= radius; l++) {
            g_x += kernel.Get(k + radius, l + radius) * Get(i + k, j + l);
            g_y += kernel_y.Get(k + radius, l + radius) * Get(i + k, j + l);
          }
        }
        result.image_[i][j] = abs(g_x) + abs(g_y);
      }
    }
  } else {
    for (size_t i = 0; i < height_; i++) {
      for (size_t j = 0; j < width_; j++) {
        int g = 0;
        int radius = kernel.GetSize() / 2;
        for (int k = -radius; k <= radius; k++) {
          for (int l = -radius; l <= radius; l++) {
            g += kernel.Get(k + radius, l + radius) * Get(i + k, j + l);
          }
        }
        result.image_[i][j] = abs(g);
      }
    }
  }
  return result;
}