#include "tiff_image.h"
#include <stdexcept>

void TIFFImage::SwapBytes(uint16_t* array) {
  if (sizeof(uint16_t) != 2) {
    return;
  }
  for (size_t i = 0; i < width_; i++) {
    array[i] = (array[i] >> 8) | (array[i] << 8);
  }
}

TIFFImage::TIFFImage() {
}

TIFFImage::TIFFImage(const char* name) noexcept(false) {
  Open(name);
}

TIFFImage::TIFFImage(const std::string name) noexcept(false) {
  Open(name);
}

TIFFImage::TIFFImage(const TIFFImage& other) {
  CopyFields(other);
  for (size_t i = 0; i < height_; i++) {
    for (size_t j = 0; j < width_; j++) {
      image_[i][j] = other.image_[i][j];
    }
  }
}

TIFFImage::~TIFFImage() {
  Close();
  if (image_ != nullptr) {
    for (size_t i = 0; i < height_; i++) {
      delete[] image_[i];
    }
    delete[] image_;
  }
}

void TIFFImage::Open(const char* name) noexcept(false) {
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
  image_ = new uint16_t*[height_];
  for (size_t i = 0; i < height_; i++) {
    image_[i] = new uint16_t[width_];
    TIFFReadScanline(tif_, image_[i], i);
    // SwapBytes(image_[i]);
  }
}

void TIFFImage::Open(const std::string name) noexcept(false) {
  Open(name.c_str());
}

void TIFFImage::Close() {
  if (tif_ != nullptr) {
    TIFFClose(tif_);
    tif_ = nullptr;
  }
}

void TIFFImage::Save(const char* name) {
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
    // SwapBytes(image_[i]);
    TIFFWriteScanline(tif_, image_[i], i);
  }
  TIFFClose(tif_);
  tif_ = nullptr;
}

void TIFFImage::Save(const std::string name) {
  Save(name.c_str());
}

void TIFFImage::Clear() {
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

uint16_t TIFFImage::Get(const int x, const int y) const noexcept(false) {
  if ((width_ != 0u) && (height_ != 0u) && (image_ != nullptr)) {
    if (x < 0 || x >= static_cast<int>(width_) || y < 0 ||
        y >= static_cast<int>(height_)) {
      return 0;
    }
    return image_[x][y];
  } else {
    throw std::runtime_error("Изображение не загружено");
  }
}

void TIFFImage::Set(const size_t x, const size_t y,
                    const uint16_t value) noexcept(false) {
  if ((width_ != 0u) && (height_ != 0u) && (image_ != nullptr)) {
    if (x < width_ || x >= width_ || y < height_ || y >= height_) {
      throw std::runtime_error("Выход за границы изображения");
    }
    image_[y][x] = value;
  } else {
    throw std::runtime_error("Изображение не загружено");
  }
}

void TIFFImage::CopyFields(const TIFFImage& other) {
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
  image_ = new uint16_t*[height_];
  for (size_t i = 0; i < height_; i++) {
    image_[i] = new uint16_t[width_];
    for (size_t j = 0; j < width_; j++) {
      image_[i][j] = other.image_[i][j];
    }
  }
}

bool TIFFImage::operator==(const TIFFImage& other) const {
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

TIFFImage& TIFFImage::operator=(const TIFFImage& other) {
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

TIFFImage TIFFImage::SetKernel(const Kernel& kernel, bool rotate) const {
  TIFFImage result(*this);
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