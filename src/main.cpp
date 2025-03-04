#include <iostream>
#include <filesystem>
#include "kernel.h"
#include "tiff_image.h"
#include <stdexcept>
#include <string>

namespace fs = std::filesystem;

int main() {
  fs::path project_source_dir(PROJECT_SOURCE_DIR);
  if (!fs::exists(project_source_dir / "images")) {
    fs::create_directory(project_source_dir / "images");
    fs::create_directory(project_source_dir / "images/original");
    fs::create_directory(project_source_dir / "images/prewitt");
    fs::create_directory(project_source_dir / "images/sobel");
    fs::create_directory(project_source_dir / "images/gaussian");
    std::cout << "Поместите исходные изображения в "
              << project_source_dir / "images/original" << std::endl;
    return 0;
  }
  fs::path original_images_dir(project_source_dir / "images/original");
  fs::path prewitt_images_dir(project_source_dir / "images/prewitt");
  fs::path sobel_images_dir(project_source_dir / "images/sobel");
  fs::path gaussian_images_dir(project_source_dir / "images/gaussian");

  if (!fs::exists(original_images_dir)) {
    fs::create_directory(project_source_dir / "images/original");
    std::cerr << "Каталог с оригинальными изображениями отсутствует. Поместите "
                 "исходные изображения в "
              << original_images_dir << std::endl;
    return 1;
  }
  for (const auto& entry : fs::directory_iterator(gaussian_images_dir)) {
    if (entry.is_regular_file()) {
      fs::remove(entry);
    }
  }
  for (const auto& entry : fs::directory_iterator(prewitt_images_dir)) {
    if (entry.is_regular_file()) {
      fs::remove(entry);
    }
  }
  for (const auto& entry : fs::directory_iterator(sobel_images_dir)) {
    if (entry.is_regular_file()) {
      fs::remove(entry);
    }
  }

  for (const auto& entry : fs::directory_iterator(original_images_dir)) {
    if (entry.is_regular_file()) {
      std::string image_name = entry.path().filename().string();
      std::string image_path = entry.path().string();
      TIFFImage image;
      try {
        image.Open(image_path);
      } catch (std::runtime_error& e) {
        std::cerr << "Ошибка при загрузке изображения " << image_name << ": "
                  << e.what() << std::endl;
        continue;
      } catch (...) {
        std::cerr << "Неизвестная ошибка при загрузке изображения "
                  << image_name << std::endl;
        continue;
      }
      TIFFImage gaussian_image = image.GaussianBlur(5);
      if (!fs::exists(gaussian_images_dir)) {
        fs::create_directory(gaussian_images_dir);
      }
      gaussian_image.Save(gaussian_images_dir / image_name);
      TIFFImage prewitt_image = gaussian_image.SetKernel(kKernelPrewitt);
      if (!fs::exists(prewitt_images_dir)) {
        fs::create_directory(prewitt_images_dir);
      }
      prewitt_image.Save(prewitt_images_dir / image_name);
      if (!fs::exists(sobel_images_dir)) {
        fs::create_directory(sobel_images_dir);
      }
      TIFFImage sobel_image = gaussian_image.SetKernel(kKernelSobel);
      sobel_image.Save(sobel_images_dir / image_name);
    }
  }
  std::cout << "Изображения обработаны" << std::endl;
  return 0;
}