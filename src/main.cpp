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
    std::cout << "Поместите исходные изображения в "
              << project_source_dir / "images/original" << std::endl;
    return 0;
  }
  fs::path original_images_dir(project_source_dir / "images/original");
  fs::path prewitt_images_dir(project_source_dir / "images/prewitt");
  fs::path sobel_images_dir(project_source_dir / "images/sobel");
  for (const auto& entry : fs::directory_iterator(original_images_dir)) {
    if (entry.is_regular_file()) {
      std::string image_name = entry.path().filename().string();
      std::string image_path = entry.path().string();
      TIFFImage<uint16_t> image;
      try {
        image.Open(image_path);
      } catch (std::runtime_error& e) {
        std::cerr << "Ошибка при загрузке изображения " << image_name << ": "
                  << e.what() << std::endl;
        continue;
      }
      TIFFImage<uint16_t> prewitt_image = image.SetKernel(kKernelPrewitt);
      prewitt_image.Save(prewitt_images_dir / image_name);
      TIFFImage<uint16_t> sobel_image = image.SetKernel(kKernelSobel);
      sobel_image.Save(sobel_images_dir / image_name);
    }
  }
  std::cout << "Изображения обработаны" << std::endl;
  return 0;
}