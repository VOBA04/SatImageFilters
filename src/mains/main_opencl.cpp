#include <filesystem>
#include <iostream>
#include <string>

#include "image_operation.h"
#include "kernel.h"
#include "tiff_image.h"

namespace fs = std::filesystem;

int main() {
  fs::path project_source_dir(PROJECT_SOURCE_DIR);
  fs::path original_images_dir(project_source_dir / "images/original");
  fs::path prewitt_images_dir(project_source_dir / "images/prewitt");
  fs::path sobel_images_dir(project_source_dir / "images/sobel");
  fs::path gaussian_images_dir(project_source_dir / "images/gaussian");
  fs::path kernel_path(project_source_dir / "kernel.txt");
  fs::path arbitrary_kernel_dir(project_source_dir / "images/arbitrary_kernel");

  if (!fs::exists(original_images_dir)) {
    std::cerr << "Каталог с оригинальными изображениями отсутствует: "
              << original_images_dir << std::endl;
    return 1;
  }

  if (fs::exists(gaussian_images_dir)) {
    fs::remove_all(gaussian_images_dir);
  }
  if (fs::exists(prewitt_images_dir)) {
    fs::remove_all(prewitt_images_dir);
  }
  if (fs::exists(sobel_images_dir)) {
    fs::remove_all(sobel_images_dir);
  }
  if (fs::exists(arbitrary_kernel_dir)) {
    fs::remove_all(arbitrary_kernel_dir);
  }

  for (const auto& entry : fs::directory_iterator(original_images_dir)) {
    if (!entry.is_regular_file()) {
      continue;
    }
    std::string image_name = entry.path().filename().string();
    std::string image_path = entry.path().string();
    TIFFImage image;
    try {
      image.Open(image_path);
    } catch (std::exception& e) {
      std::cerr << "Ошибка при загрузке изображения " << image_name << ": "
                << e.what() << std::endl;
      continue;
    }

    image.SetImagePatametersForOpenCLOps(ImageOperation::GaussianBlurSep |
                                             ImageOperation::Prewitt |
                                             ImageOperation::Sobel,
                                         9, 5);
    image.AllocateOpenCLMemory();
    image.CopyImageToOpenCLDevice();

    TIFFImage gaussian_image = image.GaussianBlurSepOpenCL(9, 5);
    if (!fs::exists(gaussian_images_dir)) {
      fs::create_directory(gaussian_images_dir);
    }
    gaussian_image.Save(gaussian_images_dir / image_name);

    TIFFImage prewitt_image = gaussian_image.SetKernelOpenCL(kKernelPrewitt);
    if (!fs::exists(prewitt_images_dir)) {
      fs::create_directory(prewitt_images_dir);
    }
    prewitt_image.Save(prewitt_images_dir / image_name);

    TIFFImage sobel_image = gaussian_image.SetKernelOpenCL(kKernelSobel);
    if (!fs::exists(sobel_images_dir)) {
      fs::create_directory(sobel_images_dir);
    }
    sobel_image.Save(sobel_images_dir / image_name);

    if (fs::exists(kernel_path)) {
      Kernel<int> arbitrary_kernel;
      try {
        arbitrary_kernel.SetFromFile(kernel_path);
        TIFFImage arbitrary_image = image.SetKernelOpenCL(arbitrary_kernel);
        if (!fs::exists(arbitrary_kernel_dir)) {
          fs::create_directory(arbitrary_kernel_dir);
        }
        arbitrary_image.Save(arbitrary_kernel_dir / image_name);
      } catch (std::exception& e) {
        std::cerr << "Ошибка ядра из файла " << kernel_path << ": " << e.what()
                  << std::endl;
      }
    }
  }
  std::cout << "OpenCL: изображения обработаны" << std::endl;
  return 0;
}
