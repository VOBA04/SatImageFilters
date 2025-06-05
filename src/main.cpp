#include <QTranslator>
#include <QLibraryInfo>
#include <qlocale.h>
#include <QApplication>
#include "main_window.h"

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
  fs::path kernel_path(project_source_dir / "kernel.txt");
  fs::path arbitrary_kernel_dir(project_source_dir / "images/arbitrary_kernel");
  if (!fs::exists(original_images_dir)) {
    fs::create_directory(project_source_dir / "images/original");
    std::cerr << "Каталог с оригинальными изображениями отсутствует. Поместите "
                 "исходные изображения в "
              << original_images_dir << std::endl;
    return 1;
  }
  if (fs::exists(original_images_dir)) {
    bool has_images = false;
    for (const auto& entry : fs::directory_iterator(original_images_dir)) {
      if (entry.is_regular_file()) {
        has_images = true;
        break;
      }
    }
    if (!has_images) {
      std::cerr << "Каталог с оригинальными изображениями пуст. Поместите "
                << "исходные изображения в " << original_images_dir
                << std::endl;
      return 1;
    }
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
      image.ImageToDeviceMemory(ImageOperation::GaussianBlurSep |
                                    ImageOperation::Prewitt |
                                    ImageOperation::Sobel,
                                9, 5);
      // TIFFImage gaussian_image = image.GaussianBlur(9, 5);
      // TIFFImage gaussian_image = image.GaussianBlurSep(9, 5);
      // TIFFImage gaussian_image = image.GaussianBlurCuda(9, 5);
      TIFFImage gaussian_image = image.GaussianBlurSepCuda(9, 5);
      if (!fs::exists(gaussian_images_dir)) {
        fs::create_directory(gaussian_images_dir);
      }
      gaussian_image.Save(gaussian_images_dir / image_name);
      // TIFFImage prewitt_image = gaussian_image.SetKernel(kKernelPrewitt);
      // TIFFImage prewitt_image = gaussian_image.SetKernelCuda(kKernelPrewitt);
      // TIFFImage prewitt_image = gaussian_image.SetKernelPrewittSep();
      // TIFFImage prewitt_image = gaussian_image.SetKernelPrewittSepCuda();
      TIFFImage prewitt_image = gaussian_image.SetKernelCuda(kKernelPrewitt);
      if (!fs::exists(prewitt_images_dir)) {
        fs::create_directory(prewitt_images_dir);
      }
      prewitt_image.Save(prewitt_images_dir / image_name);
      if (!fs::exists(sobel_images_dir)) {
        fs::create_directory(sobel_images_dir);
      }
      // TIFFImage sobel_image = gaussian_image.SetKernel(kKernelSobel);
      // TIFFImage sobel_image = gaussian_image.SetKernelCuda(kKernelSobel);
      // TIFFImage sobel_image = gaussian_image.SetKernelSobelSep();
      // TIFFImage sobel_image = gaussian_image.SetKernelSobelSepCuda();
      TIFFImage sobel_image = gaussian_image.SetKernelCuda(kKernelSobel);
      sobel_image.Save(sobel_images_dir / image_name);
      if (fs::exists(kernel_path)) {
        Kernel<int> arbitrary_kernel;
        try {
          arbitrary_kernel.SetFromFile(kernel_path);
          // TIFFImage arbitrary_image =
          //     gaussian_image.SetKernel(arbitrary_kernel);
          TIFFImage arbitrary_image = image.SetKernelCuda(arbitrary_kernel);
          if (!fs::exists(arbitrary_kernel_dir)) {
            fs::create_directory(arbitrary_kernel_dir);
          }
          arbitrary_image.Save(arbitrary_kernel_dir / image_name);
        } catch (KernelException& e) {
          std::cerr << "Ошибка при загрузке ядра из файла " << kernel_path
                    << ": " << e.what() << std::endl;
        } catch (...) {
          std::cerr << "Неизвестная ошибка при загрузке ядра из файла "
                    << kernel_path << std::endl;
        }
      }
    }
  }
  std::cout << "Изображения обработаны" << std::endl;
  return 0;
}