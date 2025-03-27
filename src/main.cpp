#include <iostream>
#include <filesystem>
#include "kernel.h"
#include "tiff_image.h"
#include <stdexcept>
#include <string>
#include <chrono>

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
  fs::path speedtest_dir(project_source_dir / "images/speedtest");
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
      // TIFFImage gaussian_image = image.GaussianBlur(9, 2);
      // TIFFImage gaussian_image = image.GaussianBlurSep(9, 5);
      TIFFImage gaussian_image = image.GaussianBlurCuda(9, 5);
      if (!fs::exists(gaussian_images_dir)) {
        fs::create_directory(gaussian_images_dir);
      }
      // auto start1 = std::chrono::high_resolution_clock::now();
      // image.GaussianBlur(9, 5);
      // auto end1 = std::chrono::high_resolution_clock::now();
      // auto start2 = std::chrono::high_resolution_clock::now();
      // image.GaussianBlurSep(9, 5);
      // auto end2 = std::chrono::high_resolution_clock::now();
      // std::cout << "GaussianBlur time: "
      //           << std::chrono::duration_cast<std::chrono::milliseconds>(end1
      //           -
      //                                                                    start1)
      //                  .count()
      //           << std::endl;
      // std::cout << "GaussianBlurSep time: "
      //           << std::chrono::duration_cast<std::chrono::milliseconds>(end2
      //           -
      //                                                                    start2)
      //                  .count()
      //           << std::endl;
      gaussian_image.Save(gaussian_images_dir / image_name);
      // TIFFImage prewitt_image = gaussian_image.SetKernel(kKernelPrewitt);
      TIFFImage prewitt_image = gaussian_image.SetKernelCuda(kKernelPrewitt);
      if (!fs::exists(prewitt_images_dir)) {
        fs::create_directory(prewitt_images_dir);
      }
      prewitt_image.Save(prewitt_images_dir / image_name);
      if (!fs::exists(sobel_images_dir)) {
        fs::create_directory(sobel_images_dir);
      }
      // TIFFImage sobel_image = gaussian_image.SetKernel(kKernelSobel);
      TIFFImage sobel_image = gaussian_image.SetKernelCuda(kKernelSobel);
      sobel_image.Save(sobel_images_dir / image_name);
      if (fs::exists(kernel_path)) {
        Kernel<int> arbitrary_kernel;
        try {
          arbitrary_kernel.SetFromFile(kernel_path);
          // TIFFImage arbitrary_image =
          //     gaussian_image.SetKernel(arbitrary_kernel);
          TIFFImage arbitrary_image =
              gaussian_image.SetKernelCuda(arbitrary_kernel);
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
  if (fs::exists(speedtest_dir)) {
    const auto& file = fs::directory_iterator(speedtest_dir);
    if (file->is_regular_file()) {
      std::string image_path = file->path().string();
      TIFFImage image(image_path);
      std::cout << "Запуск тестов на скорость обработки" << std::endl;
      std::cout << "Изображение: " << file->path().filename().string()
                << std::endl;
      std::cout << "Размер изображения: " << image.GetWidth() << "x"
                << image.GetHeight() << std::endl;
      std::cout << "Фильтр Гаусса:" << std::endl;
      auto start = std::chrono::high_resolution_clock::now();
      image.GaussianBlur(9, 5);
      auto end = std::chrono::high_resolution_clock::now();
      std::cout << "CPU: "
                << std::chrono::duration_cast<std::chrono::milliseconds>(end -
                                                                         start)
                       .count()
                << " ms" << std::endl;
      start = std::chrono::high_resolution_clock::now();
      image.GaussianBlurCuda(9, 5);
      end = std::chrono::high_resolution_clock::now();
      std::cout << "CUDA: "
                << std::chrono::duration_cast<std::chrono::milliseconds>(end -
                                                                         start)
                       .count()
                << " ms" << std::endl;
      std::cout << "Оператор Превитта:" << std::endl;
      start = std::chrono::high_resolution_clock::now();
      image.SetKernel(kKernelPrewitt);
      end = std::chrono::high_resolution_clock::now();
      std::cout << "CPU: "
                << std::chrono::duration_cast<std::chrono::milliseconds>(end -
                                                                         start)
                       .count()
                << " ms" << std::endl;
      start = std::chrono::high_resolution_clock::now();
      image.SetKernelCuda(kKernelPrewitt);
      end = std::chrono::high_resolution_clock::now();
      std::cout << "CUDA: "
                << std::chrono::duration_cast<std::chrono::milliseconds>(end -
                                                                         start)
                       .count()
                << " ms" << std::endl;
      std::cout << "Оператор Собеля:" << std::endl;
      start = std::chrono::high_resolution_clock::now();
      image.SetKernel(kKernelSobel);
      end = std::chrono::high_resolution_clock::now();
      std::cout << "CPU: "
                << std::chrono::duration_cast<std::chrono::milliseconds>(end -
                                                                         start)
                       .count()
                << " ms" << std::endl;
      start = std::chrono::high_resolution_clock::now();
      image.SetKernelCuda(kKernelSobel);
      end = std::chrono::high_resolution_clock::now();
      std::cout << "CUDA: "
                << std::chrono::duration_cast<std::chrono::milliseconds>(end -
                                                                         start)
                       .count()
                << " ms" << std::endl;
    }
  }
  return 0;
}