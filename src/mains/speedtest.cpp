#include "image_operation.h"
#include "tiff_image.h"
#include <algorithm>
#include <cstddef>
#include <iostream>
#include <filesystem>
#include <chrono>
#include <fstream>
#include <ostream>
#include <string>
#include <utility>
#include <vector>

namespace fs = std::filesystem;
namespace ch = std::chrono;

int main() {
  fs::path project_source_dir(PROJECT_SOURCE_DIR);
  if (!fs::exists(project_source_dir / "images")) {
    fs::create_directory(project_source_dir / "images");
    fs::create_directory(project_source_dir / "images/speedtest");
    std::cout << "Поместите изображения для теста скорости в "
              << project_source_dir / "images/speedtest" << std::endl;
    return 0;
  }
  fs::path speedtest_dir(project_source_dir / "images/speedtest");
  int i = 0;
  if (fs::exists(speedtest_dir)) {
    std::ofstream fout(project_source_dir / "result.txt");
    std::vector<std::pair<size_t, fs::directory_entry>> files;
    std::cout << "Запуск тестов на скорость обработки" << std::endl;
    for (const auto& file : fs::directory_iterator(speedtest_dir)) {
      if (file.is_regular_file() &&
          file.path().string().find(".tiff") != std::string::npos) {
        files.push_back({file.file_size(), file});
      }
    }
    std::sort(files.begin(), files.end());
    for (auto file : files) {
      std::string image_path = file.second.path().string();
      TIFFImage image(image_path);
      i++;
      std::cout << i
                << ". Изображение: " << file.second.path().filename().string()
                << std::endl;
      fout << "Изображение: " << file.second.path().filename().string()
           << std::endl;
      fout << "Размер изображения: " << image.GetWidth() << "x"
           << image.GetHeight() << std::endl;
      fout << "Фильтр Гаусса:" << std::endl;
      auto start = ch::high_resolution_clock::now();
      image.GaussianBlur(9, 5);
      auto end = ch::high_resolution_clock::now();
      fout << "CPU: "
           << ch::duration_cast<ch::microseconds>(end - start).count() / 1000.0
           << " ms" << std::endl;
      start = ch::high_resolution_clock::now();
      image.GaussianBlurSep(9, 5);
      end = ch::high_resolution_clock::now();
      fout << "CPU (Sep): "
           << ch::duration_cast<ch::microseconds>(end - start).count() / 1000.0
           << " ms" << std::endl;
      image.SetImagePatametersForDevice(ImageOperation::GaussianBlur, 9, 5);
      image.AllocateDeviceMemory();
      image.CopyImageToDevice();
      start = ch::high_resolution_clock::now();
      image.GaussianBlurCuda(9, 5);
      end = ch::high_resolution_clock::now();
      image.FreeDeviceMemory();
      fout << "CUDA: "
           << ch::duration_cast<ch::microseconds>(end - start).count() / 1000.0
           << " ms" << std::endl;
      image.SetImagePatametersForDevice(ImageOperation::GaussianBlurSep, 9, 5);
      start = ch::high_resolution_clock::now();
      image.GaussianBlurSepCuda(9, 5);
      end = ch::high_resolution_clock::now();
      image.FreeDeviceMemory();
      fout << "CUDA (Sep): "
           << ch::duration_cast<ch::microseconds>(end - start).count() / 1000.0
           << " ms" << std::endl;
      fout << "Оператор Превитта:" << std::endl;
      start = ch::high_resolution_clock::now();
      image.SetKernel(kKernelPrewitt);
      end = ch::high_resolution_clock::now();
      fout << "CPU: "
           << ch::duration_cast<ch::microseconds>(end - start).count() / 1000.0
           << " ms" << std::endl;
      start = ch::high_resolution_clock::now();
      image.SetKernelPrewittSep();
      end = ch::high_resolution_clock::now();
      fout << "CPU (Sep): "
           << ch::duration_cast<ch::microseconds>(end - start).count() / 1000.0
           << " ms" << std::endl;
      image.SetImagePatametersForDevice(ImageOperation::Sobel |
                                        ImageOperation::Prewitt |
                                        ImageOperation::Separated);
      start = ch::high_resolution_clock::now();
      image.SetKernelCuda(kKernelPrewitt);
      end = ch::high_resolution_clock::now();
      fout << "CUDA: "
           << ch::duration_cast<ch::microseconds>(end - start).count() / 1000.0
           << " ms" << std::endl;
      start = ch::high_resolution_clock::now();
      image.SetKernelPrewittSepCuda();
      end = ch::high_resolution_clock::now();
      fout << "CUDA (Sep): "
           << ch::duration_cast<ch::microseconds>(end - start).count() / 1000.0
           << " ms" << std::endl;
      fout << "Оператор Собеля:" << std::endl;
      start = ch::high_resolution_clock::now();
      image.SetKernel(kKernelSobel);
      end = ch::high_resolution_clock::now();
      fout << "CPU: "
           << ch::duration_cast<ch::microseconds>(end - start).count() / 1000.0
           << " ms" << std::endl;
      start = ch::high_resolution_clock::now();
      image.SetKernelSobelSep();
      end = ch::high_resolution_clock::now();
      fout << "CPU (Sep): "
           << ch::duration_cast<ch::microseconds>(end - start).count() / 1000.0
           << " ms" << std::endl;
      start = ch::high_resolution_clock::now();
      image.SetKernelCuda(kKernelSobel);
      end = ch::high_resolution_clock::now();
      fout << "CUDA: "
           << ch::duration_cast<ch::microseconds>(end - start).count() / 1000.0
           << " ms" << std::endl;
      start = ch::high_resolution_clock::now();
      image.SetKernelSobelSepCuda();
      end = ch::high_resolution_clock::now();
      image.FreeDeviceMemory();
      fout << "CUDA (Sep): "
           << ch::duration_cast<ch::microseconds>(end - start).count() / 1000.0
           << " ms" << std::endl;
      fout << "=============================================================="
           << std::endl;
    }
    fout.close();
    std::cout << "Тест скорости окончен" << std::endl;
  }
  return 0;
}