#include <stdexcept>
#include <string>
#include <algorithm>
#include <vector>
#include <iostream>
#include <chrono>

#include "command_line_parser.h"
#include "tiff_image.h"
#ifdef _WIN32
#include <windows.h>
#endif

enum class Functions { Sobel, SobelSep, Prewitt, PrewittSep, Gauss, GaussSep };

std::ostream& operator<<(std::ostream& out, Functions f) {
  switch (f) {
    case Functions::Sobel:
      return (out << "Sobel");
    case Functions::SobelSep:
      return (out << "SobelSep");
    case Functions::Prewitt:
      return (out << "Prewitt");
    case Functions::PrewittSep:
      return (out << "PrewittSep");
    case Functions::Gauss:
      return (out << "Gauss");
    case Functions::GaussSep:
      return (out << "GaussSep");
  }
  return out;
}

Functions CheckFunctionArg(std::string arg) {
  std::vector<std::string> args(
      {"Sobel", "SobelSep", "Prewitt", "PrewittSep" /*, "Gauss", "GaussSep"*/});
  auto it = std::find(args.begin(), args.end(), arg);
  if (it == args.end()) {
    throw std::invalid_argument("Wrong function argument: " + arg);
  }
  return static_cast<Functions>(it - args.begin());
}

std::pair<size_t, size_t> CheckSizeArg(std::string arg) {
  auto pos = arg.find("x");
  if (pos == std::string::npos) {
    throw std::invalid_argument("Wrong size argument: " + arg);
  }
  int h = std::stoi(arg.substr(0, pos));
  int w = std::stoi(arg.substr(pos + 1));
  return std::pair<size_t, size_t>(h, w);
}

size_t CheckGaussSize(std::string arg) {
  return std::stoi(arg);
}

float ChechGaussSigma(std::string arg) {
  return std::stof(arg);
}

size_t ChechCountArg(std::string arg) {
  int count = std::stoi(arg);
  return count;
}

int main(int argc, char* argv[]) {
#ifdef _WIN32
  SetConsoleOutputCP(CP_UTF8);
  SetConsoleCP(CP_UTF8);
#endif
  CommandLineParser parser;
  parser.AddArgument(
      "help", 'h', std::string("Output of information about program arguments"),
      true);
  parser.AddArgument(
      "function", 'f',
      std::string("Sets the executable function. Possible values: Sobel, "
                  "SobelSep, Prewitt, PrewittSep, Gauss, GaussSep"));
  parser.AddArgument("size", 's',
                     std::string("Sets the size of the image in HxW format"));
  parser.AddArgument("count", 'c',
                     std::string("Sets the number of images to test"));
  parser.AddArgument("shared_memory", 'm',
                     std::string("Use shared memory for CUDA operations"),
                     true);
  parser.AddArgument(
      "async", 'a',
      std::string(
          "Use asynchronous CUDA API (create stream and use async methods)"),
      true);
  parser.AddArgument("overlap", 'o',
                     std::string("Enable simple overlap demo (requires -a)"),
                     true);
  parser.AddArgument("gauss_size", '\0', "Gaussian kernel size", false, "3");
  parser.AddArgument("gauss_sigma", '\0', "Gaussian kernel sigma", false, "1");
  try {
    parser.Parse(argc, argv);
    if (parser.Has("help") || parser.Has("h")) {
      std::cout << parser.Help();
      return 0;
    }
    if ((parser.Get("function") == "" && parser.Get("f") == "") ||
        (parser.Get("size") == "" && parser.Get("s") == "") ||
        (parser.Get("count") == "" && parser.Get("c") == "")) {
      throw std::invalid_argument("Specify all the necessary arguments");
    }
    Functions function =
        CheckFunctionArg(parser.Get("function") != "" ? parser.Get("function")
                                                      : parser.Get("f"));
    std::pair<size_t, size_t> size = CheckSizeArg(
        parser.Get("size") != "" ? parser.Get("size") : parser.Get("s"));
    size_t count = ChechCountArg(parser.Get("count") != "" ? parser.Get("count")
                                                           : parser.Get("c"));
    bool use_shared_memory = parser.Has("shared_memory") || parser.Has("m");
    size_t gauss_size = 0;
    float sigma = 0;
    if (function == Functions::Gauss || function == Functions::GaussSep) {
      if (parser.Get("gauss_size") == "" || parser.Get("gauss_sigma") == "") {
        throw std::invalid_argument("Specify all the necessary arguments");
      }
      gauss_size = CheckGaussSize(parser.Get("gauss_size"));
      sigma = ChechGaussSigma(parser.Get("gauss_sigma"));
    }
    std::cout << "Function: " << function << "\nImage size: " << size.first
              << "x" << size.second << "\nImage count: " << count
              << "\nUse shared memory: " << (use_shared_memory ? "Yes" : "No")
              << "\nGaussian kernel size: " << gauss_size
              << "\nGaussian kernel sigma: " << sigma << std::endl;
    TIFFImage image;
    image.SetImage(
        size.second, size.first,
        (uint16_t*)calloc(size.first * size.second, sizeof(uint16_t)));
    switch (function) {
      case Functions::Sobel:
      case Functions::Prewitt:
        image.SetImagePatametersForDevice(ImageOperation::Sobel |
                                          ImageOperation::Prewitt);
        break;
      case Functions::SobelSep:
      case Functions::PrewittSep:
        image.SetImagePatametersForDevice(ImageOperation::Sobel |
                                          ImageOperation::Prewitt |
                                          ImageOperation::Separated);
        break;
      case Functions::Gauss:
        image.SetImagePatametersForDevice(ImageOperation::GaussianBlur,
                                          gauss_size, sigma);
        break;
      case Functions::GaussSep:
        image.SetImagePatametersForDevice(ImageOperation::GaussianBlurSep,
                                          gauss_size, sigma);
        break;
    }
    bool use_async = parser.Has("async") || parser.Has("a");
    bool use_overlap = parser.Has("overlap") || parser.Has("o");
    image.AllocateDeviceMemory();
    if (use_async) {
      // create stream and use async API
      if (use_overlap) {
        // simple double-buffer overlap demo: two streams, alternate buffers
        cudaStream_t streams[2] = {nullptr, nullptr};
        if (cudaSuccess != cudaStreamCreate(&streams[0]) ||
            cudaSuccess != cudaStreamCreate(&streams[1])) {
          std::cerr << "Warning: failed to create CUDA streams for overlap "
                       "demo. Falling back to single-stream async."
                    << std::endl;
          use_overlap = false;
        } else {
          // set first stream on image (functions forward stream from TIFFImage)
          image.SetCudaStream(streams[0]);
          // We'll manually alternate streams when copying/launching kernels by
          // setting the TIFFImage stream before each operation.

          // timings
          using Clock = std::chrono::high_resolution_clock;
          std::chrono::duration<double, std::milli> total_time{0};
          std::chrono::duration<double, std::milli> upload_time{0};
          std::chrono::duration<double, std::milli> compute_time{0};

          for (size_t i = 0; i < count; i++) {
            int slot = i % 2;  // 0 or 1
            cudaStream_t s = streams[slot];
            image.SetCudaStream(s);

            auto t0 = Clock::now();
            image.CopyImageToDeviceAsyncSlot(slot);
            auto t1 = Clock::now();
            upload_time += (t1 - t0);

            // launch kernel on same stream
            auto t2 = Clock::now();
            switch (function) {
              case Functions::Sobel:
                image.SetKernelCudaAsyncSlot(kKernelSobel, use_shared_memory,
                                             true, slot);
                break;
              case Functions::SobelSep:
                image.SetKernelSobelSepCudaAsyncSlot(use_shared_memory, slot);
                break;
              case Functions::Prewitt:
                image.SetKernelCudaAsyncSlot(kKernelPrewitt, use_shared_memory,
                                             true, slot);
                break;
              case Functions::PrewittSep:
                image.SetKernelPrewittSepCudaAsyncSlot(use_shared_memory, slot);
                break;
              case Functions::Gauss:
                image.GaussianBlurCuda(gauss_size, sigma);
                break;
              case Functions::GaussSep:
                image.GaussianBlurSepCuda(gauss_size, sigma);
                break;
            }
            auto t3 = Clock::now();
            compute_time += (t3 - t2);

            // don't synchronize here - we rely on stream ordering for each slot
            total_time += (t3 - t0);
          }

          // synchronize both streams to ensure all work is done
          cudaStreamSynchronize(streams[0]);
          cudaStreamSynchronize(streams[1]);

          // finalize: get result from both slots
          (void)image.CopyImageFromDeviceAsyncSlot(0);
          (void)image.CopyImageFromDeviceAsyncSlot(1);
          image.FreeDeviceMemory();

          // print timings
          std::cout << "Overlap demo (2 streams) timings (ms):\n";
          std::cout << "  total (accumulated): " << total_time.count() << "\n";
          std::cout << "  upload (accumulated): " << upload_time.count()
                    << "\n";
          std::cout << "  compute (accumulated): " << compute_time.count()
                    << "\n";

          cudaStreamDestroy(streams[0]);
          cudaStreamDestroy(streams[1]);
        }
      }
      if (!use_overlap) {
        // single-stream async path
        cudaStream_t stream = nullptr;
        if (cudaSuccess != cudaStreamCreate(&stream)) {
          std::cerr << "Warning: failed to create CUDA stream. Falling back to "
                       "synchronous path."
                    << std::endl;
          use_async = false;
        } else {
          image.SetCudaStream(stream);
          // timings
          using Clock = std::chrono::high_resolution_clock;
          std::chrono::duration<double, std::milli> compute_time{0};
          auto t_total_start = Clock::now();

          image.CopyImageToDeviceAsync();
          for (size_t i = 0; i < count; i++) {
            auto t1 = Clock::now();
            switch (function) {
              case Functions::Sobel:
                image.SetKernelCudaAsync(kKernelSobel, use_shared_memory);
                break;
              case Functions::SobelSep:
                image.SetKernelSobelSepCudaAsync(use_shared_memory);
                break;
              case Functions::Prewitt:
                image.SetKernelCudaAsync(kKernelPrewitt, use_shared_memory);
                break;
              case Functions::PrewittSep:
                image.SetKernelPrewittSepCudaAsync(use_shared_memory);
                break;
              case Functions::Gauss:
                image.GaussianBlurCuda(gauss_size, sigma);
                break;
              case Functions::GaussSep:
                image.GaussianBlurSepCuda(gauss_size, sigma);
                break;
            }
            auto t2 = Clock::now();
            compute_time += (t2 - t1);
          }
          // wait all
          cudaStreamSynchronize(stream);
          (void)image.CopyImageFromDeviceAsync();
          image.FreeDeviceMemory();
          auto t_total_end = Clock::now();
          std::chrono::duration<double, std::milli> total_time =
              (t_total_end - t_total_start);
          std::cout << "Async single-stream timings (ms): total="
                    << total_time.count()
                    << ", compute_accum=" << compute_time.count() << "\n";
          cudaStreamDestroy(stream);
        }
      }
    }
    if (!use_async) {
      // synchronous path
      image.CopyImageToDevice();
      for (size_t i = 0; i < count; i++) {
        switch (function) {
          case Functions::Sobel:
            image.SetKernelCuda(kKernelSobel, use_shared_memory);
            break;
          case Functions::SobelSep:
            image.SetKernelSobelSepCuda(use_shared_memory);
            break;
          case Functions::Prewitt:
            image.SetKernelCuda(kKernelPrewitt, use_shared_memory);
            break;
          case Functions::PrewittSep:
            image.SetKernelPrewittSepCuda(use_shared_memory);
            break;
          case Functions::Gauss:
            image.GaussianBlurCuda(gauss_size, sigma);
            break;
          case Functions::GaussSep:
            image.GaussianBlurSepCuda(gauss_size, sigma);
            break;
        }
      }
      image.FreeDeviceMemory();
    }
    std::cout << "END" << std::endl;
  } catch (const std::exception& e) {
    std::cerr << "Error: " << e.what() << "\n\n";
    std::cerr << parser.Help();
    return 1;
  }
  return 0;
}
