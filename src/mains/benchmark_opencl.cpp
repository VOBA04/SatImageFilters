#include <algorithm>
#include <chrono>
#include <exception>
#include <iostream>
#include <map>
#include <stdexcept>
#include <string>
#include <utility>
#include <vector>

#include "command_line_parser.h"
#include "kernel.h"
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

static Functions CheckFunctionArg(const std::string& arg) {
  const std::vector<std::string> args = {"Sobel",      "SobelSep", "Prewitt",
                                         "PrewittSep", "Gauss",    "GaussSep"};
  auto it = std::find(args.begin(), args.end(), arg);
  if (it == args.end()) {
    throw std::invalid_argument("Wrong function argument: " + arg);
  }
  return static_cast<Functions>(it - args.begin());
}

static std::pair<size_t, size_t> CheckSizeArg(const std::string& arg) {
  auto pos = arg.find('x');
  if (pos == std::string::npos) {
    throw std::invalid_argument("Wrong size argument: " + arg);
  }
  int h = std::stoi(arg.substr(0, pos));
  int w = std::stoi(arg.substr(pos + 1));
  if (h <= 0 || w <= 0) {
    throw std::invalid_argument("Size must be positive: " + arg);
  }
  return {static_cast<size_t>(h), static_cast<size_t>(w)};
}

static size_t CheckGaussSize(const std::string& arg) {
  return std::stoul(arg);
}
static float CheckGaussSigma(const std::string& arg) {
  return std::stof(arg);
}
static size_t CheckCountArg(const std::string& arg) {
  return std::stoul(arg);
}

int main(int argc, char* argv[]) {
#ifdef _WIN32
  SetConsoleOutputCP(CP_UTF8);
  SetConsoleCP(CP_UTF8);
#endif
  CommandLineParser parser;
  parser.AddArgument("help", 'h',
                     "Output of information about program arguments", true);
  parser.AddArgument("function", 'f',
                     "Sets the executable function. Possible values: Sobel, "
                     "SobelSep, Prewitt, PrewittSep, Gauss, GaussSep");
  parser.AddArgument("size", 's', "Sets the size of the image in HxW format");
  parser.AddArgument("count", 'c', "Sets the number of images to test");
  parser.AddArgument("shared_memory", 'm',
                     "Use local memory for OpenCL operations (3x3 only)", true);
  parser.AddArgument("gauss_size", '\0', "Gaussian kernel size", false, "3");
  parser.AddArgument("gauss_sigma", '\0', "Gaussian kernel sigma", false, "1");

  try {
    parser.Parse(argc, argv);
    if (parser.Has("help") || parser.Has("h")) {
      std::cout << parser.Help();
      return 0;
    }
    if ((parser.Get("function").empty() && parser.Get("f").empty()) ||
        (parser.Get("size").empty() && parser.Get("s").empty()) ||
        (parser.Get("count").empty() && parser.Get("c").empty())) {
      throw std::invalid_argument("Specify all the necessary arguments");
    }
    const Functions function = CheckFunctionArg(!parser.Get("function").empty()
                                                    ? parser.Get("function")
                                                    : parser.Get("f"));
    const auto size = CheckSizeArg(
        !parser.Get("size").empty() ? parser.Get("size") : parser.Get("s"));
    const size_t count = CheckCountArg(
        !parser.Get("count").empty() ? parser.Get("count") : parser.Get("c"));
    const bool use_local = parser.Has("shared_memory") || parser.Has("m");
    size_t gauss_size = 0;
    float sigma = 0.0f;
    if (function == Functions::Gauss || function == Functions::GaussSep) {
      if (parser.Get("gauss_size").empty() ||
          parser.Get("gauss_sigma").empty()) {
        throw std::invalid_argument("Specify all the necessary arguments");
      }
      gauss_size = CheckGaussSize(parser.Get("gauss_size"));
      sigma = CheckGaussSigma(parser.Get("gauss_sigma"));
    }

    TIFFImage image;
    std::vector<uint16_t> buffer(size.first * size.second, 0);
    image.SetImage(size.second, size.first, buffer.data());

    switch (function) {
      case Functions::Sobel:
      case Functions::Prewitt:
        image.SetImagePatametersForOpenCLOps(
            ImageOperation::Sobel | ImageOperation::Prewitt, 0, 0.0f);
        break;
      case Functions::SobelSep:
      case Functions::PrewittSep:
        image.SetImagePatametersForOpenCLOps(ImageOperation::Sobel |
                                                 ImageOperation::Prewitt |
                                                 ImageOperation::Separated,
                                             0, 0.0f);
        break;
      case Functions::Gauss:
        image.SetImagePatametersForOpenCLOps(ImageOperation::GaussianBlur,
                                             gauss_size, sigma);
        break;
      case Functions::GaussSep:
        image.SetImagePatametersForOpenCLOps(ImageOperation::GaussianBlurSep,
                                             gauss_size, sigma);
        break;
    }
    image.AllocateOpenCLMemory();
    image.CopyImageToOpenCLDevice();

    image.OpenCLProfilingEnable(true);
    image.OpenCLProfilingClear();

    using HighResClock = std::chrono::high_resolution_clock;
    const auto t0 = HighResClock::now();
    for (size_t i = 0; i < count; ++i) {
      switch (function) {
        case Functions::Sobel:
          (void)image.SetKernelOpenCL(kKernelSobel, use_local);
          break;
        case Functions::SobelSep:
          (void)image.SetKernelSobelSepOpenCL(use_local);
          break;
        case Functions::Prewitt:
          (void)image.SetKernelOpenCL(kKernelPrewitt, use_local);
          break;
        case Functions::PrewittSep:
          (void)image.SetKernelPrewittSepOpenCL(use_local);
          break;
        case Functions::Gauss:
          (void)image.GaussianBlurOpenCL(gauss_size, sigma);
          break;
        case Functions::GaussSep:
          (void)image.GaussianBlurSepOpenCL(gauss_size, sigma);
          break;
      }
    }
    const auto t1 = HighResClock::now();
    const double wall_ms =
        std::chrono::duration_cast<std::chrono::duration<double, std::milli>>(
            t1 - t0)
            .count();

    // Aggregate per-kernel times
    std::map<std::string, double> by_kernel;
    double device_total_ms = 0.0;
    for (const auto& rec : image.OpenCLProfilingRecords()) {
      by_kernel[rec.name] += rec.ms;
      device_total_ms += rec.ms;
    }

    // Output in a simple parseable format
    std::cout << "OPENCL_PROFILE_BEGIN\n";
    std::cout << "CONFIG,Function," << function << "\n";
    std::cout << "CONFIG,Size," << size.first << "x" << size.second << "\n";
    std::cout << "CONFIG,Count," << count << "\n";
    std::cout << "CONFIG,UseLocal," << (use_local ? 1 : 0) << "\n";
    if (function == Functions::Gauss || function == Functions::GaussSep) {
      std::cout << "CONFIG,GaussSize," << gauss_size << "\n";
      std::cout << "CONFIG,GaussSigma," << sigma << "\n";
    }
    for (const auto& kv : by_kernel) {
      std::cout << "KERNEL," << kv.first << ",MS," << kv.second << "\n";
    }
    std::cout << "TOTAL_DEVICE_MS," << device_total_ms << "\n";
    std::cout << "TOTAL_WALL_MS," << wall_ms << "\n";
    std::cout << "OPENCL_PROFILE_END\n";

    std::cout << "END" << std::endl;
  } catch (const std::exception& e) {
    std::cerr << "Error: " << e.what() << "\n\n";
    std::cerr << parser.Help();
    return 1;
  }
  return 0;
}
