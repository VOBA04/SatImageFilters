#include <string>

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
    image.AllocateDeviceMemory();
    image.CopyImageToDevice();
    for (size_t i = 0; i < count; i++) {
      switch (function) {
        case Functions::Sobel:
          image.SetKernelCuda(kKernelSobel);
          break;
        case Functions::SobelSep:
          image.SetKernelSobelSepCuda();
          break;
        case Functions::Prewitt:
          image.SetKernelCuda(kKernelPrewitt);
          break;
        case Functions::PrewittSep:
          image.SetKernelPrewittSepCuda();
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
    std::cout << "END" << std::endl;
  } catch (const std::exception& e) {
    std::cerr << "Error: " << e.what() << "\n\n";
    std::cerr << parser.Help();
    return 1;
  }
  return 0;
}
