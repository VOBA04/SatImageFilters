#include <string>

#include "command_line_parser.h"
#include "tiff_image.h"
#ifdef _WIN32
#include <windows.h>
#endif

int main(int argc, char* argv[]) {
#ifdef _WIN32
  SetConsoleOutputCP(CP_UTF8);
  SetConsoleCP(CP_UTF8);
#endif
  CommandLineParser parser;
  parser.AddArgument("help", 'h',
                     std::string("Вывод информации об аргументах программы"),
                     true);
  parser.AddArgument(
      "function", 'f',
      std::string("Задает исполняемую функцию. Возможные значения: Sobel, "
                  "SobelSep, Prewitt, PrewittSep, Gauss, GaussSep"));
  parser.AddArgument("size", 's',
                     std::string("Задает размер изображения в формате HxW"));
  parser.AddArgument(
      "count", 'c',
      std::string("Задает количество изображений для тестирования"));
  try {
    parser.Parse(argc, argv);
    if (parser.Has("help") || parser.Has("-h")) {
      std::cout << parser.Help();
      return 0;
    }
  } catch (const std::exception& e) {
    std::cerr << "Error: " << e.what() << "\n\n";
    std::cerr << parser.Help();
    return 1;
  }
  std::cout << parser.Help();
  return 0;
}
