/**
 * @file command_line_parser.h
 * @brief Заголовочный файл класса для парсинга аргументов командной строки.
 */

#pragma once

#include <algorithm>
#include <iomanip>
#include <sstream>
#include <stdexcept>
#include <string>
#include <tuple>
#include <unordered_map>
#include <vector>

/**
 * @class CommandLineParser
 * @brief Класс для парсинга аргументов командной строки.
 *
 * Поддерживает флаги, опции со значениями и позиционные аргументы.
 */
class CommandLineParser {
 private:
  /// Структура для хранения информации об аргументе
  struct Argument {
    std::string about;          ///< Информация об аргумента
    bool is_flag;               ///< Флаг (не требует значения)
    std::string default_value;  ///< Значение по умолчанию
    std::string value;          ///< Установленное значение
  };

  std::unordered_map<std::string, Argument>
      arguments_;                             ///< Хранилище аргументов
  std::vector<std::string> argument_order_;   ///< Порядок добавления аргументов
  std::vector<std::string> positional_args_;  ///< Позиционные аргументы
  std::string program_name_;                  ///< Имя программы
 public:
  /**
   * @brief Конструктор по умолчанию.
   */
  CommandLineParser() = default;

  /**
   * @brief Добавляет аргумент с поддержкой короткой и длинной формы
   * @param long_name Длинное имя аргумента
   * @param short_name Короткое имя аргумента
   * @param about Информация об аргументе (по умолчанию
   * пустая строка).
   * @param is_flag Если true, аргумент не требует значения (по умолчанию
   * false).
   * @param default_value Значение по умолчанию (по умолчанию
   * пустая строка).
   * @throws std::invalid_argument Если имя аргумента пустое.
   */
  void AddArgument(const std::string& long_name, char short_name = '\0',
                   const std::string& about = "", bool is_flag = false,
                   const std::string& default_value = "");

  /**
   * @brief Парсит аргументы командной строки.
   *
   * @param argc Количество аргументов (из main()).
   * @param argv Массив аргументов (из main()).
   * @throws std::runtime_error При ошибках парсинга (неизвестные аргументы,
   * пропущенные значения и т.д.).
   */
  void Parse(int argc, char* argv[]) noexcept(false);

  /**
   * @brief Получает значение не-флагового аргумента.
   *
   * @param name Имя аргумента.
   * @return Значение аргумента в виде строки.
   * @throws std::runtime_error Если аргумент не найден.
   */
  std::string Get(const std::string& name) const;

  /**
   * @brief Проверяет наличие флагового аргумента.
   *
   * @param name Имя флага.
   * @return true если флаг присутствует в командной строке.
   * @throws std::runtime_error Если аргумент не найден.
   */
  bool Has(const std::string& name) const;

  /**
   * @brief Получает все позиционные аргументы (без - или -- префикса).
   *
   * @return Константная ссылка на вектор позиционных аргументов.
   */
  const std::vector<std::string>& GetPositionalArgs() const;

  /**
   * @brief Получает имя программы (argv[0]).
   *
   * @return Имя программы в виде строки.
   */
  std::string GetProgramName() const;

  /**
   * @brief Генерирует справочное сообщение о доступных аргументах.
   *
   * @return Отформатированное справочное сообщение.
   */
  std::string Help() const;
};
