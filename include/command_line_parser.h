/**
 * @file command_line_parser.h
 * @brief Парсер командной строки.
 */

#pragma once

#include <string>
#include <unordered_map>
#include <vector>

/**
 * @class CommandLineParser
 * @brief Класс для парсинга командной строки.
 *
 * Содержит методы для анализа и обработки аргументов.
 */
class CommandLineParser {
 private:
  /// Структура для хранения информации об аргументе
  struct Argument {
    std::string about;          ///< Описание аргумента
    bool is_flag;               ///< Является ли аргумент флагом (без значения)
    std::string default_value;  ///< Значение по умолчанию
    std::string value;          ///< Текущее значение
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
   * @brief Добавляет аргумент в парсер с указанием имени и свойств.
   * @param long_name Длинное имя аргумента
   * @param short_name Короткое имя аргумента
   * @param about Описание аргумента (для справки)
   * @param is_flag Если true, аргумент является флагом (без значения)
   * @param default_value Значение по умолчанию
   * @throws std::invalid_argument Если аргумент уже существует.
   */
  void AddArgument(const std::string& long_name, char short_name = '\0',
                   const std::string& about = "", bool is_flag = false,
                   const std::string& default_value = "");

  /**
   * @brief Парсит переданные аргументы командной строки.
   *
   * @param argc Количество аргументов (из main()).
   * @param argv Массив аргументов (из main()).
   * @throws std::runtime_error При ошибке парсинга (неизвестные аргументы,
   * некорректные значения и т.д.).
   */
  void Parse(int argc, char* argv[]) noexcept(false);

  /**
   * @brief Получает значение аргумента по имени.
   *
   * @param name Имя аргумента.
   * @return Значение аргумента или пустую строку.
   * @throws std::runtime_error Если аргумент не найден.
   */
  std::string Get(const std::string& name) const;

  /**
   * @brief Проверяет наличие аргумента.
   *
   * @param name Имя аргумента.
   * @return true, если аргумент присутствует в парсере.
   * @throws std::runtime_error Если аргумент не найден.
   */
  bool Has(const std::string& name) const;

  /**
   * @brief Получает список позиционных аргументов (не - или -- аргументы).
   *
   * @return Список позиционных аргументов из командной строки.
   */
  const std::vector<std::string>& GetPositionalArgs() const;

  /**
   * @brief Получает имя программы (argv[0]).
   *
   * @return Имя программы или пустую строку.
   */
  std::string GetProgramName() const;

  /**
   * @brief Генерирует строку с помощью для аргументов.
   *
   * @return Форматированная строка с описанием аргументов.
   */
  std::string Help() const;
};