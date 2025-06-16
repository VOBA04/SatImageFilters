/**
 * @file command_line_parser.h
 * @brief ������������ ���� ������ ��� �������� ���������� ��������� ������.
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
 * @brief ����� ��� �������� ���������� ��������� ������.
 *
 * ������������ �����, ����� �� ���������� � ����������� ���������.
 */
class CommandLineParser {
 private:
  /// ��������� ��� �������� ���������� �� ���������
  struct Argument {
    std::string about;          ///< ���������� �� ���������
    bool is_flag;               ///< ���� (�� ������� ��������)
    std::string default_value;  ///< �������� �� ���������
    std::string value;          ///< ������������� ��������
  };

  std::unordered_map<std::string, Argument>
      arguments_;                             ///< ��������� ����������
  std::vector<std::string> argument_order_;   ///< ������� ���������� ����������
  std::vector<std::string> positional_args_;  ///< ����������� ���������
  std::string program_name_;                  ///< ��� ���������
 public:
  /**
   * @brief ����������� �� ���������.
   */
  CommandLineParser() = default;

  /**
   * @brief ��������� �������� � ���������� �������� � ������� �����
   * @param long_name ������� ��� ���������
   * @param short_name �������� ��� ���������
   * @param about ���������� �� ��������� (�� ���������
   * ������ ������).
   * @param is_flag ���� true, �������� �� ������� �������� (�� ���������
   * false).
   * @param default_value �������� �� ��������� (�� ���������
   * ������ ������).
   * @throws std::invalid_argument ���� ��� ��������� ������.
   */
  void AddArgument(const std::string& long_name, char short_name = '\0',
                   const std::string& about = "", bool is_flag = false,
                   const std::string& default_value = "");

  /**
   * @brief ������ ��������� ��������� ������.
   *
   * @param argc ���������� ���������� (�� main()).
   * @param argv ������ ���������� (�� main()).
   * @throws std::runtime_error ��� ������� �������� (����������� ���������,
   * ����������� �������� � �.�.).
   */
  void Parse(int argc, char* argv[]) noexcept(false);

  /**
   * @brief �������� �������� ��-��������� ���������.
   *
   * @param name ��� ���������.
   * @return �������� ��������� � ���� ������.
   * @throws std::runtime_error ���� �������� �� ������.
   */
  std::string Get(const std::string& name) const;

  /**
   * @brief ��������� ������� ��������� ���������.
   *
   * @param name ��� �����.
   * @return true ���� ���� ������������ � ��������� ������.
   * @throws std::runtime_error ���� �������� �� ������.
   */
  bool Has(const std::string& name) const;

  /**
   * @brief �������� ��� ����������� ��������� (��� - ��� -- ��������).
   *
   * @return ����������� ������ �� ������ ����������� ����������.
   */
  const std::vector<std::string>& GetPositionalArgs() const;

  /**
   * @brief �������� ��� ��������� (argv[0]).
   *
   * @return ��� ��������� � ���� ������.
   */
  std::string GetProgramName() const;

  /**
   * @brief ���������� ���������� ��������� � ��������� ����������.
   *
   * @return ����������������� ���������� ���������.
   */
  std::string Help() const;
};
