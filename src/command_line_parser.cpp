#include "command_line_parser.h"

void CommandLineParser::AddArgument(const std::string& long_name,
                                    char short_name, const std::string& about,
                                    bool is_flag,
                                    const std::string& default_value) {
  if (long_name.empty() && short_name == '\0') {
    throw std::invalid_argument("At least one name must be provided");
  }
  Argument arg{about, is_flag, default_value, ""};
  if (short_name != '\0') {
    arguments_["-" + std::string(1, short_name)] = arg;
    argument_order_.push_back("-" + std::string(1, short_name));
  }
  if (!long_name.empty()) {
    arguments_["--" + long_name] = arg;
    if (short_name == '\0') {
      argument_order_.push_back("-" + std::string(1, short_name));
    }
  }
}

void CommandLineParser::Parse(int argc, char* argv[]) {
  if (argc < 1) return;
  program_name_ = argv[0];
  for (int i = 1; i < argc; ++i) {
    std::string arg = argv[i];
    if (arg.substr(0, 2) == "--") {
      std::string option = arg.substr(2);
      if (option.empty()) {
        throw std::runtime_error("Invalid argument: " + arg);
      }
      auto it = arguments_.find(option);
      if (it == arguments_.end()) {
        throw std::runtime_error("Unknown argument: " + arg);
      }
      if (it->second.is_flag) {
        it->second.value = "true";
      } else {
        if (i + 1 >= argc) {
          throw std::runtime_error("Missing value for argument: " + arg);
        }
        it->second.value = argv[++i];
      }
    } else if (arg[0] == '-') {
      if (arg.length() != 2) {
        throw std::runtime_error("Invalid short argument: " + arg);
      }
      char option = arg[1];
      std::string option_str(1, option);
      auto it = arguments_.find(option_str);
      if (it == arguments_.end()) {
        throw std::runtime_error("Unknown argument: " + arg);
      }
      if (it->second.is_flag) {
        it->second.value = "true";
      } else {
        if (i + 1 >= argc) {
          throw std::runtime_error("Missing value for argument: " + arg);
        }
        it->second.value = argv[++i];
      }
    } else {
      positional_args_.push_back(arg);
    }
  }
  for (auto& pair : arguments_) {
    if (pair.second.value.empty() && !pair.second.default_value.empty()) {
      pair.second.value = pair.second.default_value;
    }
  }
}

std::string CommandLineParser::Get(const std::string& name) const {
  auto it = arguments_.find(name);
  if (it == arguments_.end()) {
    throw std::runtime_error("Argument not found: " + name);
  }
  return it->second.value;
}

bool CommandLineParser::Has(const std::string& name) const {
  auto it = arguments_.find(name);
  if (it == arguments_.end()) {
    throw std::runtime_error("Argument not found: " + name);
  }
  return it->second.value == "true";
}

const std::vector<std::string>& CommandLineParser::GetPositionalArgs() const {
  return positional_args_;
}

std::string CommandLineParser::GetProgramName() const { return program_name_; }

std::string CommandLineParser::Help() const {
  std::ostringstream oss;
  oss << "Usage: " << program_name_ << " [options]\n\nOptions:\n";
  std::vector<std::tuple<std::string, std::string, std::string>> help_entries;
  size_t max_option_len = 0;
  for (const auto& name : argument_order_) {
    const auto& arg = arguments_.at(name);
    std::vector<std::string> option_names = {name};
    for (const auto& pair : arguments_) {
      if (pair.second.about == arg.about &&
          pair.second.is_flag == arg.is_flag &&
          pair.second.default_value == arg.default_value &&
          pair.first != name) {
        option_names.push_back(pair.first);
      }
    }
    std::sort(option_names.begin(), option_names.end(),
              [](const auto& a, const auto& b) {
                return a.length() < b.length() ||
                       (a.length() == b.length() && a < b);
              });
    option_names.erase(std::unique(option_names.begin(), option_names.end()),
                       option_names.end());
    std::string option_str;
    for (size_t i = 0; i < option_names.size(); ++i) {
      if (i != 0) option_str += ", ";
      option_str += option_names[i];
    }
    std::string type_str =
        arg.is_flag
            ? "Flag"
            : "Option (default: " +
                  (arg.default_value.empty() ? "none" : arg.default_value) +
                  ")";
    help_entries.emplace_back(option_str, type_str, arg.about);
    max_option_len = std::max(max_option_len, option_str.length());
  }
  const size_t total_width = max_option_len + 8;
  for (const auto& [option_str, type_str, about] : help_entries) {
    oss << "  " << std::left << std::setw(total_width) << option_str;
    if (!about.empty()) {
      oss << type_str << " - " << about;
    } else {
      oss << type_str;
    }
    oss << "\n";
  }
  return oss.str();
}
