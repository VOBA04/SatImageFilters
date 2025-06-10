/**
 * @file image_operation.h
 * @brief Заголовочный файл, содержащий определения операций обработки
 * изображений.
 *
 * Этот файл содержит перечисление ImageOperation, представляющее различные
 * операции обработки изображений, а также перегрузку оператора | для их
 * комбинирования.
 */

#pragma once

#include <vector>

/**
 * @enum ImageOperation
 * @brief Представляет различные операции обработки изображений.
 *
 * - None: Без операции.
 * - Sobel: Оператор Собеля для выделения границ.
 * - Prewitt: Оператор Прюитта для выделения границ.
 * - GaussianBlur: Фильтр размытия по Гауссу.
 * - GaussianBlurSep: Разделенное размытие по Гауссу.
 * - Separated: Разделённая операция фильтрации.
 */
enum class ImageOperation {
  None = 0,             ///< Без операции.
  Sobel = 1,            ///< Оператор Собеля.
  Prewitt = 2,          ///< Оператор Прюитта.
  GaussianBlur = 4,     ///< Размытие по Гауссу.
  GaussianBlurSep = 8,  ///< Разделенное размытие по Гаусса
  Separated = 16        ///< Разделённая фильтрация.
};

/**
 * @brief Объединяет два значения ImageOperation с использованием побитового OR.
 *
 * @param lhs Первое значение ImageOperation.
 * @param rhs Второе значение ImageOperation.
 * @return Новое значение ImageOperation, представляющее комбинацию двух
 * значений.
 */
constexpr ImageOperation operator|(ImageOperation lhs, ImageOperation rhs) {
  return static_cast<ImageOperation>(static_cast<int>(lhs) |
                                     static_cast<int>(rhs));
}

/**
 * @brief Разбивает объединенное значение ImageOperation на составляющие.
 *
 * @param ops Комбинированное значение ImageOperation.
 * @return Вектор, содержащий отдельные операции ImageOperation.
 */
inline std::vector<ImageOperation> DecomposeOperations(ImageOperation ops) {
  std::vector<ImageOperation> result;
  int value = static_cast<int>(ops);
  if ((value & static_cast<int>(ImageOperation::Sobel)) != 0) {
    result.push_back(ImageOperation::Sobel);
  }
  if ((value & static_cast<int>(ImageOperation::Prewitt)) != 0) {
    result.push_back(ImageOperation::Prewitt);
  }
  if ((value & static_cast<int>(ImageOperation::GaussianBlur)) != 0) {
    result.push_back(ImageOperation::GaussianBlur);
  }
  if ((value & static_cast<int>(ImageOperation::GaussianBlurSep)) != 0) {
    result.push_back(ImageOperation::GaussianBlurSep);
  }
  if ((value & static_cast<int>(ImageOperation::Separated)) != 0) {
    result.push_back(ImageOperation::Separated);
  }
  if (value == static_cast<int>(ImageOperation::None)) {
    result.push_back(ImageOperation::None);
  }
  return result;
}