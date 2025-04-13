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

/**
 * @enum ImageOperation
 * @brief Представляет различные операции обработки изображений.
 *
 * - None: Без операции.
 * - Sobel: Оператор Собеля для выделения границ.
 * - Prewitt: Оператор Прюитта для выделения границ.
 * - GaussianBlur: Фильтр размытия по Гауссу.
 * - Separated: Разделённая операция фильтрации.
 */
enum class ImageOperation {
  None = 0,          ///< Без операции.
  Sobel = 1,         ///< Оператор Собеля.
  Prewitt = 2,       ///< Оператор Прюитта.
  GaussianBlur = 4,  ///< Размытие по Гауссу.
  Separated = 8      ///< Разделённая фильтрация.
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