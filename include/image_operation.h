#pragma once

enum class ImageOperation {
  None = 0,
  Sobel = 1,
  Prewitt = 2,
  GaussianBlur = 4,
  Separated = 8
};

constexpr ImageOperation operator|(ImageOperation lhs, ImageOperation rhs) {
  return static_cast<ImageOperation>(static_cast<int>(lhs) |
                                     static_cast<int>(rhs));
}