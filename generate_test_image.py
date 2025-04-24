import cv2
import numpy as np

# Создаем изображение 100x100
width, height = 100, 100
img = np.zeros((height, width), dtype=np.uint16)
img[:, width // 2 :] = 65535  # Правая половина белая

# Сохраняем как 16-битное TIFF
cv2.imwrite("tests/test_image.tiff", img)
