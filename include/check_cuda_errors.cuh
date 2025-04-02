/**
 * @file check_cuda_errors.cuh
 * @brief Заголовочный файл для обработки ошибок CUDA.
 *
 * Этот файл содержит макрос и функцию для проверки и обработки ошибок,
 * возникающих при вызове CUDA API. Если вызов CUDA API завершился с ошибкой,
 * программа завершает выполнение с выводом сообщения об ошибке.
 */

#include <cuda_runtime.h>
#include <iostream>

/**
 * @brief Макрос для проверки ошибок CUDA.
 *
 * @param val Результат вызова CUDA API, который необходимо проверить.
 *
 * Макрос вызывает функцию Check, передавая ей результат вызова CUDA API,
 * строковое представление вызова, имя файла и номер строки, где произошел
 * вызов. Если произошла ошибка, программа завершится с выводом сообщения об
 * ошибке.
 */
#define checkCudaErrors(val) Check((val), #val, __FILE__, __LINE__)

/**
 * @brief Функция для обработки ошибок CUDA.
 *
 * @tparam T Тип возвращаемого значения CUDA API (например, cudaError_t).
 * @param err Результат вызова CUDA API.
 * @param func Строковое представление вызова CUDA API.
 * @param file Имя файла, где произошел вызов.
 * @param line Номер строки, где произошел вызов.
 *
 * Если результат вызова CUDA API указывает на ошибку, функция выводит сообщение
 * об ошибке, содержащее имя файла, номер строки, описание ошибки и вызванную
 * функцию. После этого программа завершает выполнение.
 */
template <typename T>
void Check(T err, const char* const func, const char* const file,
           const int line) {
  if (err != cudaSuccess) {
    std::cerr << "CUDA error at: " << file << ":" << line << std::endl;
    std::cerr << cudaGetErrorString(err) << " " << func << std::endl;
    exit(1);
  }
}