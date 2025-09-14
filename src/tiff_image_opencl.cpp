#include "tiff_image.h"

#include <CL/cl.h>
#include <cstdio>
#include <cstring>
#include <fstream>
#include <iostream>
#include <stdexcept>
#include <string>

#include "kernel.h"

namespace {
constexpr size_t kBlockSize = 16;

inline void CheckCLError(cl_int err, const char* where) {
  if (err != CL_SUCCESS) {
    char buf[256];
    std::snprintf(buf, sizeof(buf), "OpenCL error %d at %s", err, where);
    throw std::runtime_error(buf);
  }
}

std::string ReadTextFile(const std::string& path) {
  std::ifstream ifs(path);
  if (!ifs) {
    throw std::runtime_error("Не удалось открыть файл: " + path);
  }
  std::string content((std::istreambuf_iterator<char>(ifs)),
                      std::istreambuf_iterator<char>());
  return content;
}
}  // namespace

void TIFFImage::EnsureOpenCLInitialized() {
  if (cl_context_ != nullptr && cl_queue_ != nullptr && cl_device_ != nullptr) {
    return;
  }
  cl_int err;
  cl_uint num_platforms = 0;
  CheckCLError(clGetPlatformIDs(0, nullptr, &num_platforms),
               "clGetPlatformIDs(count)");
  if (num_platforms == 0) {
    throw std::runtime_error("OpenCL платформа не найдена");
  }
  std::vector<cl_platform_id> platforms(num_platforms);
  CheckCLError(clGetPlatformIDs(num_platforms, platforms.data(), nullptr),
               "clGetPlatformIDs(list)");

  cl_device_id device = nullptr;
  for (auto pid : platforms) {
    cl_uint num_devices = 0;
    cl_int res =
        clGetDeviceIDs(pid, CL_DEVICE_TYPE_GPU, 0, nullptr, &num_devices);
    if (res == CL_SUCCESS && num_devices > 0) {
      std::vector<cl_device_id> devs(num_devices);
      CheckCLError(clGetDeviceIDs(pid, CL_DEVICE_TYPE_GPU, num_devices,
                                  devs.data(), nullptr),
                   "clGetDeviceIDs(GPU)");
      device = devs[0];
      cl_context_ =
          clCreateContext(nullptr, 1, &device, nullptr, nullptr, &err);
      CheckCLError(err, "clCreateContext(GPU)");
#if defined(CL_VERSION_2_0)
      cl_queue_ = clCreateCommandQueueWithProperties(cl_context_, device,
                                                     nullptr, &err);
#else
      cl_queue_ = clCreateCommandQueue(cl_context_, device, 0, &err);
#endif
      CheckCLError(err, "clCreateCommandQueue");
      cl_device_ = device;
      return;
    }
  }
  // GPU не найден, пробуем CPU
  for (auto pid : platforms) {
    cl_uint num_devices = 0;
    cl_int res =
        clGetDeviceIDs(pid, CL_DEVICE_TYPE_CPU, 0, nullptr, &num_devices);
    if (res == CL_SUCCESS && num_devices > 0) {
      std::vector<cl_device_id> devs(num_devices);
      CheckCLError(clGetDeviceIDs(pid, CL_DEVICE_TYPE_CPU, num_devices,
                                  devs.data(), nullptr),
                   "clGetDeviceIDs(CPU)");
      device = devs[0];
      cl_context_ =
          clCreateContext(nullptr, 1, &device, nullptr, nullptr, &err);
      CheckCLError(err, "clCreateContext(CPU)");
#if defined(CL_VERSION_2_0)
      cl_queue_ = clCreateCommandQueueWithProperties(cl_context_, device,
                                                     nullptr, &err);
#else
      cl_queue_ = clCreateCommandQueue(cl_context_, device, 0, &err);
#endif
      CheckCLError(err, "clCreateCommandQueue");
      cl_device_ = device;
      return;
    }
  }
  throw std::runtime_error("Не удалось выбрать устройство OpenCL");
}

void TIFFImage::EnsureOpenCLProgramBuilt() {
  EnsureOpenCLInitialized();
  if (cl_program_ != nullptr)
    return;
  cl_int err;
  std::string kernel_path =
      std::string(PROJECT_SOURCE_DIR) + "/src/kernels/kernels.cl";
  std::string source = ReadTextFile(kernel_path);
  const char* src = source.c_str();
  size_t len = source.size();
  cl_program_ = clCreateProgramWithSource(cl_context_, 1, &src, &len, &err);
  CheckCLError(err, "clCreateProgramWithSource");
  err = clBuildProgram(cl_program_, 1, &cl_device_, "", nullptr, nullptr);
  if (err != CL_SUCCESS) {
    // Получим лог сборки
    size_t log_size = 0;
    clGetProgramBuildInfo(cl_program_, cl_device_, CL_PROGRAM_BUILD_LOG, 0,
                          nullptr, &log_size);
    std::string log(log_size, '\0');
    clGetProgramBuildInfo(cl_program_, cl_device_, CL_PROGRAM_BUILD_LOG,
                          log_size, log.data(), nullptr);
    std::cerr << "OpenCL build log:\n" << log << std::endl;
    CheckCLError(err, "clBuildProgram");
  }
}

void TIFFImage::ReleaseOpenCLProgram() {
  if (cl_program_) {
    clReleaseProgram(cl_program_);
    cl_program_ = nullptr;
  }
}

void TIFFImage::ReleaseOpenCLContext() {
  if (cl_queue_) {
    clReleaseCommandQueue(cl_queue_);
    cl_queue_ = nullptr;
  }
  if (cl_context_) {
    clReleaseContext(cl_context_);
    cl_context_ = nullptr;
  }
  cl_device_ = nullptr;
}

void TIFFImage::EnsureGaussianKernelBuffer(size_t kernel_size, float sigma) {
  if (cl_gaussian_kernel_ != nullptr &&
      cl_gaussian_kernel_size_ == kernel_size &&
      std::abs(cl_gaussian_sigma_ - sigma) < 1e-6) {
    return;
  }
  // Пере-выделить
  if (cl_gaussian_kernel_) {
    clReleaseMemObject(cl_gaussian_kernel_);
    cl_gaussian_kernel_ = nullptr;
  }
  Kernel<float> kernel =
      Kernel<float>::GetGaussianKernelSep(kernel_size, sigma);
  size_t bytes = kernel.GetHeight() * kernel.GetWidth() * sizeof(float);
  cl_int err;
  cl_gaussian_kernel_ =
      clCreateBuffer(cl_context_, CL_MEM_READ_ONLY, bytes, nullptr, &err);
  CheckCLError(err, "clCreateBuffer(gaussian_kernel)");
  // Копируем на устройство
  float* h_kernel = nullptr;
  kernel.CopyKernelTo(&h_kernel);
  CheckCLError(clEnqueueWriteBuffer(cl_queue_, cl_gaussian_kernel_, CL_TRUE, 0,
                                    bytes, h_kernel, 0, nullptr, nullptr),
               "clEnqueueWriteBuffer(gaussian_kernel)");
  delete[] h_kernel;
  cl_gaussian_kernel_size_ = kernel_size;
  cl_gaussian_sigma_ = sigma;
}

void TIFFImage::SetImagePatametersForOpenCLOps(ImageOperation operations,
                                               size_t gaussian_kernel_size,
                                               float gaussian_sigma) {
  cl_image_operations_ = DecomposeOperations(operations);
  cl_image_size_ = width_ * height_ * sizeof(uint16_t);
  if (((static_cast<int>(operations) &
        static_cast<int>(ImageOperation::GaussianBlur)) != 0) ||
      ((static_cast<int>(operations) &
        static_cast<int>(ImageOperation::GaussianBlurSep)) != 0)) {
    cl_gaussian_kernel_size_ = gaussian_kernel_size;
    cl_gaussian_sigma_ = gaussian_sigma;
  }
}

void TIFFImage::AllocateOpenCLMemory() {
  EnsureOpenCLInitialized();
  if (cl_allocated_)
    return;
  cl_int err;
  cl_src_ = clCreateBuffer(cl_context_, CL_MEM_READ_WRITE, cl_image_size_,
                           nullptr, &err);
  CheckCLError(err, "clCreateBuffer(src)");
  cl_dst_ = clCreateBuffer(cl_context_, CL_MEM_READ_WRITE, cl_image_size_,
                           nullptr, &err);
  CheckCLError(err, "clCreateBuffer(dst)");

  // Доп. буферы для разделённых операций и гаусса
  size_t temps_size_i = width_ * height_ * sizeof(int);
  size_t temps_size_f = width_ * height_ * sizeof(float);

  bool need_sep = false;
  for (auto op : cl_image_operations_) {
    if (op == ImageOperation::Sobel || op == ImageOperation::Prewitt ||
        op == ImageOperation::Separated) {
      need_sep = true;
    }
  }
  if (need_sep) {
    cl_sep_g_x_ = clCreateBuffer(cl_context_, CL_MEM_READ_WRITE, temps_size_i,
                                 nullptr, &err);
    CheckCLError(err, "clCreateBuffer(gx)");
    cl_sep_g_y_ = clCreateBuffer(cl_context_, CL_MEM_READ_WRITE, temps_size_i,
                                 nullptr, &err);
    CheckCLError(err, "clCreateBuffer(gy)");
    cl_sep_result_x_ = clCreateBuffer(cl_context_, CL_MEM_READ_WRITE,
                                      temps_size_i, nullptr, &err);
    CheckCLError(err, "clCreateBuffer(result_x)");
    cl_sep_result_y_ = clCreateBuffer(cl_context_, CL_MEM_READ_WRITE,
                                      temps_size_i, nullptr, &err);
    CheckCLError(err, "clCreateBuffer(result_y)");
  }

  bool need_gauss_temp = false;
  for (auto op : cl_image_operations_) {
    if (op == ImageOperation::GaussianBlurSep)
      need_gauss_temp = true;
  }
  if (need_gauss_temp) {
    cl_gaussian_sep_temp_ = clCreateBuffer(cl_context_, CL_MEM_READ_WRITE,
                                           temps_size_f, nullptr, &err);
    CheckCLError(err, "clCreateBuffer(gaussian_temp)");
  }

  if (cl_gaussian_kernel_size_ > 0) {
    EnsureGaussianKernelBuffer(cl_gaussian_kernel_size_, cl_gaussian_sigma_);
  }
  cl_allocated_ = true;
}

void TIFFImage::ReallocateOpenCLMemory() {
  FreeOpenCLMemory();
  AllocateOpenCLMemory();
}

void TIFFImage::CopyImageToOpenCLDevice() {
  if (!image_)
    throw std::runtime_error("The image was not uploaded");
  EnsureOpenCLInitialized();
  if (!cl_allocated_)
    AllocateOpenCLMemory();
  CheckCLError(
      clEnqueueWriteBuffer(cl_queue_, cl_src_, CL_TRUE, 0, cl_image_size_,
                           image_, 0, nullptr, nullptr),
      "clEnqueueWriteBuffer(src)");
}

void TIFFImage::FreeOpenCLMemory() {
  auto rel = [](cl_mem& m) {
    if (m) {
      clReleaseMemObject(m);
      m = nullptr;
    }
  };
  rel(cl_src_);
  rel(cl_dst_);
  rel(cl_sep_g_x_);
  rel(cl_sep_g_y_);
  rel(cl_sep_result_x_);
  rel(cl_sep_result_y_);
  rel(cl_gaussian_kernel_);
  rel(cl_gaussian_sep_temp_);
  cl_allocated_ = false;
}

static inline size_t RoundUp(size_t value, size_t multiple) {
  return (value + multiple - 1) / multiple * multiple;
}

TIFFImage TIFFImage::SetKernelOpenCL(const Kernel<int>& kernel,
                                     const bool /*shared_memory*/,
                                     const bool rotate) const {
  const_cast<TIFFImage*>(this)->EnsureOpenCLProgramBuilt();
  cl_int err;
  cl_kernel k = nullptr;
  bool generic = false;
  if (kernel == kKernelSobel) {
    k = clCreateKernel(cl_program_, "Sobel", &err);
    CheckCLError(err, "clCreateKernel(Sobel)");
  } else if (kernel == kKernelPrewitt) {
    k = clCreateKernel(cl_program_, "Prewitt", &err);
    CheckCLError(err, "clCreateKernel(Prewitt)");
  } else {
    k = clCreateKernel(cl_program_, "ApplyKernel", &err);
    CheckCLError(err, "clCreateKernel(ApplyKernel)");
    generic = true;
  }

  // Выделение/копирование
  cl_mem src = cl_src_, dst = cl_dst_;
  bool ephemeral = false;
  size_t image_size = width_ * height_ * sizeof(uint16_t);
  if (!cl_allocated_) {
    ephemeral = true;
    src = clCreateBuffer(cl_context_, CL_MEM_READ_WRITE, image_size, nullptr,
                         &err);
    CheckCLError(err, "clCreateBuffer(src tmp)");
    dst = clCreateBuffer(cl_context_, CL_MEM_READ_WRITE, image_size, nullptr,
                         &err);
    CheckCLError(err, "clCreateBuffer(dst tmp)");
    CheckCLError(clEnqueueWriteBuffer(cl_queue_, src, CL_TRUE, 0, image_size,
                                      image_, 0, nullptr, nullptr),
                 "clEnqueueWriteBuffer(src tmp)");
  }

  // Параметры ядра
  int arg = 0;
  CheckCLError(clSetKernelArg(k, arg++, sizeof(cl_mem), &src),
               "clSetKernelArg(src)");
  CheckCLError(clSetKernelArg(k, arg++, sizeof(cl_mem), &dst),
               "clSetKernelArg(dst)");
  size_t h = height_, w = width_;
  CheckCLError(clSetKernelArg(k, arg++, sizeof(size_t), &h),
               "clSetKernelArg(h)");
  CheckCLError(clSetKernelArg(k, arg++, sizeof(size_t), &w),
               "clSetKernelArg(w)");

  cl_mem kernel_buf = nullptr;
  int ksize = static_cast<int>(kernel.GetHeight());
  int rot = rotate ? 1 : 0;
  if (generic) {
    size_t kbytes = kernel.GetHeight() * kernel.GetWidth() * sizeof(int);
    kernel_buf =
        clCreateBuffer(cl_context_, CL_MEM_READ_ONLY, kbytes, nullptr, &err);
    CheckCLError(err, "clCreateBuffer(kernel)");
    int* hkernel = nullptr;
    kernel.CopyKernelTo(&hkernel);
    CheckCLError(clEnqueueWriteBuffer(cl_queue_, kernel_buf, CL_TRUE, 0, kbytes,
                                      hkernel, 0, nullptr, nullptr),
                 "clEnqueueWriteBuffer(kernel)");
    delete[] hkernel;
    CheckCLError(clSetKernelArg(k, arg++, sizeof(cl_mem), &kernel_buf),
                 "clSetKernelArg(kernel)");
    CheckCLError(clSetKernelArg(k, arg++, sizeof(int), &ksize),
                 "clSetKernelArg(ksize)");
    CheckCLError(clSetKernelArg(k, arg++, sizeof(int), &rot),
                 "clSetKernelArg(rotate)");
  }

  size_t global[2] = {RoundUp(width_, kBlockSize),
                      RoundUp(height_, kBlockSize)};
  size_t local[2] = {kBlockSize, kBlockSize};
  CheckCLError(clEnqueueNDRangeKernel(cl_queue_, k, 2, nullptr, global, local,
                                      0, nullptr, nullptr),
               "clEnqueueNDRangeKernel");
  CheckCLError(clFinish(cl_queue_), "clFinish");

  // Чтение результата
  uint16_t* h_dst = new uint16_t[width_ * height_];
  CheckCLError(clEnqueueReadBuffer(cl_queue_, dst, CL_TRUE, 0, image_size,
                                   h_dst, 0, nullptr, nullptr),
               "clEnqueueReadBuffer(dst)");

  if (kernel_buf)
    clReleaseMemObject(kernel_buf);
  clReleaseKernel(k);
  if (ephemeral) {
    clReleaseMemObject(src);
    clReleaseMemObject(dst);
  }

  TIFFImage result(*this);
  std::memcpy(result.image_, h_dst, image_size);
  delete[] h_dst;
  return result;
}

TIFFImage TIFFImage::SetKernelSobelSepOpenCL(
    const bool /*shared_memory*/) const {
  const_cast<TIFFImage*>(this)->EnsureOpenCLProgramBuilt();
  cl_int err;
  bool ephemeral = false;
  size_t image_size = width_ * height_ * sizeof(uint16_t);
  size_t temps_size_i = width_ * height_ * sizeof(int);
  cl_mem src = cl_src_, dst = cl_dst_, gx = cl_sep_g_x_, gy = cl_sep_g_y_,
         rx = cl_sep_result_x_, ry = cl_sep_result_y_;
  if (!cl_allocated_) {
    ephemeral = true;
    src = clCreateBuffer(cl_context_, CL_MEM_READ_WRITE, image_size, nullptr,
                         &err);
    dst = clCreateBuffer(cl_context_, CL_MEM_READ_WRITE, image_size, nullptr,
                         &err);
    gx = clCreateBuffer(cl_context_, CL_MEM_READ_WRITE, temps_size_i, nullptr,
                        &err);
    gy = clCreateBuffer(cl_context_, CL_MEM_READ_WRITE, temps_size_i, nullptr,
                        &err);
    rx = clCreateBuffer(cl_context_, CL_MEM_READ_WRITE, temps_size_i, nullptr,
                        &err);
    ry = clCreateBuffer(cl_context_, CL_MEM_READ_WRITE, temps_size_i, nullptr,
                        &err);
    CheckCLError(err, "clCreateBuffer(tmp)");
    CheckCLError(clEnqueueWriteBuffer(cl_queue_, src, CL_TRUE, 0, image_size,
                                      image_, 0, nullptr, nullptr),
                 "clEnqueueWriteBuffer(src)");
  }

  size_t h = height_, w = width_;
  size_t global[2] = {RoundUp(width_, kBlockSize),
                      RoundUp(height_, kBlockSize)};
  size_t local[2] = {kBlockSize, kBlockSize};

  auto run2 = [&](const char* name, cl_mem a0, cl_mem a1, cl_mem a2) {
    cl_kernel k = clCreateKernel(cl_program_, name, &err);
    CheckCLError(err, "clCreateKernel");
    int arg = 0;
    CheckCLError(clSetKernelArg(k, arg++, sizeof(cl_mem), &a0), "arg0");
    CheckCLError(clSetKernelArg(k, arg++, sizeof(cl_mem), &a1), "arg1");
    CheckCLError(clSetKernelArg(k, arg++, sizeof(size_t), &h), "arg2");
    CheckCLError(clSetKernelArg(k, arg++, sizeof(size_t), &w), "arg3");
    if (a2)
      CheckCLError(clSetKernelArg(k, arg++, sizeof(cl_mem), &a2), "arg4");
    CheckCLError(clEnqueueNDRangeKernel(cl_queue_, k, 2, nullptr, global, local,
                                        0, nullptr, nullptr),
                 "clEnqueueNDRangeKernel");
    clReleaseKernel(k);
  };

  // 1) Smooth
  run2("SobelSmooth", src, gx, nullptr);
  run2("SobelSmoothY", src, gy,
       nullptr);  // используем отдельное ядро для Y, см. kernels.cl
  CheckCLError(clFinish(cl_queue_), "clFinish1");
  // 2) Diff
  {
    cl_kernel k = clCreateKernel(cl_program_, "SepKernelDiff", &err);
    CheckCLError(err, "clCreateKernel(SepKernelDiff)");
    int arg = 0;
    CheckCLError(clSetKernelArg(k, arg++, sizeof(cl_mem), &gx), "gx");
    CheckCLError(clSetKernelArg(k, arg++, sizeof(cl_mem), &gy), "gy");
    CheckCLError(clSetKernelArg(k, arg++, sizeof(cl_mem), &rx), "rx");
    CheckCLError(clSetKernelArg(k, arg++, sizeof(cl_mem), &ry), "ry");
    CheckCLError(clSetKernelArg(k, arg++, sizeof(size_t), &h), "h");
    CheckCLError(clSetKernelArg(k, arg++, sizeof(size_t), &w), "w");
    CheckCLError(clEnqueueNDRangeKernel(cl_queue_, k, 2, nullptr, global, local,
                                        0, nullptr, nullptr),
                 "enqueue diff");
    clReleaseKernel(k);
  }
  CheckCLError(clFinish(cl_queue_), "clFinish2");
  // 3) AddAbs
  run2("AddAbsMtx", rx, ry, dst);
  CheckCLError(clFinish(cl_queue_), "clFinish3");

  uint16_t* h_dst = new uint16_t[width_ * height_];
  CheckCLError(clEnqueueReadBuffer(cl_queue_, dst, CL_TRUE, 0, image_size,
                                   h_dst, 0, nullptr, nullptr),
               "read dst");

  if (ephemeral) {
    clReleaseMemObject(src);
    clReleaseMemObject(dst);
    clReleaseMemObject(gx);
    clReleaseMemObject(gy);
    clReleaseMemObject(rx);
    clReleaseMemObject(ry);
  }
  TIFFImage result(*this);
  std::memcpy(result.image_, h_dst, image_size);
  delete[] h_dst;
  return result;
}

TIFFImage TIFFImage::SetKernelPrewittSepOpenCL(
    const bool /*shared_memory*/) const {
  const_cast<TIFFImage*>(this)->EnsureOpenCLProgramBuilt();
  cl_int err;
  bool ephemeral = false;
  size_t image_size = width_ * height_ * sizeof(uint16_t);
  size_t temps_size_i = width_ * height_ * sizeof(int);
  cl_mem src = cl_src_, dst = cl_dst_, gx = cl_sep_g_x_, gy = cl_sep_g_y_,
         rx = cl_sep_result_x_, ry = cl_sep_result_y_;
  if (!cl_allocated_) {
    ephemeral = true;
    src = clCreateBuffer(cl_context_, CL_MEM_READ_WRITE, image_size, nullptr,
                         &err);
    dst = clCreateBuffer(cl_context_, CL_MEM_READ_WRITE, image_size, nullptr,
                         &err);
    gx = clCreateBuffer(cl_context_, CL_MEM_READ_WRITE, temps_size_i, nullptr,
                        &err);
    gy = clCreateBuffer(cl_context_, CL_MEM_READ_WRITE, temps_size_i, nullptr,
                        &err);
    rx = clCreateBuffer(cl_context_, CL_MEM_READ_WRITE, temps_size_i, nullptr,
                        &err);
    ry = clCreateBuffer(cl_context_, CL_MEM_READ_WRITE, temps_size_i, nullptr,
                        &err);
    CheckCLError(err, "clCreateBuffer(tmp)");
    CheckCLError(clEnqueueWriteBuffer(cl_queue_, src, CL_TRUE, 0, image_size,
                                      image_, 0, nullptr, nullptr),
                 "clEnqueueWriteBuffer(src)");
  }

  size_t h = height_, w = width_;
  size_t global[2] = {RoundUp(width_, kBlockSize),
                      RoundUp(height_, kBlockSize)};
  size_t local[2] = {kBlockSize, kBlockSize};

  auto run2 = [&](const char* name, cl_mem a0, cl_mem a1, cl_mem a2) {
    cl_int err2;
    cl_kernel k = clCreateKernel(cl_program_, name, &err2);
    CheckCLError(err2, "clCreateKernel");
    int arg = 0;
    CheckCLError(clSetKernelArg(k, arg++, sizeof(cl_mem), &a0), "arg0");
    CheckCLError(clSetKernelArg(k, arg++, sizeof(cl_mem), &a1), "arg1");
    CheckCLError(clSetKernelArg(k, arg++, sizeof(size_t), &h), "arg2");
    CheckCLError(clSetKernelArg(k, arg++, sizeof(size_t), &w), "arg3");
    if (a2)
      CheckCLError(clSetKernelArg(k, arg++, sizeof(cl_mem), &a2), "arg4");
    CheckCLError(clEnqueueNDRangeKernel(cl_queue_, k, 2, nullptr, global, local,
                                        0, nullptr, nullptr),
                 "clEnqueueNDRangeKernel");
    clReleaseKernel(k);
  };

  // 1) Average
  run2("PrewittAverage", src, gx, nullptr);
  run2("PrewittAverageY", src, gy, nullptr);
  CheckCLError(clFinish(cl_queue_), "clFinish1");
  // 2) Diff
  {
    cl_kernel k = clCreateKernel(cl_program_, "SepKernelDiff", &err);
    CheckCLError(err, "clCreateKernel(SepKernelDiff)");
    int arg = 0;
    CheckCLError(clSetKernelArg(k, arg++, sizeof(cl_mem), &gx), "gx");
    CheckCLError(clSetKernelArg(k, arg++, sizeof(cl_mem), &gy), "gy");
    CheckCLError(clSetKernelArg(k, arg++, sizeof(cl_mem), &rx), "rx");
    CheckCLError(clSetKernelArg(k, arg++, sizeof(cl_mem), &ry), "ry");
    CheckCLError(clSetKernelArg(k, arg++, sizeof(size_t), &h), "h");
    CheckCLError(clSetKernelArg(k, arg++, sizeof(size_t), &w), "w");
    CheckCLError(clEnqueueNDRangeKernel(cl_queue_, k, 2, nullptr, global, local,
                                        0, nullptr, nullptr),
                 "enqueue diff");
    clReleaseKernel(k);
  }
  CheckCLError(clFinish(cl_queue_), "clFinish2");
  // 3) AddAbs
  run2("AddAbsMtx", rx, ry, dst);
  CheckCLError(clFinish(cl_queue_), "clFinish3");

  uint16_t* h_dst = new uint16_t[width_ * height_];
  CheckCLError(clEnqueueReadBuffer(cl_queue_, dst, CL_TRUE, 0, image_size,
                                   h_dst, 0, nullptr, nullptr),
               "read dst");

  if (ephemeral) {
    clReleaseMemObject(src);
    clReleaseMemObject(dst);
    clReleaseMemObject(gx);
    clReleaseMemObject(gy);
    clReleaseMemObject(rx);
    clReleaseMemObject(ry);
  }
  TIFFImage result(*this);
  std::memcpy(result.image_, h_dst, image_size);
  delete[] h_dst;
  return result;
}

TIFFImage TIFFImage::GaussianBlurOpenCL(const size_t size, const float sigma) {
  EnsureOpenCLProgramBuilt();
  cl_int err;
  bool ephemeral = false;
  size_t image_size = width_ * height_ * sizeof(uint16_t);
  cl_mem src = cl_src_, dst = cl_dst_;
  if (!cl_allocated_) {
    ephemeral = true;
    src = clCreateBuffer(cl_context_, CL_MEM_READ_WRITE, image_size, nullptr,
                         &err);
    dst = clCreateBuffer(cl_context_, CL_MEM_READ_WRITE, image_size, nullptr,
                         &err);
    CheckCLError(err, "clCreateBuffer(tmp)");
    CheckCLError(clEnqueueWriteBuffer(cl_queue_, src, CL_TRUE, 0, image_size,
                                      image_, 0, nullptr, nullptr),
                 "clEnqueueWriteBuffer(src)");
  }
  // kernel
  Kernel<float> k2d = Kernel<float>::GetGaussianKernel(size, sigma);
  size_t kbytes = k2d.GetHeight() * k2d.GetWidth() * sizeof(float);
  cl_mem kbuf =
      clCreateBuffer(cl_context_, CL_MEM_READ_ONLY, kbytes, nullptr, &err);
  CheckCLError(err, "clCreateBuffer(k)");
  float* h_kernel = nullptr;
  k2d.CopyKernelTo(&h_kernel);
  CheckCLError(clEnqueueWriteBuffer(cl_queue_, kbuf, CL_TRUE, 0, kbytes,
                                    h_kernel, 0, nullptr, nullptr),
               "write k");
  delete[] h_kernel;

  cl_kernel kern = clCreateKernel(cl_program_, "GaussianBlur", &err);
  CheckCLError(err, "clCreateKernel(GaussianBlur)");
  size_t h = height_, w = width_;
  int ksize = static_cast<int>(k2d.GetHeight());
  int arg = 0;
  CheckCLError(clSetKernelArg(kern, arg++, sizeof(cl_mem), &src), "src");
  CheckCLError(clSetKernelArg(kern, arg++, sizeof(cl_mem), &dst), "dst");
  CheckCLError(clSetKernelArg(kern, arg++, sizeof(size_t), &h), "h");
  CheckCLError(clSetKernelArg(kern, arg++, sizeof(size_t), &w), "w");
  CheckCLError(clSetKernelArg(kern, arg++, sizeof(cl_mem), &kbuf), "k");
  CheckCLError(clSetKernelArg(kern, arg++, sizeof(int), &ksize), "ksize");

  size_t global[2] = {RoundUp(width_, kBlockSize),
                      RoundUp(height_, kBlockSize)};
  size_t local[2] = {kBlockSize, kBlockSize};
  CheckCLError(clEnqueueNDRangeKernel(cl_queue_, kern, 2, nullptr, global,
                                      local, 0, nullptr, nullptr),
               "enqueue gaussian");
  CheckCLError(clFinish(cl_queue_), "finish gaussian");

  uint16_t* h_dst = new uint16_t[width_ * height_];
  CheckCLError(clEnqueueReadBuffer(cl_queue_, dst, CL_TRUE, 0, image_size,
                                   h_dst, 0, nullptr, nullptr),
               "read dst");

  clReleaseMemObject(kbuf);
  clReleaseKernel(kern);
  if (ephemeral) {
    clReleaseMemObject(src);
    clReleaseMemObject(dst);
  }
  TIFFImage result(*this);
  std::memcpy(result.image_, h_dst, image_size);
  delete[] h_dst;
  return result;
}

TIFFImage TIFFImage::GaussianBlurSepOpenCL(const size_t size,
                                           const float sigma) {
  EnsureOpenCLProgramBuilt();
  cl_int err;
  bool ephemeral = false;
  size_t image_size = width_ * height_ * sizeof(uint16_t);
  size_t temp_size = width_ * height_ * sizeof(float);
  cl_mem src = cl_src_, dst = cl_dst_, temp = cl_gaussian_sep_temp_;
  if (!cl_allocated_) {
    ephemeral = true;
    src = clCreateBuffer(cl_context_, CL_MEM_READ_WRITE, image_size, nullptr,
                         &err);
    dst = clCreateBuffer(cl_context_, CL_MEM_READ_WRITE, image_size, nullptr,
                         &err);
    temp = clCreateBuffer(cl_context_, CL_MEM_READ_WRITE, temp_size, nullptr,
                          &err);
    CheckCLError(err, "clCreateBuffer(tmp)");
    CheckCLError(clEnqueueWriteBuffer(cl_queue_, src, CL_TRUE, 0, image_size,
                                      image_, 0, nullptr, nullptr),
                 "clEnqueueWriteBuffer(src)");
  }

  EnsureGaussianKernelBuffer(size, sigma);

  size_t h = height_, w = width_;
  int ksize = static_cast<int>(size);
  size_t global[2] = {RoundUp(width_, kBlockSize),
                      RoundUp(height_, kBlockSize)};
  size_t local[2] = {kBlockSize, kBlockSize};

  // Horizontal
  {
    cl_kernel kern =
        clCreateKernel(cl_program_, "GaussianBlurSepHorizontal", &err);
    CheckCLError(err, "clCreateKernel(Horizontal)");
    int arg = 0;
    CheckCLError(clSetKernelArg(kern, arg++, sizeof(cl_mem), &src), "src");
    CheckCLError(clSetKernelArg(kern, arg++, sizeof(cl_mem), &temp), "temp");
    CheckCLError(clSetKernelArg(kern, arg++, sizeof(size_t), &h), "h");
    CheckCLError(clSetKernelArg(kern, arg++, sizeof(size_t), &w), "w");
    CheckCLError(
        clSetKernelArg(kern, arg++, sizeof(cl_mem), &cl_gaussian_kernel_), "k");
    CheckCLError(clSetKernelArg(kern, arg++, sizeof(int), &ksize), "ksize");
    CheckCLError(clEnqueueNDRangeKernel(cl_queue_, kern, 2, nullptr, global,
                                        local, 0, nullptr, nullptr),
                 "enqueue horiz");
    clReleaseKernel(kern);
  }
  CheckCLError(clFinish(cl_queue_), "finish horiz");
  // Vertical
  {
    cl_kernel kern =
        clCreateKernel(cl_program_, "GaussianBlurSepVertical", &err);
    CheckCLError(err, "clCreateKernel(Vertical)");
    int arg = 0;
    CheckCLError(clSetKernelArg(kern, arg++, sizeof(cl_mem), &temp), "temp");
    CheckCLError(clSetKernelArg(kern, arg++, sizeof(cl_mem), &dst), "dst");
    CheckCLError(clSetKernelArg(kern, arg++, sizeof(size_t), &h), "h");
    CheckCLError(clSetKernelArg(kern, arg++, sizeof(size_t), &w), "w");
    CheckCLError(
        clSetKernelArg(kern, arg++, sizeof(cl_mem), &cl_gaussian_kernel_), "k");
    CheckCLError(clSetKernelArg(kern, arg++, sizeof(int), &ksize), "ksize");
    CheckCLError(clEnqueueNDRangeKernel(cl_queue_, kern, 2, nullptr, global,
                                        local, 0, nullptr, nullptr),
                 "enqueue vert");
    clReleaseKernel(kern);
  }
  CheckCLError(clFinish(cl_queue_), "finish vert");

  uint16_t* h_dst = new uint16_t[width_ * height_];
  CheckCLError(clEnqueueReadBuffer(cl_queue_, dst, CL_TRUE, 0, image_size,
                                   h_dst, 0, nullptr, nullptr),
               "read dst");

  if (ephemeral) {
    clReleaseMemObject(src);
    clReleaseMemObject(dst);
    clReleaseMemObject(temp);
  }
  TIFFImage result(*this);
  std::memcpy(result.image_, h_dst, image_size);
  delete[] h_dst;
  return result;
}
