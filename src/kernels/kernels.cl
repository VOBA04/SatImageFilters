// OpenCL kernels analogous to CUDA versions in tiff_image.cu

__constant int KERNEL_SOBEL[9] = {-1, 0, 1, -2, 0, 2, -1, 0, 1};
__constant int KERNEL_PREWITT[9] = {-1, 0, 1, -1, 0, 1, -1, 0, 1};
__constant int KERNEL_SMOOTH[3] = {1, 2, 1};
__constant int KERNEL_AVG[3] = {1, 1, 1};
__constant int KERNEL_GRAD[3] = {-1, 0, 1};

inline ushort clamp_u16(int v) {
  return (ushort)((v < 0) ? 0 : (v > 65535 ? 65535 : v));
}

kernel void ApplyKernel(global ushort* src, global ushort* dst, ulong height,
                        ulong width, global int* k, int ksize, int rotate) {
  const int j = get_global_id(0);
  const int i = get_global_id(1);
  if ((ulong)i >= height || (ulong)j >= width)
    return;
  if (rotate != 0) {
    int gx = 0, gy = 0;
    for (int r = 0; r < ksize; ++r) {
      for (int c = 0; c < ksize; ++c) {
        int x = j + c - ksize / 2;
        int y = i + r - ksize / 2;
        if (x < 0)
          x = 0;
        if (x >= (int)width)
          x = (int)width - 1;
        if (y < 0)
          y = 0;
        if (y >= (int)height)
          y = (int)height - 1;
        int val = src[(ulong)y * width + (ulong)x];
        gx += val * k[r * ksize + c];
        gy += val * k[(ksize - 1 - c) * ksize + r];
      }
    }
    int s = abs(gx) + abs(gy);
    dst[(ulong)i * width + (ulong)j] = (ushort)(s > 65535 ? 65535 : s);
  } else {
    int g = 0;
    for (int r = 0; r < ksize; ++r) {
      for (int c = 0; c < ksize; ++c) {
        int x = j + c - ksize / 2;
        int y = i + r - ksize / 2;
        if (x < 0)
          x = 0;
        if (x >= (int)width)
          x = (int)width - 1;
        if (y < 0)
          y = 0;
        if (y >= (int)height)
          y = (int)height - 1;
        int val = src[(ulong)y * width + (ulong)x];
        g += val * k[r * ksize + c];
      }
    }
    if (g < 0)
      g = 0;
    if (g > 65535)
      g = 65535;
    dst[(ulong)i * width + (ulong)j] = (ushort)g;
  }
}

kernel void Sobel(global ushort* src, global ushort* dst, ulong h, ulong w) {
  const int j = get_global_id(0);
  const int i = get_global_id(1);
  if ((ulong)i >= h || (ulong)j >= w)
    return;
  int gx = 0, gy = 0;
  for (int r = 0; r < 3; ++r) {
    for (int c = 0; c < 3; ++c) {
      int x = j + c - 1;
      int y = i + r - 1;
      if (x < 0)
        x = 0;
      if (x >= (int)w)
        x = (int)w - 1;
      if (y < 0)
        y = 0;
      if (y >= (int)h)
        y = (int)h - 1;
      int v = src[(ulong)y * w + (ulong)x];
      gx += v * KERNEL_SOBEL[r * 3 + c];
      gy += v * KERNEL_SOBEL[(2 - c) * 3 + r];
    }
  }
  int s = abs(gx) + abs(gy);
  dst[(ulong)i * w + (ulong)j] = (ushort)(s > 65535 ? 65535 : s);
}

kernel void Prewitt(global ushort* src, global ushort* dst, ulong h, ulong w) {
  const int j = get_global_id(0);
  const int i = get_global_id(1);
  if ((ulong)i >= h || (ulong)j >= w)
    return;
  int gx = 0, gy = 0;
  for (int r = 0; r < 3; ++r) {
    for (int c = 0; c < 3; ++c) {
      int x = j + c - 1;
      int y = i + r - 1;
      if (x < 0)
        x = 0;
      if (x >= (int)w)
        x = (int)w - 1;
      if (y < 0)
        y = 0;
      if (y >= (int)h)
        y = (int)h - 1;
      int v = src[(ulong)y * w + (ulong)x];
      gx += v * KERNEL_PREWITT[r * 3 + c];
      gy += v * KERNEL_PREWITT[(2 - c) * 3 + r];
    }
  }
  int s = abs(gx) + abs(gy);
  dst[(ulong)i * w + (ulong)j] = (ushort)(s > 65535 ? 65535 : s);
}

// ---------------- Local memory optimized 3x3 variants ----------------
// Tile-based loading into __local memory with R=1 halo
inline int clampi(int v, int lo, int hi) {
  return v < lo ? lo : (v > hi ? hi : v);
}

kernel void SobelLocal(global ushort* src, global ushort* dst, ulong h, ulong w,
                       local ushort* tile) {
  const int gx = get_global_id(0);
  const int gy = get_global_id(1);
  const int lx = get_local_id(0);
  const int ly = get_local_id(1);
  const int ldx = get_local_size(0);
  const int ldy = get_local_size(1);
  const int R = 1;
  const int tileW = ldx + 2 * R;
  const int tx = lx + R;
  const int ty = ly + R;

  // Center pixel
  int cx = clampi(gx, 0, (int)w - 1);
  int cy = clampi(gy, 0, (int)h - 1);
  tile[ty * tileW + tx] = src[(ulong)cy * w + (ulong)cx];

  // Halo (R=1): left/right/top/bottom
  if (lx < R) {
    int x = clampi(gx - 1, 0, (int)w - 1);
    tile[ty * tileW + (tx - 1)] = src[(ulong)cy * w + (ulong)x];
  }
  if (lx >= ldx - R) {
    int x = clampi(gx + 1, 0, (int)w - 1);
    tile[ty * tileW + (tx + 1)] = src[(ulong)cy * w + (ulong)x];
  }
  if (ly < R) {
    int y = clampi(gy - 1, 0, (int)h - 1);
    tile[(ty - 1) * tileW + tx] = src[(ulong)y * w + (ulong)cx];
  }
  if (ly >= ldy - R) {
    int y = clampi(gy + 1, 0, (int)h - 1);
    tile[(ty + 1) * tileW + tx] = src[(ulong)y * w + (ulong)cx];
  }
  // Corners
  if (lx < R && ly < R) {
    int x = clampi(gx - 1, 0, (int)w - 1);
    int y = clampi(gy - 1, 0, (int)h - 1);
    tile[(ty - 1) * tileW + (tx - 1)] = src[(ulong)y * w + (ulong)x];
  }
  if (lx < R && ly >= ldy - R) {
    int x = clampi(gx - 1, 0, (int)w - 1);
    int y = clampi(gy + 1, 0, (int)h - 1);
    tile[(ty + 1) * tileW + (tx - 1)] = src[(ulong)y * w + (ulong)x];
  }
  if (lx >= ldx - R && ly < R) {
    int x = clampi(gx + 1, 0, (int)w - 1);
    int y = clampi(gy - 1, 0, (int)h - 1);
    tile[(ty - 1) * tileW + (tx + 1)] = src[(ulong)y * w + (ulong)x];
  }
  if (lx >= ldx - R && ly >= ldy - R) {
    int x = clampi(gx + 1, 0, (int)w - 1);
    int y = clampi(gy + 1, 0, (int)h - 1);
    tile[(ty + 1) * tileW + (tx + 1)] = src[(ulong)y * w + (ulong)x];
  }

  barrier(CLK_LOCAL_MEM_FENCE);
  if ((ulong)gy >= h || (ulong)gx >= w)
    return;

  int gxv = 0, gyv = 0;
  // Convolution in local memory
  for (int r = -1; r <= 1; ++r) {
    for (int c = -1; c <= 1; ++c) {
      int v = (int)tile[(ty + r) * tileW + (tx + c)];
      int kr = KERNEL_SOBEL[(r + 1) * 3 + (c + 1)];
      int kr_t = KERNEL_SOBEL[(1 - c) * 3 + (r + 1)];
      gxv += v * kr;
      gyv += v * kr_t;
    }
  }
  int s = abs(gxv) + abs(gyv);
  dst[(ulong)gy * w + (ulong)gx] = (ushort)(s > 65535 ? 65535 : s);
}

kernel void PrewittLocal(global ushort* src, global ushort* dst, ulong h,
                         ulong w, local ushort* tile) {
  const int gx = get_global_id(0);
  const int gy = get_global_id(1);
  const int lx = get_local_id(0);
  const int ly = get_local_id(1);
  const int ldx = get_local_size(0);
  const int ldy = get_local_size(1);
  const int R = 1;
  const int tileW = ldx + 2 * R;
  const int tx = lx + R;
  const int ty = ly + R;

  int cx = clampi(gx, 0, (int)w - 1);
  int cy = clampi(gy, 0, (int)h - 1);
  tile[ty * tileW + tx] = src[(ulong)cy * w + (ulong)cx];
  if (lx < R) {
    int x = clampi(gx - 1, 0, (int)w - 1);
    tile[ty * tileW + (tx - 1)] = src[(ulong)cy * w + (ulong)x];
  }
  if (lx >= ldx - R) {
    int x = clampi(gx + 1, 0, (int)w - 1);
    tile[ty * tileW + (tx + 1)] = src[(ulong)cy * w + (ulong)x];
  }
  if (ly < R) {
    int y = clampi(gy - 1, 0, (int)h - 1);
    tile[(ty - 1) * tileW + tx] = src[(ulong)y * w + (ulong)cx];
  }
  if (ly >= ldy - R) {
    int y = clampi(gy + 1, 0, (int)h - 1);
    tile[(ty + 1) * tileW + tx] = src[(ulong)y * w + (ulong)cx];
  }
  if (lx < R && ly < R) {
    int x = clampi(gx - 1, 0, (int)w - 1);
    int y = clampi(gy - 1, 0, (int)h - 1);
    tile[(ty - 1) * tileW + (tx - 1)] = src[(ulong)y * w + (ulong)x];
  }
  if (lx < R && ly >= ldy - R) {
    int x = clampi(gx - 1, 0, (int)w - 1);
    int y = clampi(gy + 1, 0, (int)h - 1);
    tile[(ty + 1) * tileW + (tx - 1)] = src[(ulong)y * w + (ulong)x];
  }
  if (lx >= ldx - R && ly < R) {
    int x = clampi(gx + 1, 0, (int)w - 1);
    int y = clampi(gy - 1, 0, (int)h - 1);
    tile[(ty - 1) * tileW + (tx + 1)] = src[(ulong)y * w + (ulong)x];
  }
  if (lx >= ldx - R && ly >= ldy - R) {
    int x = clampi(gx + 1, 0, (int)w - 1);
    int y = clampi(gy + 1, 0, (int)h - 1);
    tile[(ty + 1) * tileW + (tx + 1)] = src[(ulong)y * w + (ulong)x];
  }

  barrier(CLK_LOCAL_MEM_FENCE);
  if ((ulong)gy >= h || (ulong)gx >= w)
    return;

  int gxv = 0, gyv = 0;
  for (int r = -1; r <= 1; ++r) {
    for (int c = -1; c <= 1; ++c) {
      int v = (int)tile[(ty + r) * tileW + (tx + c)];
      int kr = KERNEL_PREWITT[(r + 1) * 3 + (c + 1)];
      int kr_t = KERNEL_PREWITT[(1 - c) * 3 + (r + 1)];
      gxv += v * kr;
      gyv += v * kr_t;
    }
  }
  int s = abs(gxv) + abs(gyv);
  dst[(ulong)gy * w + (ulong)gx] = (ushort)(s > 65535 ? 65535 : s);
}

// Generic 3x3 with provided kernel (rotate behaves like in ApplyKernel)
kernel void ApplyKernel3Local(global ushort* src, global ushort* dst, ulong h,
                              ulong w, global int* k, int ksize, int rotate,
                              local ushort* tile) {
  const int gx = get_global_id(0);
  const int gy = get_global_id(1);
  const int lx = get_local_id(0);
  const int ly = get_local_id(1);
  const int ldx = get_local_size(0);
  const int ldy = get_local_size(1);
  const int R = 1;  // only for ksize==3
  const int tileW = ldx + 2 * R;
  const int tx = lx + R;
  const int ty = ly + R;

  int cx = clampi(gx, 0, (int)w - 1);
  int cy = clampi(gy, 0, (int)h - 1);
  tile[ty * tileW + tx] = src[(ulong)cy * w + (ulong)cx];
  if (lx < R) {
    int x = clampi(gx - 1, 0, (int)w - 1);
    tile[ty * tileW + (tx - 1)] = src[(ulong)cy * w + (ulong)x];
  }
  if (lx >= ldx - R) {
    int x = clampi(gx + 1, 0, (int)w - 1);
    tile[ty * tileW + (tx + 1)] = src[(ulong)cy * w + (ulong)x];
  }
  if (ly < R) {
    int y = clampi(gy - 1, 0, (int)h - 1);
    tile[(ty - 1) * tileW + tx] = src[(ulong)y * w + (ulong)cx];
  }
  if (ly >= ldy - R) {
    int y = clampi(gy + 1, 0, (int)h - 1);
    tile[(ty + 1) * tileW + tx] = src[(ulong)y * w + (ulong)cx];
  }
  if (lx < R && ly < R) {
    int x = clampi(gx - 1, 0, (int)w - 1);
    int y = clampi(gy - 1, 0, (int)h - 1);
    tile[(ty - 1) * tileW + (tx - 1)] = src[(ulong)y * w + (ulong)x];
  }
  if (lx < R && ly >= ldy - R) {
    int x = clampi(gx - 1, 0, (int)w - 1);
    int y = clampi(gy + 1, 0, (int)h - 1);
    tile[(ty + 1) * tileW + (tx - 1)] = src[(ulong)y * w + (ulong)x];
  }
  if (lx >= ldx - R && ly < R) {
    int x = clampi(gx + 1, 0, (int)w - 1);
    int y = clampi(gy - 1, 0, (int)h - 1);
    tile[(ty - 1) * tileW + (tx + 1)] = src[(ulong)y * w + (ulong)x];
  }
  if (lx >= ldx - R && ly >= ldy - R) {
    int x = clampi(gx + 1, 0, (int)w - 1);
    int y = clampi(gy + 1, 0, (int)h - 1);
    tile[(ty + 1) * tileW + (tx + 1)] = src[(ulong)y * w + (ulong)x];
  }

  barrier(CLK_LOCAL_MEM_FENCE);
  if ((ulong)gy >= h || (ulong)gx >= w)
    return;

  if (rotate != 0) {
    int gvx = 0, gvy = 0;
    for (int r = -1; r <= 1; ++r) {
      for (int c = -1; c <= 1; ++c) {
        int v = (int)tile[(ty + r) * tileW + (tx + c)];
        int kk = k[(r + 1) * 3 + (c + 1)];
        int kk_t = k[(1 - c) * 3 + (r + 1)];
        gvx += v * kk;
        gvy += v * kk_t;
      }
    }
    int s = abs(gvx) + abs(gvy);
    dst[(ulong)gy * w + (ulong)gx] = (ushort)(s > 65535 ? 65535 : s);
  } else {
    int g = 0;
    for (int r = -1; r <= 1; ++r) {
      for (int c = -1; c <= 1; ++c) {
        int v = (int)tile[(ty + r) * tileW + (tx + c)];
        int kk = k[(r + 1) * 3 + (c + 1)];
        g += v * kk;
      }
    }
    g = g < 0 ? 0 : (g > 65535 ? 65535 : g);
    dst[(ulong)gy * w + (ulong)gx] = (ushort)g;
  }
}

// Separable pre-smoothing (Sobel)
kernel void SobelSmooth(global ushort* src, global int* gx, ulong h, ulong w) {
  const int j = get_global_id(0);
  const int i = get_global_id(1);
  if ((ulong)i >= h || (ulong)j >= w)
    return;
  int sum = 0;
  for (int k = 0; k < 3; ++k) {
    int x = j + k - 1;
    if (x < 0)
      x = 0;
    if (x >= (int)w)
      x = (int)w - 1;
    sum += (int)src[(ulong)i * w + (ulong)x] * KERNEL_SMOOTH[k];
  }
  gx[(ulong)i * w + (ulong)j] = sum;
}

kernel void SobelSmoothY(global ushort* src, global int* gy, ulong h, ulong w) {
  const int j = get_global_id(0);
  const int i = get_global_id(1);
  if ((ulong)i >= h || (ulong)j >= w)
    return;
  int sum = 0;
  for (int k = 0; k < 3; ++k) {
    int y = i + k - 1;
    if (y < 0)
      y = 0;
    if (y >= (int)h)
      y = (int)h - 1;
    sum += (int)src[(ulong)y * w + (ulong)j] * KERNEL_SMOOTH[k];
  }
  gy[(ulong)i * w + (ulong)j] = sum;
}

kernel void PrewittAverage(global ushort* src, global int* gx, ulong h,
                           ulong w) {
  const int j = get_global_id(0);
  const int i = get_global_id(1);
  if ((ulong)i >= h || (ulong)j >= w)
    return;
  int sum = 0;
  for (int k = 0; k < 3; ++k) {
    int x = j + k - 1;
    if (x < 0)
      x = 0;
    if (x >= (int)w)
      x = (int)w - 1;
    sum += (int)src[(ulong)i * w + (ulong)x] * KERNEL_AVG[k];
  }
  gx[(ulong)i * w + (ulong)j] = sum;
}

kernel void PrewittAverageY(global ushort* src, global int* gy, ulong h,
                            ulong w) {
  const int j = get_global_id(0);
  const int i = get_global_id(1);
  if ((ulong)i >= h || (ulong)j >= w)
    return;
  int sum = 0;
  for (int k = 0; k < 3; ++k) {
    int y = i + k - 1;
    if (y < 0)
      y = 0;
    if (y >= (int)h)
      y = (int)h - 1;
    sum += (int)src[(ulong)y * w + (ulong)j] * KERNEL_AVG[k];
  }
  gy[(ulong)i * w + (ulong)j] = sum;
}

kernel void SepKernelDiff(global int* gx, global int* gy, global int* rx,
                          global int* ry, ulong h, ulong w) {
  const int j = get_global_id(0);
  const int i = get_global_id(1);
  if ((ulong)i >= h || (ulong)j >= w)
    return;
  int sumx = 0, sumy = 0;
  for (int k = 0; k < 3; ++k) {
    int x = j + k - 1;
    if (x < 0)
      x = 0;
    if (x >= (int)w)
      x = (int)w - 1;
    int y = i + k - 1;
    if (y < 0)
      y = 0;
    if (y >= (int)h)
      y = (int)h - 1;
    sumy += gy[(ulong)i * w + (ulong)x] * KERNEL_GRAD[k];
    sumx += gx[(ulong)y * w + (ulong)j] * KERNEL_GRAD[k];
  }
  rx[(ulong)i * w + (ulong)j] = sumx;
  ry[(ulong)i * w + (ulong)j] = sumy;
}

kernel void AddAbsMtx(global int* m1, global int* m2, global ushort* dst,
                      ulong h, ulong w) {
  const int j = get_global_id(0);
  const int i = get_global_id(1);
  if ((ulong)i >= h || (ulong)j >= w)
    return;
  int s = abs(m1[(ulong)i * w + (ulong)j]) + abs(m2[(ulong)i * w + (ulong)j]);
  dst[(ulong)i * w + (ulong)j] = (ushort)(s > 65535 ? 65535 : s);
}

kernel void GaussianBlur(global ushort* src, global ushort* dst, ulong h,
                         ulong w, global float* k, int ksize) {
  const int j = get_global_id(0);
  const int i = get_global_id(1);
  if ((ulong)i >= h || (ulong)j >= w)
    return;
  float sum = 0.0f;
  for (int r = 0; r < ksize; ++r) {
    for (int c = 0; c < ksize; ++c) {
      int x = j + c - ksize / 2;
      int y = i + r - ksize / 2;
      if (x < 0)
        x = 0;
      if (x >= (int)w)
        x = (int)w - 1;
      if (y < 0)
        y = 0;
      if (y >= (int)h)
        y = (int)h - 1;
      sum += (float)src[(ulong)y * w + (ulong)x] * k[r * ksize + c];
    }
  }
  int iv = (int)floor(sum + 0.5f);
  dst[(ulong)i * w + (ulong)j] =
      (ushort)(iv < 0 ? 0 : (iv > 65535 ? 65535 : iv));
}

kernel void GaussianBlurSepHorizontal(global ushort* src, global float* tmp,
                                      ulong h, ulong w, global float* k,
                                      int ksize) {
  const int j = get_global_id(0);
  const int i = get_global_id(1);
  if ((ulong)i >= h || (ulong)j >= w)
    return;
  float sum = 0.0f;
  for (int kx = 0; kx < ksize; ++kx) {
    int x = j + kx - ksize / 2;
    if (x < 0)
      x = 0;
    if (x >= (int)w)
      x = (int)w - 1;
    sum += (float)src[(ulong)i * w + (ulong)x] * k[kx];
  }
  tmp[(ulong)i * w + (ulong)j] = sum;
}

kernel void GaussianBlurSepVertical(global float* tmp, global ushort* dst,
                                    ulong h, ulong w, global float* k,
                                    int ksize) {
  const int j = get_global_id(0);
  const int i = get_global_id(1);
  if ((ulong)i >= h || (ulong)j >= w)
    return;
  float sum = 0.0f;
  for (int ky = 0; ky < ksize; ++ky) {
    int y = i + ky - ksize / 2;
    if (y < 0)
      y = 0;
    if (y >= (int)h)
      y = (int)h - 1;
    sum += tmp[(ulong)y * w + (ulong)j] * k[ky];
  }
  int iv = (int)floor(sum + 0.5f);
  dst[(ulong)i * w + (ulong)j] =
      (ushort)(iv < 0 ? 0 : (iv > 65535 ? 65535 : iv));
}
