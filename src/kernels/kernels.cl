// OpenCL kernels analogous to CUDA versions in tiff_image.cu

__constant int KERNEL_SOBEL[9] = {-1, 0, 1, -2, 0, 2, -1, 0, 1};
__constant int KERNEL_PREWITT[9] = {-1, 0, 1, -1, 0, 1, -1, 0, 1};
__constant int KERNEL_SMOOTH[3] = {1, 2, 1};
__constant int KERNEL_AVG[3] = {1, 1, 1};
__constant int KERNEL_GRAD[3] = {-1, 0, 1};

inline ushort clamp_u16(int v) {
  return (ushort)((v < 0) ? 0 : (v > 65535 ? 65535 : v));
}

inline int clampi(int v, int lo, int hi) {
  return v < lo ? lo : (v > hi ? hi : v);
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
        x = clampi(x, 0, (int)width - 1);
        y = clampi(y, 0, (int)height - 1);
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
        x = clampi(x, 0, (int)width - 1);
        y = clampi(y, 0, (int)height - 1);
        int val = src[(ulong)y * width + (ulong)x];
        g += val * k[r * ksize + c];
      }
    }
    int ga = g < 0 ? -g : g;
    if (ga > 65535)
      ga = 65535;
    dst[(ulong)i * width + (ulong)j] = (ushort)ga;
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
      x = clampi(x, 0, (int)w - 1);
      y = clampi(y, 0, (int)h - 1);
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
      x = clampi(x, 0, (int)w - 1);
      y = clampi(y, 0, (int)h - 1);
      int v = src[(ulong)y * w + (ulong)x];
      gx += v * KERNEL_PREWITT[r * 3 + c];
      gy += v * KERNEL_PREWITT[(2 - c) * 3 + r];
    }
  }
  int s = abs(gx) + abs(gy);
  dst[(ulong)i * w + (ulong)j] = (ushort)(s > 65535 ? 65535 : s);
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
  int li = clampi(gy - R, 0, (int)h - 1);
  int lj = clampi(gx - R, 0, (int)w - 1);
  tile[ly * tileW + lx] = src[(ulong)li * w + (ulong)lj];
  if (ly < 2 * R) {
    int gy2 = clampi(gy - R + ldy, 0, (int)h - 1);
    tile[(ly + ldy) * tileW + lx] = src[(ulong)gy2 * w + (ulong)lj];
  }
  if (lx < 2 * R) {
    int gx2 = clampi(gx - R + ldx, 0, (int)w - 1);
    tile[ly * tileW + (lx + ldx)] = src[(ulong)li * w + (ulong)gx2];
  }
  if (ly >= ldy - 2 * R && lx >= ldx - 2 * R) {
    int gy2 = clampi(gy + R, 0, (int)h - 1);
    int gx2 = clampi(gx + R, 0, (int)w - 1);
    tile[(ly + 2 * R) * tileW + (lx + 2 * R)] =
        src[(ulong)gy2 * w + (ulong)gx2];
  }
  barrier(CLK_LOCAL_MEM_FENCE);
  if ((ulong)gy >= h || (ulong)gx >= w)
    return;
  int gxv = 0, gyv = 0;
  for (int r = 0; r < 3; ++r) {
    for (int c = 0; c < 3; ++c) {
      int v = (int)tile[(ly + r) * tileW + (lx + c)];
      int kr = KERNEL_SOBEL[r * 3 + c];
      int kr_t = KERNEL_SOBEL[(2 - c) * 3 + r];
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
  int li = clampi(gy - R, 0, (int)h - 1);
  int lj = clampi(gx - R, 0, (int)w - 1);
  tile[ly * tileW + lx] = src[(ulong)li * w + (ulong)lj];
  if (ly < 2 * R) {
    int gy2 = clampi(gy - R + ldy, 0, (int)h - 1);
    tile[(ly + ldy) * tileW + lx] = src[(ulong)gy2 * w + (ulong)lj];
  }
  if (lx < 2 * R) {
    int gx2 = clampi(gx - R + ldx, 0, (int)w - 1);
    tile[ly * tileW + (lx + ldx)] = src[(ulong)li * w + (ulong)gx2];
  }
  if (ly >= ldy - 2 * R && lx >= ldx - 2 * R) {
    int gy2 = clampi(gy + R, 0, (int)h - 1);
    int gx2 = clampi(gx + R, 0, (int)w - 1);
    tile[(ly + 2 * R) * tileW + (lx + 2 * R)] =
        src[(ulong)gy2 * w + (ulong)gx2];
  }
  barrier(CLK_LOCAL_MEM_FENCE);
  if ((ulong)gy >= h || (ulong)gx >= w)
    return;
  int gxv = 0, gyv = 0;
  for (int r = 0; r < 3; ++r) {
    for (int c = 0; c < 3; ++c) {
      int v = (int)tile[(ly + r) * tileW + (lx + c)];
      int kr = KERNEL_PREWITT[r * 3 + c];
      int kr_t = KERNEL_PREWITT[(2 - c) * 3 + r];
      gxv += v * kr;
      gyv += v * kr_t;
    }
  }
  int s = abs(gxv) + abs(gyv);
  dst[(ulong)gy * w + (ulong)gx] = (ushort)(s > 65535 ? 65535 : s);
}

kernel void ApplyKernelLocal(global ushort* src, global ushort* dst, ulong h,
                             ulong w, global int* k, int ksize, int rotate,
                             local ushort* tile) {
  const int gx = get_global_id(0);
  const int gy = get_global_id(1);
  const int lx = get_local_id(0);
  const int ly = get_local_id(1);
  const int ldx = get_local_size(0);
  const int ldy = get_local_size(1);
  const int R = ksize / 2;
  const int tileW = ldx + 2 * R;
  int li = clampi(gy - R, 0, (int)h - 1);
  int lj = clampi(gx - R, 0, (int)w - 1);
  tile[ly * tileW + lx] = src[(ulong)li * w + (ulong)lj];
  if (ly < 2 * R) {
    int gy2 = clampi(gy - R + ldy + ly, 0, (int)h - 1);
    tile[(ly + ldy) * tileW + lx] = src[(ulong)gy2 * w + (ulong)lj];
  }
  if (lx < 2 * R) {
    int gx2 = clampi(gx - R + ldx + lx, 0, (int)w - 1);
    tile[ly * tileW + (lx + ldx)] = src[(ulong)li * w + (ulong)gx2];
  }
  if (ly < 2 * R && lx < 2 * R) {
    int gy2 = clampi(gy - R + ldy + ly, 0, (int)h - 1);
    int gx2 = clampi(gx - R + ldx + lx, 0, (int)w - 1);
    tile[(ly + ldy) * tileW + (lx + ldx)] = src[(ulong)gy2 * w + (ulong)gx2];
  }
  barrier(CLK_LOCAL_MEM_FENCE);
  if ((ulong)gy >= h || (ulong)gx >= w)
    return;
  if (rotate != 0) {
    int gvx = 0, gvy = 0;
    for (int r = 0; r < ksize; ++r) {
      for (int c = 0; c < ksize; ++c) {
        int v = (int)tile[(ly + r) * tileW + (lx + c)];
        int kk = k[r * ksize + c];
        int kk_t = k[(ksize - 1 - c) * ksize + r];
        gvx += v * kk;
        gvy += v * kk_t;
      }
    }
    int s = abs(gvx) + abs(gvy);
    dst[(ulong)gy * w + (ulong)gx] = (ushort)(s > 65535 ? 65535 : s);
  } else {
    int g = 0;
    for (int r = 0; r < ksize; ++r) {
      for (int c = 0; c < ksize; ++c) {
        int v = (int)tile[(ly + r) * tileW + (lx + c)];
        int kk = k[r * ksize + c];
        g += v * kk;
      }
    }
    int ga = g < 0 ? -g : g;
    if (ga > 65535)
      ga = 65535;
    dst[(ulong)gy * w + (ulong)gx] = (ushort)ga;
  }
}

kernel void SobelSmooth(global ushort* src, global int* gx, global int* gy,
                        ulong h, ulong w) {
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
    sumx += (int)src[(ulong)i * w + (ulong)x] * KERNEL_SMOOTH[k];
    sumy += (int)src[(ulong)y * w + (ulong)j] * KERNEL_SMOOTH[k];
  }
  gx[(ulong)i * w + (ulong)j] = sumx;
  gy[(ulong)i * w + (ulong)j] = sumy;
}

kernel void PrewittAverage(global ushort* src, global int* gx, global int* gy,
                           ulong h, ulong w) {
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
    sumx += (int)src[(ulong)i * w + (ulong)x] * KERNEL_AVG[k];
    sumy += (int)src[(ulong)y * w + (ulong)j] * KERNEL_AVG[k];
  }
  gx[(ulong)i * w + (ulong)j] = sumx;
  gy[(ulong)i * w + (ulong)j] = sumy;
}

kernel void SobelSmoothLocal(global ushort* src, global int* gx, global int* gy,
                             ulong h, ulong w, local ushort* tile) {
  const int j = get_global_id(0);
  const int i = get_global_id(1);
  const int lx = get_local_id(0);
  const int ly = get_local_id(1);
  const int ldx = get_local_size(0);
  const int ldy = get_local_size(1);
  const int R = 1;
  const int tileW = ldx + 2 * R;
  int li = clampi(i - R, 0, (int)h - 1);
  int lj = clampi(j - R, 0, (int)w - 1);
  tile[ly * tileW + lx] = src[(ulong)li * w + (ulong)lj];
  if (ly < 2 * R) {
    int i2 = clampi(i - R + ldy, 0, (int)h - 1);
    tile[(ly + ldy) * tileW + lx] = src[(ulong)i2 * w + (ulong)lj];
  }
  if (lx < 2 * R) {
    int j2 = clampi(j - R + ldx, 0, (int)w - 1);
    tile[ly * tileW + (lx + ldx)] = src[(ulong)li * w + (ulong)j2];
  }
  if (ly >= ldy - 2 * R && lx >= ldx - 2 * R) {
    int i2 = clampi(i + R, 0, (int)h - 1);
    int j2 = clampi(j + R, 0, (int)w - 1);
    tile[(ly + 2 * R) * tileW + (lx + 2 * R)] = src[(ulong)i2 * w + (ulong)j2];
  }
  barrier(CLK_LOCAL_MEM_FENCE);
  if ((ulong)i >= h || (ulong)j >= w)
    return;
  const int cy = ly + R;
  const int cx = lx + R;
  int sumx = 0, sumy = 0;
  for (int c = 0; c < 3; ++c) {
    int v = (int)tile[cy * tileW + (lx + c)];
    sumx += v * KERNEL_SMOOTH[c];
  }
  for (int r = 0; r < 3; ++r) {
    int v = (int)tile[(ly + r) * tileW + cx];
    sumy += v * KERNEL_SMOOTH[r];
  }
  gx[(ulong)i * w + (ulong)j] = sumx;
  gy[(ulong)i * w + (ulong)j] = sumy;
}

kernel void PrewittAverageLocal(global ushort* src, global int* gx,
                                global int* gy, ulong h, ulong w,
                                local ushort* tile) {
  const int j = get_global_id(0);
  const int i = get_global_id(1);
  const int lx = get_local_id(0);
  const int ly = get_local_id(1);
  const int ldx = get_local_size(0);
  const int ldy = get_local_size(1);
  const int R = 1;
  const int tileW = ldx + 2 * R;
  int li = clampi(i - R, 0, (int)h - 1);
  int lj = clampi(j - R, 0, (int)w - 1);
  tile[ly * tileW + lx] = src[(ulong)li * w + (ulong)lj];
  if (ly < 2 * R) {
    int i2 = clampi(i - R + ldy, 0, (int)h - 1);
    tile[(ly + ldy) * tileW + lx] = src[(ulong)i2 * w + (ulong)lj];
  }
  if (lx < 2 * R) {
    int j2 = clampi(j - R + ldx, 0, (int)w - 1);
    tile[ly * tileW + (lx + ldx)] = src[(ulong)li * w + (ulong)j2];
  }
  if (ly >= ldy - 2 * R && lx >= ldx - 2 * R) {
    int i2 = clampi(i + R, 0, (int)h - 1);
    int j2 = clampi(j + R, 0, (int)w - 1);
    tile[(ly + 2 * R) * tileW + (lx + 2 * R)] = src[(ulong)i2 * w + (ulong)j2];
  }
  barrier(CLK_LOCAL_MEM_FENCE);
  if ((ulong)i >= h || (ulong)j >= w)
    return;
  const int cy = ly + R;
  const int cx = lx + R;
  int sumx = 0, sumy = 0;
  for (int c = 0; c < 3; ++c) {
    int v = (int)tile[cy * tileW + (lx + c)];
    sumx += v * KERNEL_AVG[c];
  }
  for (int r = 0; r < 3; ++r) {
    int v = (int)tile[(ly + r) * tileW + cx];
    sumy += v * KERNEL_AVG[r];
  }
  gx[(ulong)i * w + (ulong)j] = sumx;
  gy[(ulong)i * w + (ulong)j] = sumy;
}

kernel void SepKernelDiff(global int* gx, global int* gy, global ushort* dst,
                          ulong h, ulong w) {
  const int j = get_global_id(0);
  const int i = get_global_id(1);
  if ((ulong)i >= h || (ulong)j >= w)
    return;
  int sumx = 0, sumy = 0;
  for (int k = 0; k < 3; ++k) {
    int x = j + k - 1;
    int y = i + k - 1;
    x = clampi(x, 0, (int)w - 1);
    y = clampi(y, 0, (int)h - 1);
    sumy += gy[(ulong)i * w + (ulong)x] * KERNEL_GRAD[k];
    sumx += gx[(ulong)y * w + (ulong)j] * KERNEL_GRAD[k];
  }
  int sum = (sumx < 0 ? -sumx : sumx) + (sumy < 0 ? -sumy : sumy);
  if (sum > 65535)
    sum = 65535;
  dst[(ulong)i * w + (ulong)j] = (ushort)sum;
}

kernel void SepKernelDiffLocal(global int* gx, global int* gy,
                               global ushort* dst, ulong h, ulong w,
                               local int* tilex, local int* tiley) {
  const int j = get_global_id(0);
  const int i = get_global_id(1);
  const int lx = get_local_id(0);
  const int ly = get_local_id(1);
  const int ldx = get_local_size(0);
  const int ldy = get_local_size(1);
  const int R = 1;
  const int tileW = ldx + 2 * R;
  int li = clampi(i - R, 0, (int)h - 1);
  int lj = clampi(j - R, 0, (int)w - 1);
  tilex[ly * tileW + lx] = gx[(ulong)li * w + (ulong)lj];
  tiley[ly * tileW + lx] = gy[(ulong)li * w + (ulong)lj];
  if (ly < 2 * R) {
    int i2 = clampi(i - R + ldy, 0, (int)h - 1);
    tilex[(ly + ldy) * tileW + lx] = gx[(ulong)i2 * w + (ulong)lj];
    tiley[(ly + ldy) * tileW + lx] = gy[(ulong)i2 * w + (ulong)lj];
  }
  if (lx < 2 * R) {
    int j2 = clampi(j - R + ldx, 0, (int)w - 1);
    tilex[ly * tileW + (lx + ldx)] = gx[(ulong)li * w + (ulong)j2];
    tiley[ly * tileW + (lx + ldx)] = gy[(ulong)li * w + (ulong)j2];
  }
  if (ly >= ldy - 2 * R && lx >= ldx - 2 * R) {
    int i2 = clampi(i + R, 0, (int)h - 1);
    int j2 = clampi(j + R, 0, (int)w - 1);
    tilex[(ly + 2 * R) * tileW + (lx + 2 * R)] = gx[(ulong)i2 * w + (ulong)j2];
    tiley[(ly + 2 * R) * tileW + (lx + 2 * R)] = gy[(ulong)i2 * w + (ulong)j2];
  }
  barrier(CLK_LOCAL_MEM_FENCE);
  if ((ulong)i >= h || (ulong)j >= w)
    return;
  const int cy = ly + R;
  const int cx = lx + R;
  int sumx = 0, sumy = 0;
  for (int c = 0; c < 3; ++c) {
    int v = tiley[cy * tileW + (lx + c)];
    sumy += v * KERNEL_GRAD[c];
  }
  for (int r = 0; r < 3; ++r) {
    int v = tilex[(ly + r) * tileW + cx];
    sumx += v * KERNEL_GRAD[r];
  }
  int sum = (sumx < 0 ? -sumx : sumx) + (sumy < 0 ? -sumy : sumy);
  if (sum > 65535)
    sum = 65535;
  dst[(ulong)i * w + (ulong)j] = (ushort)sum;
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
      x = clampi(x, 0, (int)w - 1);
      y = clampi(y, 0, (int)h - 1);
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
    x = clampi(x, 0, (int)w - 1);
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
    y = clampi(y, 0, (int)h - 1);
    sum += tmp[(ulong)y * w + (ulong)j] * k[ky];
  }
  int iv = (int)floor(sum + 0.5f);
  dst[(ulong)i * w + (ulong)j] =
      (ushort)(iv < 0 ? 0 : (iv > 65535 ? 65535 : iv));
}
