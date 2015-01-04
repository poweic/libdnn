#ifndef _CNN_UTILITY_H_
#define _CNN_UTILITY_H_

#include <dnn-utility.h>
#include <iomanip>

enum ConvType {
  SAME,
  SAME_SHM,
  VALID,
  VALID_SHM,
  FULL,
  FULL_SHM
};

struct SIZE {
  size_t m, n;
  SIZE(): m(0), n(0) {}
  SIZE(size_t m, size_t n): m(m), n(n) {}

  bool operator == (const SIZE& rhs) const { return m == rhs.m && n == rhs.n; }

  SIZE operator + (const SIZE& rhs) const { return SIZE(m + rhs.m, n + rhs.n); }
  SIZE operator - (const SIZE& rhs) const { return SIZE(m - rhs.m, n - rhs.n); }

  SIZE operator + (size_t x) const { return SIZE(m + x, n + x); }
  SIZE operator - (size_t x) const { return SIZE(m - x, n - x); }
  SIZE operator * (size_t x) const { return SIZE(m * x, n * x); }
  SIZE operator / (size_t x) const { return SIZE(m / x, n / x); }

  size_t area() const { return m * n; }

  friend SIZE max(const SIZE& s1, const SIZE& s2) {
    return SIZE(max(s1.m, s2.m), max(s1.n, s2.n));
  }

  operator string () {
    char buffer[12];
    sprintf(buffer, "%2lu x %-2lu", m, n);
    return buffer;
  }

  friend ostream& operator << (ostream& os, const SIZE& s) {
    os << setw(3) << s.m <<  " x " << std::left << setw(3) << s.n;
    return os;
  }
};

mat gaussian_kernel(int h, int w);
void showImage(const mat& x);

vector<mat> reshapeVectors2Images(const mat& data, const SIZE s);
mat reshapeImages2Vectors(const vector<mat>& images);

SIZE parseImageDimension(const string &m_by_n);

__global__ void downsample_kernel(float *dst, float *src, size_t scale, int H, int W);
__global__ void upsample_kernel(float *dst, float *src, size_t scale, int H, int W);

SIZE get_convn_size(SIZE data, SIZE kernel, ConvType type = FULL);
SIZE get_convn_size(const mat& data, const mat& kernel, ConvType type = FULL);

mat convn_2(const mat& data, const mat& kernels, size_t i, size_t batch_idx, size_t N, SIZE imgIn, SIZE imgOut);
mat convn(const mat& data, const mat& kernel, SIZE s, ConvType type);
mat convn(const mat& data, const mat& kernel, ConvType type = FULL);

mat cross_convn(const mat& dataIn, const mat& dataOut, SIZE imgIn, SIZE imgOut, ConvType type);

vector<mat> de_concat(const mat& concated_features, int n);
mat concat(const vector<mat>& smalls);

mat downsample(const mat& x, size_t scale, SIZE s);

mat upsample(const mat& x, SIZE s, SIZE img);

mat rot180(const mat& x);

#endif // _CNN_UTILITY_H_
