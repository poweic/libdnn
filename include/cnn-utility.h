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

  friend SIZE max(const SIZE& s1, const SIZE& s2) {
    return SIZE(max(s1.m, s2.m), max(s1.n, s2.n));
  }

  friend ostream& operator << (ostream& os, const SIZE& s) {
    os << setw(3) << s.m <<  " x " << std::left << setw(3) << s.n;
    return os;
  }
};

void gogo();

mat gaussian_kernel(int h, int w);
void showImage(const mat& x);

vector<mat> reshapeVectors2Images(const mat& data, const SIZE s);
mat reshapeImages2Vectors(const vector<mat>& images);

SIZE parseInputDimension(const string &m_by_n);

__global__ void downsample_kernel(float *dst, float *src, size_t scale, int H, int W);
__global__ void upsample_kernel(float *dst, float *src, size_t scale, int H, int W);

SIZE get_convn_size(SIZE data, SIZE kernel, ConvType type = FULL);
SIZE get_convn_size(const mat& data, const mat& kernel, ConvType type = FULL);

mat convn_2(const mat& data, const mat& kernel, SIZE k);
mat convn(const mat& data, const mat& kernel, SIZE s, ConvType type);
mat convn(const mat& data, const mat& kernel, ConvType type = FULL);
// mat xcorrn(const mat& data, const mat& kernel, ConvType type = FULL);
mat cross_convn(const mat& dataIn, const mat& dataOut, SIZE imgIn, SIZE imgOut, ConvType type);

vector<mat> de_concat(const mat& concated_features, int n);
mat concat(const vector<mat>& smalls);

mat downsample(const mat& x, size_t scale);
mat downsample(const mat& x, size_t scale, SIZE s);
// mat upsample(const mat& x, size_t scale, SIZE s = SIZE(0,0) );

mat upsample(const mat& x, SIZE s, SIZE img);
mat upsample(const mat& x, SIZE s);
mat upsample(const mat& x, size_t scale);

mat rot180(const mat& x);
float sum_all(const mat& x);

// Codes for unit-testing
void plotL2normInSemilogy();
void test_batch_convn_full();
void test_downsample();
void test_convn(ConvType type);
void test_convn_with_and_without_shm(ConvType type, const int N = 10000);
void test_valid_shm_vs_valid_2();
void test_reshape_images_between_vectors();
void benchmark_valid_and_valid_shm();
void benchmark_batch_convn();
