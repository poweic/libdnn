#include <dnn-utility.h>
#include <iomanip>

struct SIZE {
  size_t m, n;
  SIZE(): m(0), n(0) {}
  SIZE(size_t m, size_t n): m(m), n(n) {}

  SIZE operator + (const SIZE& rhs) const { return SIZE(m - rhs.m, n - rhs.n); }
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

void benchmark();
void playground();

vector<mat> reshapeVectors2Images(const mat& data, const SIZE s);
mat reshapeImages2Vectors(const vector<mat>& images);

SIZE parseInputDimension(const string &m_by_n);

__global__ void downsample_kernel(float *dst, float *src, size_t scale, int H, int W);
__global__ void upsample_kernel(float *dst, float *src, size_t scale, int H, int W);

SIZE get_convn_size(SIZE data, SIZE kernel, string type = "full");
SIZE get_convn_size(const mat& data, const mat& kernel, string type = "full");
mat convn(const mat& data, const mat& kernel, string type = "full", int N_STREAM = 4);
mat xcorrn(const mat& data, const mat& kernel, string type = "full");

mat downsample(const mat& x, size_t scale);
mat upsample(const mat& x, size_t scale);

mat rot180(const mat& x);
float sum_all(const mat& x);

// Codes for unit-testing
void plotL2normInSemilogy();
void test_downsample();
void test_convn(string type, int N);
