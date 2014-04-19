#include <dnn-utility.h>

struct Size {
  size_t m, n;
  Size(size_t m, size_t n): m(m), n(n) {}
};

__global__ void downsample_kernel(float *dst, float *src, size_t scale, int H, int W);
__global__ void upsample_kernel(float *dst, float *src, size_t scale, int H, int W);

Size get_convn_size(const mat& data, const mat& kernel, string type = "full");
mat convn(const mat& data, const mat& kernel, string type = "full");
mat xcorrn(const mat& data, const mat& kernel, string type = "full");

mat downsample(const mat& x, size_t scale);
mat upsample(const mat& x, size_t scale);

mat rot180(const mat& x);
float sum_all(const mat& x);

// Codes for unit-testing
void plotL2normInSemilogy();
void test_downsample();
void test_convn(string type, int N);
