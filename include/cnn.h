#include <dnn-utility.h>

void plotL2normInSemilogy();
mat rand(int m, int n);
void test_convn(string type, int N);
mat convn(const mat& data, const mat& kernel, string type = "full");
mat downsample(const mat& x, size_t scale);
