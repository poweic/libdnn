#include <dnn-utility.h>

mat rand(int m, int n);
void test_convn(string type, int N);
mat convn(const mat& data, const mat& kernel, string type = "full");

