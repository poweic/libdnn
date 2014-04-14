#ifndef __UTILITY_H_
#define __UTILITY_H_

#include <limits>
#include <cstdio>
#include <string>
#include <vector>
#include <sstream>
#include <algorithm>

#include <perf.h>

#include <device_matrix.h>
typedef device_matrix<float> mat;

#define float_min std::numeric_limits<float>::min()
#define float_max std::numeric_limits<float>::max()

#ifdef DEBUG
#define PAUSE { printf("Press Enter key to continue..."); fgetc(stdin); }
#define matlog(x) { printf("\33[34m"#x"\33[0m = [\n"); x.print(); printf("];\n"); }
#define mylog(x) { cout << #x << " = " << x << endl; }
#else
#define PAUSE {}
#define matlog(x) {}
#define mylog(x) {}
#endif

enum ERROR_MEASURE {
  L2ERROR,  /* for binary-classification only */
  CROSS_ENTROPY
};

template <typename T>
void print(const vector<T>& v) {
  cout << "[";
  for (int i=0; i<v.size(); ++i)
    cout << v[i] << ", ";
  cout << "\b\b]" << endl;
}

float str2float(const std::string &s);
std::vector<std::string> split(const std::string &s, char delim);
std::vector<std::string>& split(const std::string &s, char delim, std::vector<std::string>& elems);
std::vector<size_t> splitAsInt(const std::string &s, char delim);
std::vector<size_t> randperm(size_t N);
bool is_number(const std::string& s);
void linearRegression(const std::vector<float> &x, const std::vector<float>& y, float* const &m, float* const &c);

void showAccuracy(size_t nError, size_t nTotal);

#endif
