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

#define __WHERE__ (string("In function \"") + __func__ + string("\" (at ") + \
    __FILE__ + string(":") + to_string(__LINE__) + string("): "))

#define RED_ERROR (util::red("[Error] ") + __WHERE__ )
#define YELLOW_WARNING (util::yellow("[WARNING] ") + __WHERE__ )

#ifdef DEBUG
#define PAUSE { printf("Press Enter key to continue..."); fgetc(stdin); }
#define matlog(x) { printf(#x": "); (x).status(); printf("\33[34m"#x"\33[0m = [\n"); (x).print(); printf("];\n"); }
// #define matlog(x) { printf(#x" = [\n"); (x).print(); printf("];\n"); }
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

void SetGpuCardId(size_t card_id);

/*#include <sys/stat.h>
long getFileSize(std::string filename) {
  struct stat stat_buf;
  int rc = stat(filename.c_str(), &stat_buf);
  return rc == 0 ? stat_buf.st_size : -1;
}*/
namespace util {
  inline string red(const string& str)	  { return "\33[31m" + str + "\33[0m"; }
  inline string green(const string& str)  { return "\33[32m" + str + "\33[0m"; }
  inline string yellow(const string& str) { return "\33[33m" + str + "\33[0m"; }
  inline string blue(const string& str)   { return "\33[34m" + str + "\33[0m"; }
  inline string purple(const string& str) { return "\33[35m" + str + "\33[0m"; }
  inline string cyan(const string& str)   { return "\33[36m" + str + "\33[0m"; }
};

template <typename T>
void print(const vector<T>& v) {
  cout << "[";
  for (int i=0; i<v.size(); ++i)
    cout << v[i] << ", ";
  cout << "\b\b]" << endl;
}

template <typename T>
string to_string(T n) {
  stringstream ss;
  ss << n;
  return ss.str();
}

int str2int(const std::string &s);
float str2float(const std::string &s);
std::vector<std::string> split(const std::string &s, char delim);
std::vector<std::string>& split(const std::string &s, char delim, std::vector<std::string>& elems);
std::vector<size_t> splitAsInt(const std::string &s, char delim);
std::vector<size_t> randperm(size_t N);
bool is_number(const std::string& s);
void linearRegression(const std::vector<float> &x, const std::vector<float>& y, float* const &m, float* const &c);

void showAccuracy(size_t nError, size_t nTotal);

#endif
