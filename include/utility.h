#ifndef _UTILITY_H_
#define _UTILITY_H_
#include <iostream>
#include <sstream>
#include <string>
#include <cstdlib>
#include <cstdio>
#include <vector>
#include <cassert>
#include <algorithm>
#include <cmath>
#include <csignal>
#include <sys/stat.h>
#include <fstream>

#include <color.h>
using namespace std;

#define foreach(i, arr) for (size_t i=0; i<arr.size(); ++i)
#define range(i, size) for (size_t i=0; i<size; ++i)
#define reverse_foreach(i, arr) for (int i=(int)arr.size()-1; i>=0; --i)
#define FLOAT_MIN (std::numeric_limits<float>::lowest())

#ifdef DEBUG
  #define debug(token) {cout << #token " = " << token << endl;}
#else
  #define debug(token) {}
#endif

#define mylog(token) {cout << #token " = " << token << endl;}

#define checkNAN(x) assert((x) == (x))
#define warnNAN(x) { if (x!=x) cout << #x" is NAN" << endl; }

#define __DIVIDER__ "=========================================================="

string int2str(int n);
int str2int(const string& str);
float str2float(const string& str);
double str2double(const string& str);
string getValueStr(string& str);
string join(const vector<string>& arr);

std::string exec(std::string cmd);

bool isInt(string str);

vector<string> split(const string &s, char delim);
vector<string>& split(const string &s, char delim, vector<string>& elems);

inline bool exists (const string& name) {
  struct stat buffer;   
  return (stat (name.c_str(), &buffer) == 0); 
}

// ====================================
// ===== Vector Utility Functinos =====
// ====================================
template <typename T>
void fillwith(vector<T>& v, T val) {
  std::fill(v.begin(), v.end(), val);
}

template <typename T, typename S>
vector<T> max(S a, const vector<T>& v) {
  vector<T> m(v.size());
  foreach (i, m)
    m[i] = MAX(a, v[i]);
  return m;
}

template <typename T, typename S>
vector<T> min(S a, const vector<T>& v) {
  vector<T> m(v.size());
  foreach (i, m)
    m[i] = MIN(a, v[i]);
  return m;
}

template <typename T>
T norm(const vector<T>& v) {
  T sum = 0;
  foreach (i, v)
    sum += pow(v[i], (T) 2);
  return sqrt(sum);
}

vector<size_t> randperm(size_t N);

template <typename T>
void normalize(vector<T>& v, int type = 2) {
  // T n = (type == 2) ? norm(v) : ext::sum(v);
  T n = norm(v);
  if (n == 0) return;

  T normalizer = 1/n;
  foreach (i, v)
    v[i] *= normalizer;
}

template <typename T>
void print(const vector<T>& v, size_t n_digits = 3) {

  string format = "%." + int2str(n_digits) + "f ";
  printf("[");
  foreach (i, v)
    printf(format.c_str(), v[i]);
  printf("]\n");
}

// ==============================================
// ===== Split vectors in vector of vectors =====
// ==============================================
template <typename T>
vector<vector<T> > split(const vector<T>& v, const vector<size_t>& lengths) {
  vector<vector<T> > result;

  size_t totalLength = 0;
  for (size_t i=0; i<lengths.size(); ++i)
    totalLength += lengths[i];

  assert(totalLength <= v.size());

  size_t offset = 0;
  for (size_t i=0; i<lengths.size(); ++i) {
    size_t l = lengths[i];
    vector<T> sub_v(l);
    for (size_t j=0; j<l; ++j)
      sub_v[j] = v[j+offset];

    result.push_back(sub_v);
    offset += l;
  }

  return result;
}

template <typename T> int sign(T val) {
  return (T(0) < val) - (val < T(0));
}

void doPause();

namespace bash {
  vector<string> ls(string path);
}

string replace_all(const string& str, const string &token, const string &s);

#endif // _UTILITY_H_
