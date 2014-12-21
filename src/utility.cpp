// Copyright 2013-2014 [Author: Po-Wei Chou]
// 
// Licensed under the Apache License, Version 2.0 (the "License");
// you may not use this file except in compliance with the License.
// You may obtain a copy of the License at
// 
//     http://www.apache.org/licenses/LICENSE-2.0
// 
// Unless required by applicable law or agreed to in writing, software
// distributed under the License is distributed on an "AS IS" BASIS,
// WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
// See the License for the specific language governing permissions and
// limitations under the License.

#include <utility.h>

int str2int(const std::string &s) {
  return atoi(s.c_str());
}

float str2float(const std::string &s) {
  return atof(s.c_str());
}

std::vector<std::string>& split(const std::string &s, char delim, std::vector<std::string>& elems) {
  std::stringstream ss(s);
  std::string item;
  while(getline(ss, item, delim))
    elems.push_back(item);
  return elems;
}

std::vector<std::string> split(const std::string &s, char delim) {
  std::vector<std::string> elems;
  return split(s, delim, elems);
}

std::vector<size_t> splitAsInt(const std::string &s, char delim) {
  std::vector<std::string> tokens = split(s, delim);
  std::vector<size_t> ints(tokens.size());

  for (size_t i=0; i<ints.size(); ++i)
    ints[i] = ::atoi(tokens[i].c_str());

  return ints;
}

std::vector<size_t> randperm(size_t N) {
  std::vector<size_t> perm(N);

  for (size_t i=0; i<N; ++i)
    perm[i] = i;
  
  std::random_shuffle ( perm.begin(), perm.end() );

  return perm;
}

void linearRegression(const std::vector<float> &x, const std::vector<float>& y, float* const &m, float* const &c) {
  int n = x.size();
  double A=0.0,B=0.0,C=0.0,D=0.0;

  for (int i=0; i<n; ++i) {
    A += x[i];
    B += y[i];
    C += x[i]*x[i];
    D += x[i]*y[i];
  }

  *m = (n*D-A*B) / (n*C-A*A);
  *c = (B-(*m)*A) / n;
} 

bool is_number(const std::string& s) {
  std::string::const_iterator it = s.begin();
  while (it != s.end() && std::isdigit(*it)) ++it;
  return !s.empty() && it == s.end();
}

void showAccuracy(size_t nError, size_t nTotal) {
  size_t nCorr = nTotal - nError;
  printf("Accuracy = %.2f%% ( %lu / %lu ) \n", (float) nCorr / nTotal * 100, nCorr, nTotal);
}
