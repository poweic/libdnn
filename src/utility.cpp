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
using namespace std;

void SetGpuCardId(size_t card_id) {
  int nDevices;
  cudaGetDeviceCount(&nDevices);

  card_id = min(nDevices - 1, (int) card_id);

  clog << "\33[34m[Info]\33[0m Using GPU card " << card_id << endl;
  cudaSetDevice(card_id);
}

int str2int(const string &s) {
  return atoi(s.c_str());
}

float str2float(const string &s) {
  return atof(s.c_str());
}

vector<string>& split(const string &s, char delim, vector<string>& elems) {
  stringstream ss(s);
  string item;
  while(getline(ss, item, delim))
    elems.push_back(item);
  return elems;
}

vector<string> split(const string &s, char delim) {
  vector<string> elems;
  return split(s, delim, elems);
}

vector<size_t> splitAsInt(const string &s, char delim) {
  vector<string> tokens = split(s, delim);
  vector<size_t> ints(tokens.size());

  for (size_t i=0; i<ints.size(); ++i)
    ints[i] = ::atoi(tokens[i].c_str());

  return ints;
}

vector<size_t> randperm(size_t N) {
  vector<size_t> perm(N);

  for (size_t i=0; i<N; ++i)
    perm[i] = i;
  
  random_shuffle ( perm.begin(), perm.end() );

  return perm;
}

void linearRegression(const vector<float> &x, const vector<float>& y, float* const &m, float* const &c) {
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

bool is_number(const string& s) {
  string::const_iterator it = s.begin();
  while (it != s.end() && isdigit(*it)) ++it;
  return !s.empty() && it == s.end();
}

void showAccuracy(size_t nError, size_t nTotal) {
  size_t nCorr = nTotal - nError;
  printf("Accuracy = %.2f%% ( %lu / %lu ) \n", (float) nCorr / nTotal * 100, nCorr, nTotal);
}

size_t parseInputDimension(const string &input_dim) {
  size_t dim = 1;
  for (auto d : splitAsInt(input_dim, 'x'))
    dim *= d;
  return dim;
}

SIZE parseImageDimension(const string &m_by_n) {
  vector<size_t> dims = splitAsInt(m_by_n, 'x');

  if (dims.size() < 2)
    throw runtime_error(RED_ERROR + "For convolutional neural network, "
	"please use --input-dim like this: 32x32");

  return SIZE(dims[0], dims[1]);
}
