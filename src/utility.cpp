#include <utility.h>

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

void showAccuracy(size_t nError, size_t nTotal) {
  size_t nCorr = nTotal - nError;
  printf("Accuracy = %.2f%% ( %lu / %lu ) \n", (float) nCorr / nTotal * 100, nCorr, nTotal);
}
