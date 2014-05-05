#include <utility.h>

/*! \brief divide --struct into cnn-structure and nn-structure
 * A network is a Convolutional Neural Network followed by a traditional Neural
 * Network. 
 * For example, --struct provided by user is something like "8x5x5-2s-256-128".
 * This function will find the hyphen "-" followed by "2s" and split the string
 * into 2 parts. The 1st part will be "8x5x5-2s" and the 2nd part will be "256-128"
 *
 * Post Condition: structure is always equal to cnn_struct + "-" + nn_struct.
 *
 * \param structure network structure specified by --struct in command line
 * \param cnn_struct represents the structure of Convolutional Neural Network
 * \param nn_struct  represents the structure of traditional Neural Network
 * */
void parseNetworkStructure(const string &structure,
    string& cnn_struct, string& nn_struct) {

  //printf("    structure: %s\n", structure.c_str());

  // the network structure 
  int pos_of_s = structure.find_last_of("s"),	// sub-sampling layer
      pos_of_x = structure.find_last_of("x");	// convolutional layer

  size_t pos = structure.find("-", max(pos_of_s, pos_of_x));

  cnn_struct = structure.substr(0, pos);

  if (pos == string::npos)
    nn_struct = "";
  else
    nn_struct  = structure.substr(pos+1);

  //printf("CNN structure: %s\n", cnn_struct.c_str());
  //printf("NN  structure: %s\n", nn_struct.c_str());
}

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
