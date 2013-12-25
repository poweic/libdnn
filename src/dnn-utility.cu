#include <dnn-utility.h>

void zeroOneLabels(const mat& label) {
  thrust::device_ptr<float> dptr(label.getData());
  thrust::host_vector<float> y(dptr, dptr + label.size());

  map<float, bool> classes;
  for (size_t i=0; i<y.size(); ++i)
    classes[y[i]] = true;

  size_t nClasses = classes.size();
  assert(nClasses == 2);

  thrust::replace(dptr, dptr + label.size(), -1, 0);
}

size_t zeroOneError(const mat& predict, const mat& label) {
  assert(predict.size() == label.size());

  size_t L = label.size();
  thrust::device_ptr<float> l_ptr(label.getData());
  thrust::device_ptr<float> p_ptr(predict.getData());

  thrust::device_vector<float> p_vec(L);
  thrust::transform(p_ptr, p_ptr + L, p_vec.begin(), func::to_zero_one<float>());

  float nError = thrust::inner_product(p_vec.begin(), p_vec.end(), l_ptr, 0.0, thrust::plus<float>(), thrust::not_equal_to<float>());
  return nError;
}

void print(const std::vector<mat>& vm) {
  for (size_t i=0; i<vm.size(); ++i) {
    printf("rows = %lu, cols = %lu\n", vm[i].getRows(), vm[i].getCols());
    vm[i].print();
  }
}

void showSummary(const mat& data, const mat& label) {
  size_t input_dim  = data.getCols();
  size_t output_dim = label.getCols();
  size_t nData	    = data.getRows();

  printf("---------------------------------------------\n");
  printf("  Number of input feature (data) %10lu \n", nData);
  printf("  Dimension of  input feature    %10lu \n", input_dim);
  printf("  Dimension of output feature    %10lu \n", output_dim);
  printf("---------------------------------------------\n");
}

void showAccuracy(size_t nError, size_t nTotal) {
  size_t nCorr = nTotal - nError;
  printf("Accuracy = %.2f%% ( %lu / %lu ) \n", (float) nCorr / nTotal * 100, nCorr, nTotal);
}

void getDataAndLabels(string train_fn, mat& data, mat& labels) {

  string tmpfn = getTempFilename();

  exec("cut -f 1 -d ':' " + train_fn + " > " + tmpfn);
  labels = mat(tmpfn);

  exec("cut -f 2 -d ':' " + train_fn + " > " + tmpfn);
  data	 = mat(tmpfn);

  exec("rm " + tmpfn);
}

bool isFileSparse(string train_fn) {
  ifstream fin(train_fn.c_str());
  string line;
  std::getline(fin, line);
  return line.find(':') != string::npos;
}

string getTempFilename() {
  std::stringstream ss;
  time_t seconds;
  time(&seconds);
  ss << seconds;
  return ".tmp." + ss.str();
}

void exec(string command) {
  system(command.c_str());
}

float str2float(const string &s) {
  return atof(s.c_str());
}

size_t getLineNumber(ifstream& fin) {
  int previous_pos = fin.tellg();
  string a;
  size_t n = 0;
  while(std::getline(fin, a) && ++n);
  fin.clear();
  fin.seekg(previous_pos);
  return n;
}

size_t findMaxDimension(ifstream& fin) {
  int previous_pos = fin.tellg();

  string token;
  size_t maxDimension = 0;
  while (fin >> token) {
    size_t pos = token.find(':');
    if (pos == string::npos)
      continue;

    size_t dim = atoi(token.substr(0, pos).c_str());
    if (dim > maxDimension)
      maxDimension = dim;
  }

  fin.clear();
  fin.seekg(previous_pos);

  return maxDimension;
}

void readSparseFeature(ifstream& fin, float* data, float* labels, size_t rows, size_t cols) {

  string line, token;
  size_t i = 0;
  while (std::getline(fin, line)) {
    stringstream ss(line);
  
    ss >> token;
    labels[i] = str2float(token);

    while (ss >> token) {
      size_t pos = token.find(':');
      if (pos == string::npos)
	continue;

      size_t j = str2float(token.substr(0, pos)) - 1;
      float value = str2float(token.substr(pos + 1));
      
      data[j * rows + i] = value;
    }
    ++i;
  }
}

void readDenseFeature(ifstream& fin, float* data, float* labels, size_t rows, size_t cols) {

  string line, token;
  size_t i = 0;
  while (std::getline(fin, line)) {
    stringstream ss(line);
  
    ss >> token;
    labels[i] = str2float(token);

    size_t j = 0;
    while (ss >> token)
      data[(j++) * rows + i] = str2float(token);
    ++i;
  }
}

size_t findDimension(ifstream& fin) {

  size_t dim = 0;

  int previous_pos = fin.tellg();

  string line;
  std::getline(fin, line);
  stringstream ss(line);

  // First token is class label
  string token;
  ss >> token;

  while (ss >> token)
    ++dim;
  
  fin.clear();
  fin.seekg(previous_pos);

  return dim;
}

void readFeature(const string &fn, mat& X, mat& y) {
  ifstream fin(fn.c_str());

  bool isSparse = isFileSparse(fn);
  size_t cols = isSparse ? findMaxDimension(fin) : findDimension(fin);
  size_t rows = getLineNumber(fin);

  printf("rows = %lu, cols = %lu \n", rows, cols);

  float* data = new float[rows * cols];
  float* labels = new float[rows];
  memset(data, 0, sizeof(float) * rows * cols);

  if (isSparse)
    readSparseFeature(fin, data, labels, rows, cols);
  else
    readDenseFeature(fin, data, labels, rows, cols);

  X = mat(data, rows, cols);
  y = mat(labels, rows, 1);

  delete [] data;
  delete [] labels;

  fin.close();
}


bool isLabeled(const mat& labels) {

  size_t L = labels.size();

  thrust::device_vector<float> zero_vec(L, 0);
  thrust::device_ptr<float> label_ptr(labels.getData());

  bool isAllZero = thrust::equal(label_ptr, label_ptr + L, zero_vec.begin());
  return !isAllZero;
}

