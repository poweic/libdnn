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

mat& calcError(const mat& output, mat& trainY, size_t offset, size_t nData) {

  mat error(nData, trainY.getCols());

  device_matrix<float>::cublas_geam(
      CUBLAS_OP_N, CUBLAS_OP_N,
      nData, trainY.getCols(),
      1.0, output.getData(), nData,
      -1.0, trainY.getData() + offset, trainY.getRows(),
      error.getData(), nData);

  return error;
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

std::vector<size_t> randshuf(size_t N) {
  std::vector<size_t> perm(N);

  for (size_t i=0; i<N; ++i)
    perm[i] = i;
  
  std::random_shuffle ( perm.begin(), perm.end() );

  return perm;
}

void shuffleFeature(float* const data, float* const labels, int rows, int cols) {

  printf("Random shuffling features for latter training...\n");

  perf::Timer timer;
  timer.start();

  std::vector<size_t> perm = randshuf(rows);

  float* tmp_data = new float[rows*cols];
  float* tmp_labels = new float[rows];

  memcpy(tmp_data, data, sizeof(float) * rows * cols);
  memcpy(tmp_labels, labels, sizeof(float) * rows);

  for (size_t i=0; i<rows; ++i) {
    size_t from = i;
    size_t to = perm[i];

    for (size_t j=0; j<cols; ++j)
      data[j * rows + to] = tmp_data[j * rows + from];

    labels[to] = tmp_labels[from];
  }

  delete [] tmp_data;
  delete [] tmp_labels;

  timer.elapsed();
}

void splitIntoTrainingAndValidationSet(
    mat& trainX, mat& trainY,
    mat& validX, mat& validY,
    int ratio,
    mat& X, mat& y) {

  size_t rows = X.getRows(),
	 cols = X.getCols();
  
  float *h_X = new float[rows*cols],
        *h_y = new float[rows];

  CCE(cudaMemcpy(h_X, X.getData(), sizeof(float) * X.size(), cudaMemcpyDeviceToHost));
  CCE(cudaMemcpy(h_y, y.getData(), sizeof(float) * y.size(), cudaMemcpyDeviceToHost));

  float* h_trainX, *h_trainY, *h_validX, *h_validY;
  size_t nTrain, nValid;
  splitIntoTrainingAndValidationSet(
      h_trainX, h_trainY, nTrain,
      h_validX, h_validY, nValid,
      ratio,
      h_X, h_y,
      rows, cols);

  trainX = mat(h_trainX, nTrain, cols);
  trainY = mat(h_trainY, nTrain, 1);

  validX = mat(h_validX, nValid, cols);
  validY = mat(h_validY, nValid, 1);

  delete [] h_X;
  delete [] h_y;
}

void splitIntoTrainingAndValidationSet(
    float* &trainX, float* &trainY, size_t& nTrain,
    float* &validX, float* & validY, size_t& nValid,
    int ratio, /* ratio of training / validation */
    const float* const data, const float* const labels,
    int rows, int cols) {

  nValid = rows / ratio;
  nTrain = rows - nValid;
  printf("nTrain = %lu, nValid = %lu\n", nTrain, nValid);

  trainX = new float[nTrain * cols];
  trainY = new float[nTrain];

  validX = new float[nValid * cols];
  validY = new float[nValid];

  for (size_t i=0; i<nTrain; ++i) {
    for (size_t j=0; j<cols; ++j)
      trainX[j * nTrain + i] = data[j * rows + i];
    trainY[i] = labels[i];
  }

  for (size_t i=0; i<nValid; ++i) {
    for (size_t j=0; j<cols; ++j)
      validX[j * nValid + i] = data[j * rows + i + nTrain];
    validY[i] = labels[i + nTrain];
  }
}

void getFeature(const string &fn, mat& X, mat& y) {

  float* data, *labels;
  int rows, cols;
  readFeature(fn, data, labels, rows, cols);

  X = mat(data, rows, cols);
  y = mat(labels, rows, 1);

  delete [] data;
  delete [] labels;
}

void readFeature(const string &fn, float* &data, float* &labels, int &rows, int &cols) {
  ifstream fin(fn.c_str());

  bool isSparse = isFileSparse(fn);

  cols = isSparse ? findMaxDimension(fin) : findDimension(fin);
  rows = getLineNumber(fin);
  data = new float[rows * cols];
  labels = new float[rows];

  memset(data, 0, sizeof(float) * rows * cols);

  if (isSparse)
    readSparseFeature(fin, data, labels, rows, cols);
  else
    readDenseFeature(fin, data, labels, rows, cols);

  shuffleFeature(data, labels, rows, cols);

  fin.close();
}


bool isLabeled(const mat& labels) {

  size_t L = labels.size();

  thrust::device_vector<float> zero_vec(L, 0);
  thrust::device_ptr<float> label_ptr(labels.getData());

  bool isAllZero = thrust::equal(label_ptr, label_ptr + L, zero_vec.begin());
  return !isAllZero;
}

