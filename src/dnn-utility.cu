#include <dnn-utility.h>

map<int, int> getLabelMapping(const mat& labels) {
  thrust::device_ptr<float> dptr(labels.getData());
  thrust::host_vector<float> y(dptr, dptr + labels.size());

  map<int, int> classes;
  for (size_t i=0; i<y.size(); ++i)
    classes[(int) y[i]] = 1;

  int counter = 0;
  map<int, int>::iterator itr = classes.begin();
  for (; itr != classes.end(); ++itr)
    itr->second = ++counter;

  return classes;
}

size_t getClassNumber(const DataSet& data) {
  thrust::device_ptr<float> dptr(data.y.getData());
  thrust::host_vector<float> y(dptr, dptr + data.y.size());

  map<float, bool> classes;
  for (size_t i=0; i<y.size(); ++i)
    classes[y[i]] = true;

  return classes.size();
}

/*float max(const mat& v) {
  thrust::device_ptr<float> vPtr(v.getData());
  thrust::device_ptr<float> maxPtr = thrust::max_element(vPtr, vPtr + v.size());
  thrust::host_vector<float> hMaxPtr(maxPtr, maxPtr + 1);
  return hMaxPtr[0];
}*/

mat getStandardLabels(const mat& labels) {
  // Assume class label index start from 1, and there's no skipping.
  assert(labels.getCols() == 1);

  size_t nData = labels.getRows();

  float* hy = new float[nData];
  std::vector<bool> replaced(nData, false);

  CCE(cudaMemcpy(hy, labels.getData(), sizeof(float) * nData, cudaMemcpyDeviceToHost));

  // Replace labels to 1, 2, 3, N, using mapping
  map<int, int> classes = getLabelMapping(labels);

  map<int, int>::const_iterator itr = classes.begin();
  for (; itr != classes.end(); ++itr) {
    int from = itr->first,
	to   = itr->second;
    
    for (size_t i=0; i<nData; ++i) {
      if (!replaced[i] && hy[i] == from) {
	replaced[i] = true;
	hy[i] = to;
      }
    }
  }

  mat sLabels(hy, nData, 1);
  delete [] hy;

  return sLabels;
}

mat label2PosteriorProb(const mat& labels) {

  map<int, int> classes = getLabelMapping(labels);
  size_t nClasses = classes.size();
  size_t nData = labels.getRows();

  // Convert labels to posterior probabilities
  float* h_prob = new float[nData * nClasses];
  memset(h_prob, 0, sizeof(float) * nData * nClasses);

  vector<float> hy = copyToHost(labels);
  for (size_t i=0; i<nData; ++i)
    h_prob[(size_t) (hy[i] - 1) * nData + i] = 1;

  mat probs(h_prob, nData, nClasses);

  delete [] h_prob;

  return probs;
}

mat posteriorProb2Label(const mat& prob) {

  assert(prob.getCols() > 1);

  size_t rows = prob.getRows(),
	 cols = prob.getCols();

  float* h_prob = new float[prob.size()];
  float* h_labels  = new float[rows];
  CCE(cudaMemcpy(h_prob, prob.getData(), sizeof(float) * prob.size(), cudaMemcpyDeviceToHost));

  for (size_t i=0; i<rows; ++i) {

    float max = -1e10;
    size_t maxIdx = 0;

    for (size_t j=0; j<cols; ++j) {
      if (h_prob[j * rows + i] > max) {
	max = h_prob[j * rows + i];
	maxIdx = j;
      }
    }

    h_labels[i] = maxIdx + 1;
  }

  mat labels(h_labels, rows, 1);

  delete [] h_prob;
  delete [] h_labels;

  return labels;
}

vector<float> copyToHost(const mat& m) {
  vector<float> hm(m.size());
  thrust::device_ptr<float> dPtr(m.getData());
  thrust::copy(dPtr, dPtr + m.size(), hm.begin());
  return hm;
}

size_t countDifference(const mat& m1, const mat& m2) {

  assert(m1.size() == m2.size());

  size_t L = m1.size();
  thrust::device_ptr<float> ptr1(m1.getData());
  thrust::device_ptr<float> ptr2(m2.getData());

  return thrust::inner_product(ptr1, ptr1 + L, ptr2, 0.0, thrust::plus<float>(), thrust::not_equal_to<float>());
}

size_t zeroOneError(const mat& prob, const mat& label, ERROR_MEASURE errorMeasure) {
  assert(prob.getRows() == label.getRows());
  assert(label.getCols() == 1);

  size_t nError = 0;

  if (errorMeasure == L2ERROR) {
    // nError = countDifference(label, prob);
  }
  else {
    mat L = posteriorProb2Label(prob);
    nError = countDifference(L, label);

    // matlog(prob); matlog(L); PAUSE;
  }

  return nError;
}

mat& calcError(const mat& output, const mat& trainY, size_t offset, size_t nData) {

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

void showSummary(const DataSet& data) {
  size_t input_dim  = data.X.getCols();
  size_t nData	    = data.X.getRows();
  size_t nClasses   = data.prob.getCols();

  printf("+--------------------------------+----------+\n");
  printf("| Number of classes              |%9lu |\n", nClasses);
  printf("| Number of input feature (data) |%9lu |\n", nData);
  printf("| Dimension of  input feature    |%9lu |\n", input_dim);
  printf("+--------------------------------+----------+\n");
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

void shuffleFeature(DataSet& data) {

  float *h_X = new float[data.X.size()],
	*h_y = new float[data.y.size()];

  CCE(cudaMemcpy(h_X, data.X.getData(), sizeof(float) * data.X.size(), cudaMemcpyDeviceToHost));
  CCE(cudaMemcpy(h_y, data.y.getData(), sizeof(float) * data.y.size(), cudaMemcpyDeviceToHost));

  shuffleFeature(h_X, h_y, data.X.getRows(), data.X.getCols());

  delete [] h_X;
  delete [] h_y;
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
    DataSet& train, DataSet& valid,
    DataSet& data, int ratio) {

  size_t rows = data.X.getRows(),
	 inputDim = data.X.getCols(),
	 outputDim = data.prob.getCols();
  
  float *h_X = new float[rows*inputDim],
	*h_y = new float[rows],
        *h_prob = new float[rows*outputDim];

  CCE(cudaMemcpy(h_X, data.X.getData(), sizeof(float) * data.X.size(), cudaMemcpyDeviceToHost));
  CCE(cudaMemcpy(h_y, data.y.getData(), sizeof(float) * data.y.size(), cudaMemcpyDeviceToHost));
  CCE(cudaMemcpy(h_prob, data.prob.getData(), sizeof(float) * data.prob.size(), cudaMemcpyDeviceToHost));

  float* h_trainX, *h_trainY, *h_trainProb, *h_validX, *h_validY, *h_validProb;
  size_t nTrain, nValid;
  splitIntoTrainingAndValidationSet(
      h_trainX, h_trainProb, h_trainY, nTrain,
      h_validX, h_validProb, h_validY, nValid,
      ratio,
      h_X, h_prob, h_y,
      rows, inputDim, outputDim);

  train.X    = mat(h_trainX   , nTrain, inputDim );
  train.prob = mat(h_trainProb, nTrain, outputDim);
  train.y    = mat(h_trainY   , nTrain, 1        );

  valid.X    = mat(h_validX   , nValid, inputDim );
  valid.prob = mat(h_validProb, nValid, outputDim);
  valid.y    = mat(h_validY   , nValid, 1	 );

  delete [] h_X;
  delete [] h_prob;
  delete [] h_y;

  delete [] h_trainX;
  delete [] h_trainY;
  delete [] h_trainProb;

  delete [] h_validX;
  delete [] h_validY;
  delete [] h_validProb;
}

void splitIntoTrainingAndValidationSet(
    float* &trainX, float* &trainProb, float* &trainY, size_t& nTrain,
    float* &validX, float* &validProb, float* &validY, size_t& nValid,
    int ratio, /* ratio of training / validation */
    const float* const data, const float* const prob, const float* const labels,
    int rows, int inputDim, int outputDim) {

  nValid = rows / ratio;
  nTrain = rows - nValid;
  printf("| nTrain                         |%9lu |\n", nTrain);
  printf("| nValid                         |%9lu |\n", nValid);

  trainX    = new float[nTrain * inputDim];
  trainProb = new float[nTrain * outputDim];
  trainY    = new float[nTrain];

  validX    = new float[nValid * inputDim];
  validProb = new float[nValid * outputDim];
  validY    = new float[nValid];

  for (size_t i=0; i<nTrain; ++i) {
    for (size_t j=0; j<inputDim; ++j)
      trainX[j * nTrain + i] = data[j * rows + i];
    for (size_t j=0; j<outputDim; ++j)
      trainProb[j * nTrain + i] = prob[j * rows + i];
    trainY[i] = labels[i];
  }

  for (size_t i=0; i<nValid; ++i) {
    for (size_t j=0; j<inputDim; ++j)
      validX[j * nValid + i] = data[j * rows + i + nTrain];
    for (size_t j=0; j<outputDim; ++j)
      validProb[j * nValid + i] = prob[j * rows + i + nTrain];
    validY[i] = labels[i + nTrain];
  }
}

void getFeature(const string &fn, DataSet& dataset) {

  float* data, *labels;
  int rows, cols;
  readFeature(fn, data, labels, rows, cols);

  mat rawX(data, rows, cols);

  dataset.y = mat(labels, rows, 1);
  dataset.X = mat(rows, cols + 1);
  CCE(cudaMemcpy(dataset.X.getData(), rawX.getData(), sizeof(float) * rawX.size(), cudaMemcpyDeviceToDevice));

  dataset.y = getStandardLabels(dataset.y);
  dataset.prob = label2PosteriorProb(dataset.y);

  fillLastColumnWith(dataset.X, (float) 1.0);

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

  // shuffleFeature(data, labels, rows, cols);

  fin.close();
}


bool isLabeled(const mat& labels) {

  return getLabelMapping(labels).size() > 1;

  /*size_t L = labels.size();

  thrust::device_vector<float> zero_vec(L, 0);
  thrust::device_ptr<float> label_ptr(labels.getData());

  bool isAllZero = thrust::equal(label_ptr, label_ptr + L, zero_vec.begin());
  return !isAllZero;*/
}

