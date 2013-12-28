#include <dnn-utility.h>

void zeroOneLabels(mat& label) {
  thrust::device_ptr<float> dptr(label.getData());
  thrust::host_vector<float> y(dptr, dptr + label.size());

  map<float, bool> classes;
  for (size_t i=0; i<y.size(); ++i)
    classes[y[i]] = true;

  size_t nClasses = classes.size();
  assert(nClasses == 2);

  thrust::replace(dptr, dptr + label.size(), -1, 0);
}

void reformatLabels(mat& label) {
  thrust::device_ptr<float> dptr(label.getData());
  thrust::host_vector<float> y(dptr, dptr + label.size());

  map<float, bool> classes;
  for (size_t i=0; i<y.size(); ++i)
    classes[y[i]] = true;

  size_t nClasses = classes.size();
  assert(nClasses == 2);

  thrust::replace(dptr, dptr + label.size(), 1, 2);
  thrust::replace(dptr, dptr + label.size(), -1, 1);
}

float max(const mat& v) {
  thrust::device_ptr<float> vPtr(v.getData());
  thrust::device_ptr<float> maxPtr = thrust::max_element(vPtr, vPtr + v.size());
  thrust::host_vector<float> hMaxPtr(maxPtr, maxPtr + 1);
  return hMaxPtr[0];
}

void label2PosteriorProb(mat& y) {
  // Assume class label index start from 1, and there's no skipping.
  assert(y.getCols() == 1);

  size_t nData = y.getRows();
  size_t nClasses = (size_t) max(y);

  float* hy = new float[nData];

  CCE(cudaMemcpy(hy, y.getData(), sizeof(float) * nData, cudaMemcpyDeviceToHost));

  float* prob = new float[nData * nClasses];
  memset(prob, 0, sizeof(float) * nData * nClasses);

  for (size_t i=0; i<nData; ++i)
    prob[(size_t) (hy[i] - 1) * nData + i] = 1;

  y = mat(prob, nData, nClasses);

  delete [] hy;
  delete [] prob;
}


size_t zeroOneError(const mat& predict, const mat& label, ERROR_MEASURE errorMeasure) {
  assert(predict.getRows() == label.getRows() && predict.getCols() == label.getCols());

  size_t nError = 0;

  if (errorMeasure == L2ERROR) {

    size_t L = label.size();
    thrust::device_ptr<float> l_ptr(label.getData());
    thrust::device_ptr<float> p_ptr(predict.getData());

    thrust::device_vector<float> p_vec(L);
    thrust::transform(p_ptr, p_ptr + L, p_vec.begin(), func::to_zero_one<float>());

    nError = (size_t) thrust::inner_product(p_vec.begin(), p_vec.end(), l_ptr, 0.0, thrust::plus<float>(), thrust::not_equal_to<float>());
  }
  else {
    float* hp = new float[predict.size()];
    float* ht = new float[label.size()];
    CCE(cudaMemcpy(hp, predict.getData(), sizeof(float) * predict.size(), cudaMemcpyDeviceToHost));
    CCE(cudaMemcpy(ht, label.getData(), sizeof(float) * label.size(), cudaMemcpyDeviceToHost));

    size_t rows = predict.getRows();
    size_t cols = predict.getCols();
    
    for (size_t i=0; i<rows; ++i) {
      
      float max1 = 0, max2 = 0;
      size_t maxIdx1 = 0, maxIdx2 = 0;

      for (size_t j=0; j<cols; ++j) {
	if (hp[j * rows + i] > max1) {
	  max1 = hp[j * rows + i];
	  maxIdx1 = j;
	}

	if (ht[j * rows + i] > max2) {
	  max2 = ht[j * rows + i];
	  maxIdx2 = j;
	}
	
      }

      nError += (size_t) maxIdx1 != maxIdx2;
    }

    delete [] hp;
    delete [] ht;
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
	 inputDim = X.getCols(),
	 outputDim = y.getCols();
  
  float *h_X = new float[rows*inputDim],
        *h_y = new float[rows*outputDim];

  CCE(cudaMemcpy(h_X, X.getData(), sizeof(float) * X.size(), cudaMemcpyDeviceToHost));
  CCE(cudaMemcpy(h_y, y.getData(), sizeof(float) * y.size(), cudaMemcpyDeviceToHost));

  float* h_trainX, *h_trainY, *h_validX, *h_validY;
  size_t nTrain, nValid;
  splitIntoTrainingAndValidationSet(
      h_trainX, h_trainY, nTrain,
      h_validX, h_validY, nValid,
      ratio,
      h_X, h_y,
      rows, inputDim, outputDim);

  trainX = mat(h_trainX, nTrain, inputDim);
  trainY = mat(h_trainY, nTrain, outputDim);

  validX = mat(h_validX, nValid, inputDim);
  validY = mat(h_validY, nValid, outputDim);

  delete [] h_X;
  delete [] h_y;
}

void splitIntoTrainingAndValidationSet(
    float* &trainX, float* &trainY, size_t& nTrain,
    float* &validX, float* & validY, size_t& nValid,
    int ratio, /* ratio of training / validation */
    const float* const data, const float* const labels,
    int rows, int inputDim, int outputDim) {

  nValid = rows / ratio;
  nTrain = rows - nValid;
  printf("nTrain = %lu, nValid = %lu\n", nTrain, nValid);

  trainX = new float[nTrain * inputDim];
  trainY = new float[nTrain * outputDim];

  validX = new float[nValid * inputDim];
  validY = new float[nValid * outputDim];

  for (size_t i=0; i<nTrain; ++i) {
    for (size_t j=0; j<inputDim; ++j)
      trainX[j * nTrain + i] = data[j * rows + i];
    for (size_t j=0; j<outputDim; ++j)
      trainY[j * nTrain + i] = labels[j * rows + i];
    // trainY[i] = labels[i];
  }

  for (size_t i=0; i<nValid; ++i) {
    for (size_t j=0; j<inputDim; ++j)
      validX[j * nValid + i] = data[j * rows + i + nTrain];
    for (size_t j=0; j<outputDim; ++j)
      validY[j * nValid + i] = labels[j * rows + i + nTrain];
    // validY[i] = labels[i + nTrain];
  }
}

void getFeature(const string &fn, mat& X, mat& y) {

  float* data, *labels;
  int rows, cols;
  readFeature(fn, data, labels, rows, cols);

  mat rawX(data, rows, cols);

  y = mat(labels, rows, 1);
  X = mat(rows, cols + 1);
  CCE(cudaMemcpy(X.getData(), rawX.getData(), sizeof(float) * rawX.size(), cudaMemcpyDeviceToDevice));

  fillLastColumnWith(X, (float) 1.0);

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

