#include <dataset.h>
#include <dnn-utility.h>

DataSet::DataSet() {
}

DataSet::DataSet(const string &fn, bool rescale) {

  read(fn, rescale);

  _y = getStandardLabels();
  _prob = label2PosteriorProb(_y);
}

mat DataSet::getStandardLabels() {
  assert(_y.getCols() == 1);

  size_t N = _y.getRows();
  float* hy = new float[N];
  CCE(cudaMemcpy(hy, _y.getData(), sizeof(float) * N, cudaMemcpyDeviceToHost));

  // Replace labels to 1, 2, 3, N, using mapping
  map<int, int> classes = getLabelMapping(_y);
  for (size_t i=0; i<N; ++i)
    hy[i] = classes[hy[i]];

  mat sLabels(hy, N, 1);
  delete [] hy;

  return sLabels;
}

size_t DataSet::getInputDimension() const {
  // FIXME the input dimension shouldn't be so unclear
  return _X.getCols() - 1;
}

size_t DataSet::getOutputDimension() const {
  return _prob.getCols();
}

void DataSet::rescaleFeature(float* data, size_t rows, size_t cols, float lower, float upper) {
  for (size_t i=0; i<rows; ++i) {
    float min = data[i],
	  max = data[i];

    for (size_t j=0; j<cols; ++j) {
      float x = data[j*rows + i];
      if (x > max) max = x;
      if (x < min) min = x;
    }

    if (max == min) {
      for (size_t j=0; j<cols; ++j)
	data[j*rows + i] = upper;
      continue;
    }

    float ratio = (upper - lower) / (max - min);
    for (size_t j=0; j<cols; ++j)
      data[j*rows + i] = (data[j*rows + i] - min) * ratio + lower;
  }
}

void DataSet::read(const string &fn, bool rescale) {
  ifstream fin(fn.c_str());

  bool isSparse = isFileSparse(fn);

  size_t cols = isSparse ? findMaxDimension(fin) : findDimension(fin);
  size_t rows = getLineNumber(fin);
  float* data = new float[rows * cols];
  float* labels = new float[rows];

  memset(data, 0, sizeof(float) * rows * cols);

  if (isSparse)
    readSparseFeature(fin, data, labels, rows, cols);
  else
    readDenseFeature(fin, data, labels, rows, cols);

  fin.close();

  // --------------------------------------
  if (rescale) {
    printf("\33[33m[Info]\33[0m rescale each feature to [0, 1]\n");
    rescaleFeature(data, rows, cols);
  }

  _X = mat(data, rows, cols);
  _X.reserve(rows * (cols + 1));
  _X.resize(rows, cols + 1);
  fillLastColumnWith(_X, (float) 1.0);

  _y = mat(labels, rows, 1);
  // --------------------------------------

  delete [] data;
  delete [] labels;
}

void DataSet::readSparseFeature(ifstream& fin, float* data, float* labels, size_t rows, size_t cols) {

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

void DataSet::readDenseFeature(ifstream& fin, float* data, float* labels, size_t rows, size_t cols) {

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

void DataSet::showSummary() const {
  size_t input_dim  = _X.getCols();
  size_t nData	    = _X.getRows();
  size_t nClasses   = _prob.getCols();

  printf("+--------------------------------+-----------+\n");
  printf("| Number of classes              | %9lu |\n", nClasses);
  printf("| Number of input feature (data) | %9lu |\n", nData);
  printf("| Dimension of  input feature    | %9lu |\n", input_dim);
  printf("+--------------------------------+-----------+\n");

}

size_t DataSet::getClassNumber() const {
  thrust::device_ptr<float> dptr(_y.getData());
  thrust::host_vector<float> y(dptr, dptr + _y.size());

  map<float, bool> classes;
  for (size_t i=0; i<_y.size(); ++i)
    classes[y[i]] = true;

  return classes.size();
}

void DataSet::shuffleFeature() {

  float *h_X = new float[_X.size()],
	*h_y = new float[_y.size()];

  CCE(cudaMemcpy(h_X, _X.getData(), sizeof(float) * _X.size(), cudaMemcpyDeviceToHost));
  CCE(cudaMemcpy(h_y, _y.getData(), sizeof(float) * _y.size(), cudaMemcpyDeviceToHost));

  shuffleFeature(h_X, h_y, _X.getRows(), _X.getCols());

  CCE(cudaMemcpy(_X.getData(), h_X, sizeof(float) * _X.size(), cudaMemcpyHostToDevice));
  CCE(cudaMemcpy(_y.getData(), h_y, sizeof(float) * _y.size(), cudaMemcpyHostToDevice));

  _prob = label2PosteriorProb(_y);

  delete [] h_X;
  delete [] h_y;
}

bool isFileSparse(string train_fn) {
  ifstream fin(train_fn.c_str());
  string line;
  std::getline(fin, line);
  return line.find(':') != string::npos;
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

void DataSet::shuffleFeature(float* const data, float* const labels, int rows, int cols) {

  std::vector<size_t> perm = randperm(rows);

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
}

mat& DataSet::getX() {
  return _X;
}

const mat& DataSet::getX() const {
  return _X;
}

mat& DataSet::getY() {
  return _y;
}

const mat& DataSet::getY() const {
  return _y;
}

mat& DataSet::getProb() {
  return _prob;
}

const mat& DataSet::getProb() const {
  return _prob;
}

void splitIntoTrainingAndValidationSet(
    DataSet& train, DataSet& valid,
    DataSet& data, int ratio) {

  size_t rows = data.getX().getRows(),
	 inputDim = data.getX().getCols(),
	 outputDim = data.getProb().getCols();
  
  float *h_X = new float[rows*inputDim],
	*h_y = new float[rows],
        *h_prob = new float[rows*outputDim];

  CCE(cudaMemcpy(h_X, data.getX().getData(), sizeof(float) * data.getX().size(), cudaMemcpyDeviceToHost));
  CCE(cudaMemcpy(h_y, data.getY().getData(), sizeof(float) * data.getY().size(), cudaMemcpyDeviceToHost));
  CCE(cudaMemcpy(h_prob, data.getProb().getData(), sizeof(float) * data.getProb().size(), cudaMemcpyDeviceToHost));

  float* h_trainX, *h_trainY, *h_trainProb, *h_validX, *h_validY, *h_validProb;
  size_t nTrain, nValid;
  splitIntoTrainingAndValidationSet(
      h_trainX, h_trainProb, h_trainY, nTrain,
      h_validX, h_validProb, h_validY, nValid,
      ratio,
      h_X, h_prob, h_y,
      rows, inputDim, outputDim);

  train.getX()    = mat(h_trainX   , nTrain, inputDim );
  train.getProb() = mat(h_trainProb, nTrain, outputDim);
  train.getY()    = mat(h_trainY   , nTrain, 1        );

  valid.getX()    = mat(h_validX   , nValid, inputDim );
  valid.getProb() = mat(h_validProb, nValid, outputDim);
  valid.getY()    = mat(h_validY   , nValid, 1	      );

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
  printf("| nTrain                         | %9lu |\n", nTrain);
  printf("| nValid                         | %9lu |\n", nValid);

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
