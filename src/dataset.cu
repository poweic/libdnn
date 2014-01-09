#include <dataset.h>

void DataSet::getFeature(const string &fn, bool rescale) {

  float* data, *labels;
  int rows, cols;
  readFeature(fn, data, labels, rows, cols);

  if (rescale) {
    printf("\33[33m[Info]\33[0m rescale each feature to [0, 1]\n");
    rescaleFeature(data, rows, cols);
  }

  mat rawX(data, rows, cols);

  this->y = mat(labels, rows, 1);
  this->X = mat(rows, cols + 1);
  CCE(cudaMemcpy(this->X.getData(), rawX.getData(), sizeof(float) * rawX.size(), cudaMemcpyDeviceToDevice));
  fillLastColumnWith(this->X, (float) 1.0);

  this->y = getStandardLabels(this->y);
  this->prob = label2PosteriorProb(this->y);

  delete [] data;
  delete [] labels;
}

mat DataSet::getStandardLabels(const mat& labels) {
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

void DataSet::readFeature(const string &fn, float* &data, float* &labels, int &rows, int &cols) {
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

  fin.close();
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


std::vector<size_t> DataSet::getDimensions(const string& structure) const {

  // Initialize hidden structure
  size_t input_dim  = this->X.getCols() - 1;
  size_t output_dim = this->prob.getCols();

  vector<size_t> dims = splitAsInt(structure, '-');
  dims.insert(dims.begin(), input_dim);
  dims.push_back(output_dim);

  printf("| Number of Hidden Layers        |%9lu |\n", dims.size() - 2);

  return dims;
}

size_t DataSet::getClassNumber() const {
  thrust::device_ptr<float> dptr(this->y.getData());
  thrust::host_vector<float> y(dptr, dptr + this->y.size());

  map<float, bool> classes;
  for (size_t i=0; i<y.size(); ++i)
    classes[y[i]] = true;

  return classes.size();
}

void DataSet::showSummary() const {
  size_t input_dim  = this->X.getCols();
  size_t nData	    = this->X.getRows();
  size_t nClasses   = this->prob.getCols();

  printf("+--------------------------------+----------+\n");
  printf("| Number of classes              |%9lu |\n", nClasses);
  printf("| Number of input feature (data) |%9lu |\n", nData);
  printf("| Dimension of  input feature    |%9lu |\n", input_dim);
  printf("+--------------------------------+----------+\n");
}

void DataSet::shuffleFeature() {

  float *h_X = new float[this->X.size()],
	*h_y = new float[this->y.size()];

  CCE(cudaMemcpy(h_X, this->X.getData(), sizeof(float) * this->X.size(), cudaMemcpyDeviceToHost));
  CCE(cudaMemcpy(h_y, this->y.getData(), sizeof(float) * this->y.size(), cudaMemcpyDeviceToHost));

  shuffleFeature(h_X, h_y, this->X.getRows(), this->X.getCols());

  CCE(cudaMemcpy(this->X.getData(), h_X, sizeof(float) * this->X.size(), cudaMemcpyHostToDevice));
  CCE(cudaMemcpy(this->y.getData(), h_y, sizeof(float) * this->y.size(), cudaMemcpyHostToDevice));

  this->prob = label2PosteriorProb(this->y);

  delete [] h_X;
  delete [] h_y;
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

