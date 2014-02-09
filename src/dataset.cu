#include <dataset.h>
#include <dnn-utility.h>

DataSet::DataSet() {
}

DataSet::DataSet(const string &fn, bool rescale) {

  read(fn, rescale);

  convertToStandardLabels();
  label2PosteriorProb();
}

void DataSet::convertToStandardLabels() {
  assert(_hy.getCols() == 1);

  // Replace labels to 1, 2, 3, N, using mapping
  map<int, int> classes = getLabelMapping(_hy);
  for (size_t i=0; i<_hy.getRows(); ++i)
    _hy.getData()[i] = classes[_hy.getData()[i]];
}

size_t DataSet::getInputDimension() const {
  // FIXME the input dimension shouldn't be so unclear
  return _hx.getCols() - 1;
}

size_t DataSet::getOutputDimension() const {
  return _hprob.getCols();
}

void DataSet::rescaleFeature(float lower, float upper) {

  size_t rows = _hx.getRows(),
	 cols = _hx.getCols();

  for (size_t i=0; i<rows; ++i) {
    float min = _hx.getData()[i],
	  max = _hx.getData()[i];

    for (size_t j=0; j<cols; ++j) {
      float x = _hx(i, j);
      if (x > max) max = x;
      if (x < min) min = x;
    }

    if (max == min) {
      for (size_t j=0; j<cols; ++j)
	_hx(i, j) = upper;
      continue;
    }

    float ratio = (upper - lower) / (max - min);
    for (size_t j=0; j<cols; ++j)
      _hx(i, j) = (_hx(i, j) - min) * ratio + lower;
  }
}

void DataSet::read(const string &fn, bool rescale) {
  ifstream fin(fn.c_str());

  bool isSparse = isFileSparse(fn);

  size_t cols = isSparse ? findMaxDimension(fin) : findDimension(fin);
  size_t rows = getLineNumber(fin);

  _hx.resize(rows, cols);
  _hy.resize(rows, 1);

  // memset(data, 0, sizeof(float) * rows * cols);

  if (isSparse)
    readSparseFeature(fin);
  else
    readDenseFeature(fin);

  fin.close();

  // --------------------------------------
  if (rescale) {
    printf("\33[33m[Info]\33[0m rescale each feature to [0, 1]\n");
    rescaleFeature();
  }

  /*_X = mat(_hx.getData(), rows, cols);
  _X.reserve(rows * (cols + 1));
  _X.resize(rows, cols + 1);
  fillLastColumnWith(_X, (float) 1.0);
  _y = mat(_hy.getData(), rows, 1);*/

  _hx.reserve(rows * (cols + 1));
  _hx.resize(rows, cols + 1);

  float* lastColumn = _hx.getData() + rows * cols;
  std::fill(lastColumn, lastColumn + rows, 1.0f);

  // --------------------------------------
}

void DataSet::readSparseFeature(ifstream& fin) {

  size_t rows = _hx.getRows(),
	 cols = _hx.getCols();

  string line, token;
  size_t i = 0;
  while (std::getline(fin, line)) {
    stringstream ss(line);
  
    ss >> token;
    _hy.getData()[i] = str2float(token);

    while (ss >> token) {
      size_t pos = token.find(':');
      if (pos == string::npos)
	continue;

      size_t j = str2float(token.substr(0, pos)) - 1;
      float value = str2float(token.substr(pos + 1));
      
      _hx.getData()[j * rows + i] = value;
    }
    ++i;
  }
}

void DataSet::readDenseFeature(ifstream& fin) {
  
  size_t rows = _hx.getRows(),
	 cols = _hx.getCols();

  string line, token;
  size_t i = 0;
  while (std::getline(fin, line)) {
    stringstream ss(line);
  
    ss >> token;
    _hy.getData()[i] = str2float(token);

    size_t j = 0;
    while (ss >> token)
      _hx.getData()[(j++) * rows + i] = str2float(token);
    ++i;
  }
}

void DataSet::showSummary() const {
  size_t input_dim  = _hx.getCols();
  size_t nData	    = _hx.getRows();
  size_t nClasses   = _hprob.getCols();

  printf("+--------------------------------+-----------+\n");
  printf("| Number of classes              | %9lu |\n", nClasses);
  printf("| Number of input feature (data) | %9lu |\n", nData);
  printf("| Dimension of  input feature    | %9lu |\n", input_dim);
  printf("+--------------------------------+-----------+\n");

}

size_t DataSet::getClassNumber() const {
  return getLabelMapping(_hy).size();
}

void DataSet::label2PosteriorProb() {
  
  map<int, int> classes = getLabelMapping(_hy);
  size_t nClasses = classes.size();

  // Convert labels to posterior probabilities
  _hprob.resize(_hy.getRows(), nClasses);
  _hprob.fillwith(0);

  for (size_t i=0; i<_hprob.getRows(); ++i)
    _hprob(i, (_hy[i] - 1)) = 1;
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

void DataSet::shuffleFeature() {

  size_t rows = _hx.getRows(),
	 cols = _hx.getCols();

  std::vector<size_t> perm = randperm(rows);

  hmat x(_hx), y(_hy);

  for (size_t i=0; i<rows; ++i) {
    for (size_t j=0; j<cols; ++j)
      _hx(perm[i], j) = x (i, j);
    _hy[perm[i]] = y[i];
  }

  label2PosteriorProb();
}

bool DataSet::isLabeled() const {
  return getLabelMapping(_hy).size() > 1;
}

mat DataSet::getX() const {
  return mat(_hx.getData(), _hx.getRows(), _hx.getCols());
}

mat DataSet::getY() const {
  return mat(_hy.getData(), _hy.getRows(), _hy.getCols());
}

mat DataSet::getProb() const {
  return mat(_hprob.getData(), _hprob.getRows(), _hprob.getCols());
}

void DataSet::splitIntoTrainingAndValidationSet(
    DataSet& train, DataSet& valid,
    DataSet& data, int ratio) {

  size_t rows = data.getX().getRows(),
	 inputDim = data.getX().getCols(),
	 outputDim = data.getProb().getCols();
  
  /*float *h_X = new float[rows*inputDim],
	*h_y = new float[rows],
        *h_prob = new float[rows*outputDim];

  CCE(cudaMemcpy(h_X, data.getX().getData(), sizeof(float) * data.getX().size(), cudaMemcpyDeviceToHost));
  CCE(cudaMemcpy(h_y, data.getY().getData(), sizeof(float) * data.getY().size(), cudaMemcpyDeviceToHost));
  CCE(cudaMemcpy(h_prob, data.getProb().getData(), sizeof(float) * data.getProb().size(), cudaMemcpyDeviceToHost));*/

  //float* h_trainX, *h_trainY, *h_trainProb, *h_validX, *h_validY, *h_validProb;

  size_t nValid = rows / ratio,
	 nTrain = rows - nValid;

  printf("| nTrain                         | %9lu |\n", nTrain);
  printf("| nValid                         | %9lu |\n", nValid);

  train._hx.resize(nTrain, inputDim);
  train._hy.resize(nTrain, 1);
  train._hprob.resize(nTrain, outputDim);

  valid._hx.resize(nValid, inputDim);
  valid._hy.resize(nValid, 1);
  valid._hprob.resize(nValid, outputDim);

  this->splitIntoTrainingAndValidationSet(
      train._hx.getData(), train._hprob.getData(), train._hy.getData(), nTrain,
      valid._hx.getData(), valid._hprob.getData(), valid._hy.getData(), nValid,
      ratio,
      _hx.getData(), _hprob.getData(), _hy.getData(),
      rows, inputDim, outputDim);

  /*train.getX()    = mat(h_trainX   , nTrain, inputDim );
  train.getProb() = mat(h_trainProb, nTrain, outputDim);
  train.getY()    = mat(h_trainY   , nTrain, 1        );

  valid.getX()    = mat(h_validX   , nValid, inputDim );
  valid.getProb() = mat(h_validProb, nValid, outputDim);
  valid.getY()    = mat(h_validY   , nValid, 1	      );*/

  /*delete [] h_X;
  delete [] h_prob;
  delete [] h_y;*/

  /*delete [] h_trainX;
  delete [] h_trainY;
  delete [] h_trainProb;

  delete [] h_validX;
  delete [] h_validY;
  delete [] h_validProb;*/
}

void DataSet::splitIntoTrainingAndValidationSet(
    float* trainX, float* trainProb, float* trainY, size_t nTrain,
    float* validX, float* validProb, float* validY, size_t nValid,
    int ratio, /* ratio of training / validation */
    const float* const data, const float* const prob, const float* const labels,
    int rows, int inputDim, int outputDim) {

  /*nValid = rows / ratio;
  nTrain = rows - nValid;
  printf("| nTrain                         | %9lu |\n", nTrain);
  printf("| nValid                         | %9lu |\n", nValid);*/

  /*trainX    = new float[nTrain * inputDim];
  trainProb = new float[nTrain * outputDim];
  trainY    = new float[nTrain];

  validX    = new float[nValid * inputDim];
  validProb = new float[nValid * outputDim];
  validY    = new float[nValid];*/

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
