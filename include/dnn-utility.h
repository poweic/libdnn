#ifndef _DNN_UTILITY_H_
#define _DNN_UTILITY_H_

#include <dnn.h>
#include <perf.h>

size_t getClassNumber(const DataSet& data);

map<int, int> getLabelMapping(const mat& labels);

float max(const mat& v);
mat label2PosteriorProb(const mat& labels);
mat posteriorProb2Label(const mat& prob);

size_t zeroOneError(const mat& predict, const mat& label, ERROR_MEASURE errorMeasure);
mat& calcError(const mat& output, const mat& trainY, size_t offset = 0, size_t nData = 0);

vector<float> copyToHost(const mat& m);
size_t countDifference(const mat& m1, const mat& m2);

void print(const std::vector<mat>& vm);

void showSummary(const DataSet& data);
void showAccuracy(size_t nError, size_t nTotal);

void getDataAndLabels(string train_fn, mat& data, mat& labels);

bool isFileSparse(string train_fn);

string getTempFilename();
void exec(string command);
float str2float(const string &s);
vector<string> split(const string &s, char delim);
vector<string>& split(const string &s, char delim, vector<string>& elems);
vector<size_t> splitAsInt(const string &s, char delim);

std::vector<size_t> randshuf(size_t N);
void shuffleFeature(float* const data, float* const labels, int rows, int cols);

void splitIntoTrainingAndValidationSet(
    DataSet& train, DataSet& valid,
    DataSet& data, int ratio);

void splitIntoTrainingAndValidationSet(
    float* &trainX, float* &trainProb, float* &trainY, size_t& nTrain,
    float* &validX, float* &validProb, float* &validY, size_t& nValid,
    int ratio, /* ratio of training / validation */
    const float* const data, const float* const prob, const float* const labels,
    int rows, int inputDim, int outputDim);

void getFeature(const string &fn, DataSet& dataset);
void readFeature(const string &fn, float* &X, float* &y, int &rows, int &cols);
void readSparseFeature(ifstream& fin, float* data, float* labels, size_t rows, size_t cols);
void readDenseFeature(ifstream& fin, float* data, float* labels, size_t rows, size_t cols);

size_t getLineNumber(ifstream& fin);
size_t findMaxDimension(ifstream& fin);
size_t findDimension(ifstream& fin);
bool isLabeled(const mat& labels);

#endif // _DNN_UTILITY_H_
