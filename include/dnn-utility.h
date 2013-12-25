#ifndef _DNN_UTILITY_H_
#define _DNN_UTILITY_H_

#include <dnn.h>

void zeroOneLabels(const mat& label);
size_t zeroOneError(const mat& predict, const mat& label);

void print(const std::vector<mat>& vm);

void showSummary(const mat& data, const mat& label);
void showAccuracy(size_t nError, size_t nTotal);

void getDataAndLabels(string train_fn, mat& data, mat& labels);

bool isFileSparse(string train_fn);

string getTempFilename();
void exec(string command);
float str2float(const string &s);

void readFeature(const string &fn, mat& X, mat& y);
void readSparseFeature(ifstream& fin, float* data, float* labels, size_t rows, size_t cols);
void readDenseFeature(ifstream& fin, float* data, float* labels, size_t rows, size_t cols);

size_t getLineNumber(ifstream& fin);
size_t findMaxDimension(ifstream& fin);
size_t findDimension(ifstream& fin);
bool isLabeled(const mat& labels);

#endif // _DNN_UTILITY_H_
