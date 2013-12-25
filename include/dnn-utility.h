#ifndef _DNN_UTILITY_H_
#define _DNN_UTILITY_H_

#include <dnn.h>

void evaluate(DNN& dnn, mat& X, mat& y);

void zeroOneLabels(const mat& label);
size_t zeroOneError(const mat& predict, const mat& label);

void print(const std::vector<mat>& vm);

void showSummary(const mat& data, const mat& label);
void showAccuracy(size_t nError, size_t nTotal);

void getDataAndLabels(string train_fn, mat& data, mat& labels);

string getTempFilename();
void exec(string command);

#endif // _DNN_UTILITY_H_
