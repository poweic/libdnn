#ifndef __RBM_H_
#define __RBM_H_

#include <pbar.h>
#include <dnn-utility.h>

std::vector<mat> rbminit(DataSet& data, const std::vector<size_t> &dims);

__global__ void turnOnWithProbabilityKernel(float* const data, const float* const prob, unsigned int rows, unsigned int cols);

void turnOnWithProbability(mat &y);

mat RBMinit(mat& data, size_t nHiddenUnits);

#endif
