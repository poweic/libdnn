#ifndef __RBM_H_
#define __RBM_H_

#include <pbar.h>
#include <dnn-utility.h>
#include <dataset.h>
#include <host_matrix.h>

void playground();

std::vector<mat> rbminit(DataSet& data, const std::vector<size_t>& dims, float slopeThres);

__global__ void turnOnWithProbabilityKernel(float* const data, const float* const prob, unsigned int rows, unsigned int cols);

void turnOnWithProbability(mat &y);

mat RBMinit(const hmat& data, size_t nHiddenUnits, float threshold);

std::vector<size_t> getDimensionsForRBM(const DataSet& data, const string& structure);

void linearRegression(const std::vector<float> &x, const std::vector<float>& y, float* const &m, float* const &c);

float getSlope(const std::vector<float> &error, size_t N);

float getAsymptoticBound(const std::vector<float> &error, size_t epoch, size_t maxEpoch, size_t N);

#endif
