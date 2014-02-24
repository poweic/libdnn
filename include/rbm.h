#ifndef __RBM_H_
#define __RBM_H_

#include <pbar.h>
#include <dnn-utility.h>
#include <dataset.h>
#include <host_matrix.h>
#include <feature-transform.h>

void playground();

std::vector<mat> rbminit(DataSet& data, const std::vector<size_t>& dims, float slopeThres);

__global__ void turn_on_kernel(float* const data, const float* const prob, unsigned int rows, unsigned int cols);

void turnOn(mat &prob);

mat rbmTrain(const hmat& data, size_t nHiddenUnits, float threshold);

std::vector<size_t> getDimensionsForRBM(const DataSet& data, const string& structure);

float getSlope(const std::vector<float> &error, size_t N);

float getAsymptoticBound(const std::vector<float> &error, size_t epoch, size_t maxEpoch, size_t N);

#endif
