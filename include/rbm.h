#ifndef __RBM_H_
#define __RBM_H_

#include <pbar.h>
#include <dnn-utility.h>
#include <dataset.h>
#include <host_matrix.h>
#include <feature-transform.h>

void playground();

enum RBM_TYPE {
  BERNOULLI_BERNOULLI,
  GAUSSIAN_BERNOULLI
};

ostream& operator << (ostream& os, const RBM_TYPE& type);

std::vector<mat> initStackedRBM(DataSet& data, const std::vector<size_t>& dims,
    float slopeThres, RBM_TYPE type, float learning_rate = 0.1);

void sample(mat &prob);
void apply_cmvn(hmat& data);

mat rbmTrain(const hmat& data, size_t nHiddenUnits, float threshold,
    RBM_TYPE type, float learning_rate = 0.1);

size_t getOutputDimension();

std::vector<size_t> getDimensionsForRBM(
    size_t input_dim, 
    const string& hidden_structure, 
    size_t output_dim);

float getSlope(const std::vector<float> &error, size_t N);

float getAsymptoticBound(const std::vector<float> &error, size_t epoch, size_t maxEpoch, size_t N);

#endif
