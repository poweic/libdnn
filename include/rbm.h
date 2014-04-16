#ifndef __RBM_H_
#define __RBM_H_

#include <pbar.h>
#include <dnn-utility.h>
#include <dataset.h>
#include <host_matrix.h>
#include <feature-transform.h>

void playground();

enum RBM_UNIT_TYPE {
  BERNOULLI,
  GAUSSIAN
};

ostream& operator << (ostream& os, const RBM_UNIT_TYPE& type);

hmat batchFeedForwarding(const hmat& X, const mat& w);

float calcAverageStandardDeviation(const mat& x);

std::vector<mat> initStackedRBM(DataSet& data, const std::vector<size_t>& dims,
    float slopeThres, RBM_UNIT_TYPE type, float learning_rate = 0.1);

void antiWeightExplosion(mat& W, const mat& v1, const mat& v2, float &learning_rate);

void sample(mat &prob);

float getFreeEnergy(const mat& visible, const mat& W);
float getFreeEnergyGap(const hmat& data, size_t batch_size, const mat& W);

void up_propagate(const mat& W, const mat& visible, mat& hidden, RBM_UNIT_TYPE type);
void down_propagate(const mat& W, mat& visible, const mat& hidden, RBM_UNIT_TYPE type);

mat rbmTrain(const hmat& data, size_t nHiddenUnits, float threshold,
    RBM_UNIT_TYPE vis_type, RBM_UNIT_TYPE hid_type, float learning_rate = 0.1);

size_t getOutputDimension();

std::vector<size_t> getDimensionsForRBM(
    size_t input_dim, 
    const string& hidden_structure, 
    size_t output_dim);

float getSlope(const std::vector<float> &error, size_t N);

float getAsymptoticBound(const std::vector<float> &error, size_t epoch, size_t maxEpoch, size_t N);

#endif
