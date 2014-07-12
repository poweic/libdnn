#ifndef __RBM_H_
#define __RBM_H_

#include <pbar.h>
#include <dnn-utility.h>
#include <dataset.h>
#include <host_matrix.h>
#include <feature-transform.h>
using namespace std;

ostream& operator << (ostream& os, const UNIT_TYPE& type);

class StackedRbm {
public:
  StackedRbm(UNIT_TYPE vis_type, const vector<size_t>& dims, 
      float slopeThres, float learning_rate = 0.1);

  void save(const string& fn);

  float getReconstructionError(DataSet& data, const mat& W,
      UNIT_TYPE vis_type, UNIT_TYPE hid_type, int layer);

  float getFreeEnergy(const mat& visible, const mat& W);
  float getFreeEnergyGap(DataSet& data, size_t batch_size, const mat& W, int layer);
  void antiWeightExplosion(mat& W, const mat& v1, const mat& v2, float &learning_rate);

  void   up_propagate(const mat& W, const mat& visible, mat& hidden, UNIT_TYPE type);
  void down_propagate(const mat& W, mat& visible, const mat& hidden, UNIT_TYPE type);

  void train(DataSet& data);
  void rbm_train(DataSet& data, int layer, UNIT_TYPE vis_type, UNIT_TYPE hid_type);

  mat getBatchData(DataSet& data, const Batches::iterator& itr, int layer);

  static const float initial_momentum, final_momentum, L2_penalty;

  static size_t AskUserForOutputDimension();

  static vector<size_t> parseDimensions(
      size_t input_dim, 
      const string& hidden_structure, 
      size_t output_dim);

private:
  UNIT_TYPE _vis_type;
  vector<size_t> _dims;
  vector<mat> _weights;
  float _slopeThres;
  float _learning_rate;
};

float calcAverageStandardDeviation(const mat& x);

float getSlope(const vector<float> &error, size_t N);

float getAsymptoticBound(const vector<float> &error, size_t epoch, size_t maxEpoch, size_t N);

#endif
