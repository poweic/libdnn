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
  StackedRbm(const vector<size_t>& dims);

  void setParams(size_t max_epoch, float slope_thres, float learning_rate,
    float initial_momentum, float final_momentum, float l2_penalty);

  void printParams() const;

  void save(const string& fn);

  float getReconstructionError(DataSet& data, const mat& W,
      UNIT_TYPE vis_type, UNIT_TYPE hid_type, int layer);

  float getFreeEnergy(const mat& visible, const mat& W);
  float getFreeEnergyGap(DataSet& data, size_t batch_size, const mat& W, int layer);
  void antiWeightExplosion(mat& W, const mat& v1, const mat& v2, float &learning_rate);

  void   up_propagate(const mat& W, const mat& visible, mat& hidden, UNIT_TYPE type);
  void down_propagate(const mat& W, mat& visible, const mat& hidden, UNIT_TYPE type);

  void train(DataSet& data, UNIT_TYPE vis_type);
  void rbm_train(DataSet& data, int layer, UNIT_TYPE vis_type, UNIT_TYPE hid_type);

  mat getBatchData(DataSet& data, const Batches::iterator& itr, int layer);

  static vector<size_t> parseDimensions(
      size_t input_dim, 
      const string& hidden_structure, 
      size_t output_dim);

private:
  vector<size_t> _dims;
  vector<mat> _weights;

  size_t _max_epoch;

  float _slope_thres;
  float _learning_rate;

  float _initial_momentum;
  float _final_momentum;
  float _l2_penalty;
};

float calcAverageStandardDeviation(const mat& x);

float getSlope(const vector<float> &error, size_t N);

float getAsymptoticBound(const vector<float> &error, size_t epoch, size_t maxEpoch, size_t N);

#endif
