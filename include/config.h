#ifndef __CONFIG_H_
#define __CONFIG_H_

#include <utility.h>
#include <vector>
#include <dataset.h>

struct Config {
  Config();
    
  float learningRate;
  size_t maxEpoch;
  float variance;
  size_t batchSize;
  size_t trainValidRatio;
  size_t nNonIncEpoch;
  float minValidAccuracy;
  bool randperm;
  std::vector<size_t> dims;

  void print() const;
  void setDimensions(const std::string& structure, const DataSet& data);
};

#endif // __CONFIG_H_
