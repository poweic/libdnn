#ifndef __CONFIG_H_
#define __CONFIG_H_

#include <utility.h>

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

  void print() const;
};

std::vector<size_t> getDimensions(const std::string& structure, size_t input_dim, size_t output_dim);

#endif // __CONFIG_H_
