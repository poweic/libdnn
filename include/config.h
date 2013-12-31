#ifndef __CONFIG_H_
#define __CONFIG_H_

#include <iostream>

struct Config {
  Config();
    
  float learningRate;
  size_t maxEpoch;
  float variance;
  size_t batchSize;
  size_t trainValidRatio;
  size_t nNonIncEpoch;

  void print() const;
};













































#endif // __CONFIG_H_
