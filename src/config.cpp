#include <config.h>

Config::Config():
    learningRate(0.01),
    maxEpoch(1024),
    variance(0.01),
    batchSize(32),
    trainValidRatio(5),
    nNonIncEpoch(6),
    minValidAccuracy(0),
    randperm(false) {
  }
    

void Config::print() const {
  printf("| learning rate                  |%9g |\n", learningRate);
  printf("| maxEpoch                       |%9lu |\n", maxEpoch);
  printf("| std for random init            |%9g |\n", variance);
  printf("| batchSize                      |%9lu |\n", batchSize);
  printf("| training / validation          |%9lu |\n", trainValidRatio);
  printf("| minimun validation accuracy    |%9g |\n", minValidAccuracy);
  printf("| random permuation each epoch   |%9s |\n", randperm?"true":"false");
  printf("+--------------------------------+----------+\n");
  printf("When the accuracy on validation set doesn't go up for %lu epochs, the training procedure would stop\n", nNonIncEpoch);
}

std::vector<size_t> getDimensions(const std::string& structure, size_t input_dim, size_t output_dim) {

  // Initialize hidden structure
  std::vector<size_t> dims = splitAsInt(structure, '-');
  dims.insert(dims.begin(), input_dim);
  dims.push_back(output_dim);

  printf("| Number of Hidden Layers        |%9lu |\n", dims.size() - 2);

  return dims;
}
