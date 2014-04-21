#include <cuda_profiler_api.h>

#include <cmdparser.h>
#include <pbar.h>

#include <dataset.h>
#include <cnn.h>

SIZE parseInputDimension(const string &m_by_n);

void cnn_train(CNN& cnn, const DataSet& train, const DataSet& valid,
    size_t batchSize, ERROR_MEASURE errorMeasure);

void playground() {
  mat x = randn(128, 128),
      h = randn(20, 20);

  perf::Timer timer;
  timer.start();
  cudaProfilerStart(); 
  
  mat z;
  for (int i=0; i<10000; ++i) {
    z = convn(x, h, "valid_shm", 4);
  }

  CCE(cudaDeviceSynchronize());
  cudaProfilerStop();
  timer.elapsed();
}

int main(int argc, char* argv[]) {

  // go_test();
  playground();
  // benchmark();
  return 0;

  CmdParser cmd(argc, argv);

  cmd.add("training_set_file")
    .add("model_in", false)
    .add("model_out", false);

  cmd.addGroup("Feature options:")
     .add("--input-dim", "specify the input dimension (dimension of feature).\n"
	 "For example: --input-dim 39x9 \n"
	 "0 for auto detection.", "0");

  cmd.addGroup("Network structure:")
     .add("--struct",
      "Specify the structure of Convolutional neural network\n"
      "For example: --struct=9x5x5-3s-4x3x3-2s-256-128\n"
      "\"9x5x5-3s\" means a convolutional layer consists of 9 output feature maps\n"
      "with a 5x5 kernel, which is followed by a sub-sampling layer with scale\n"
      "of 3. After \"9x5x5-3s-4x3x3-2s\", a neural network of of 2 hidden layers\n"
      "of width 256 and 128 is appended to it.\n"
      "Each layer should be seperated by a hyphen \"-\".");

  cmd.addGroup("Training options:")
     .add("-v", "ratio of training set to validation set (split automatically)", "5")
     .add("--batch-size", "number of data per mini-batch", "1");

  cmd.addGroup("Example usage: cnn-train data/train3.dat --struct=12x5x5-2-8x3x3-2");
  
  if (!cmd.isOptionLegal())
    cmd.showUsageAndExit();

  string train_fn   = cmd[1];
  string model_in   = cmd[2];
  string model_out  = cmd[2];

  string input_dim  = cmd["--input-dim"];
  string structure  = cmd["--struct"];

  int ratio	      = cmd["-v"];
  size_t batchSize    = cmd["--batch-size"];

  // Parse input dimension
  SIZE imgSize = parseInputDimension(input_dim);
  printf("Image dimension = %ld x %lu\n", imgSize.m, imgSize.n);

  // Load dataset
  DataSet data(train_fn, imgSize.m * imgSize.n);
  data.shuffle();
  data.showSummary();

  DataSet train, valid;
  data.splitIntoTrainAndValidSet(train, valid, ratio);

  // Parse structure
  string cnn_struct, nn_struct;
  parseNetworkStructure(structure, cnn_struct, nn_struct);

  // Initialize CNN
  CNN cnn(imgSize);
  if (model_in.empty())
    cnn.init(cnn_struct);
  else
    cnn.read(model_in);

  // Show CNN status
  cnn.status();

  cnn_train(cnn, train, valid, batchSize, CROSS_ENTROPY);

  if (model_out.empty())
    model_out = train_fn.substr(train_fn.find_last_of('/') + 1) + ".model";

  return 0;
}

void cnn_train(CNN& cnn, const DataSet& train, const DataSet& valid,
    size_t batchSize, ERROR_MEASURE errorMeasure) {

  perf::Timer timer;
  timer.start();

  const size_t MAX_EPOCH = 1;
  size_t nTrain = train.size();
	 /*nValid = valid.size();*/

  mat fout;

  ProgressBar pbar("Training...");

  for (size_t epoch=0; epoch<MAX_EPOCH; ++epoch) {
    Batches batches(batchSize, nTrain);
    for (auto itr = batches.begin(); itr != batches.end(); ++itr) {
      mat fin = train.getX(*itr);
      cnn.feedForward(fout, fin);
    }

    char status[100];
    sprintf(status, " epoch #%lu", epoch);
    pbar.refresh(0, MAX_EPOCH, status);
  }

  timer.elapsed();
}

