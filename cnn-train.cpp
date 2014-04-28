#include <cuda_profiler_api.h>

#include <cmdparser.h>
#include <pbar.h>

#include <dataset.h>
#include <dnn.h>
#include <cnn.h>

vector<mat> getRandWeights(size_t input_dim, string structure, size_t output_dim);

size_t cnn_predict(const DNN& dnn, CNN& cnn, DataSet& data,
    ERROR_MEASURE errorMeasure);

void cnn_train(DNN& dnn, CNN& cnn, DataSet& train, DataSet& valid,
    size_t batchSize, ERROR_MEASURE errorMeasure);

void cuda_profiling_ground();

int main(int argc, char* argv[]) {

  CmdParser cmd(argc, argv);

  cmd.add("training_set_file")
    .add("model_in", false)
    .add("model_out", false);

  cmd.addGroup("Feature options:")
     .add("--input-dim", "specify the input dimension (dimension of feature).\n"
	 "For example: --input-dim 39x9 \n")
     .add("--normalize", "Feature normalization: \n"
	"0 -- Do not normalize.\n"
	"1 -- Rescale each dimension to [0, 1] respectively.\n"
	"2 -- Normalize to standard score. z = (x-u)/sigma ."
	"filename -- Read mean and variance from file", "0")
     .add("--base", "Label id starts from 0 or 1 ?", "0")
     .add("--output-dim", "specify the output dimension (the # of class to predict).\n");

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
     .add("--batch-size", "number of data per mini-batch", "32");

  cmd.addGroup("Example usage: cnn-train data/train3.dat --struct=12x5x5-2-8x3x3-2");
  
  if (!cmd.isOptionLegal())
    cmd.showUsageAndExit();

  string train_fn   = cmd[1];
  string model_in   = cmd[2];
  string model_out  = cmd[2];

  string input_dim  = cmd["--input-dim"];
  NormType n_type   = (NormType) (int) cmd["--normalize"];
  int base	    = cmd["--base"];
  string structure  = cmd["--struct"];
  size_t output_dim = cmd["--output-dim"];

  int ratio	      = cmd["-v"];
  size_t batchSize    = cmd["--batch-size"];

  // Parse input dimension
  SIZE imgSize = parseInputDimension(input_dim);
  printf("Image dimension = %ld x %lu\n", imgSize.m, imgSize.n);

  // Load dataset
  DataSet data(train_fn, imgSize.m * imgSize.n, base);
  data.setNormType(n_type);
  data.showSummary();

  DataSet train, valid;
  DataSet::split(data, train, valid, ratio);

  // Parse structure
  string cnn_struct, nn_struct;
  parseNetworkStructure(structure, cnn_struct, nn_struct);

  // Initialize CNN
  CNN cnn;
  if (model_in.empty())
    cnn.init(cnn_struct, imgSize);
  else
    cnn.read(model_in);

  DNN dnn;
  dnn.init(getRandWeights(cnn.getOutputDimension(), nn_struct, output_dim));

  // Show CNN status
  cnn.status();

  cnn_train(dnn, cnn, train, valid, batchSize, CROSS_ENTROPY);

  if (model_out.empty())
    model_out = train_fn.substr(train_fn.find_last_of('/') + 1) + ".model";

  cout << "Leaving..." << endl;

  return 0;
}

void cnn_train(DNN& dnn, CNN& cnn, DataSet& train, DataSet& valid,
    size_t batchSize, ERROR_MEASURE errorMeasure) {

  perf::Timer timer;
  timer.start();

  const size_t MAX_EPOCH = 1024;
  size_t nTrain = train.size(),
	 nValid = valid.size();

  mat fmiddle, fout;
  float t_start = timer.getTime();

  for (size_t epoch=0; epoch<MAX_EPOCH; ++epoch) {

    Batches batches(batchSize, nTrain);
    for (auto itr = batches.begin(); itr != batches.end(); ++itr) {
      mat fin = train.getX(*itr);

      cnn.feedForward(fmiddle, fin);
      dnn.feedForward(fout, fmiddle);

      // matlog(fmiddle);
      // matlog(fout);
      mat error = getError( train.getY(*itr), fout, errorMeasure);
      // matlog(error);

      dnn.backPropagate(error, fmiddle, fout, 1.0f / itr->nData );
      // matlog(error);
      cnn.backPropagate(error, fin, fmiddle, 1);
      // matlog(error);
      // exit(-1);
    }

    size_t Ein  = cnn_predict(dnn, cnn, train, errorMeasure),
	   Eout = cnn_predict(dnn, cnn, valid, errorMeasure);

    float trainAcc = 1.0f - (float) Ein / nTrain;
    float validAcc = 1.0f - (float) Eout / nValid;
    printf("Epoch #%lu: Training Accuracy = %.4f %% ( %lu / %lu ), Validation Accuracy = %.4f %% ( %lu / %lu ), elapsed %.3f seconds.\n",
      epoch, trainAcc * 100, nTrain - Ein, nTrain, validAcc * 100, nValid - Eout, nValid, (timer.getTime() - t_start) / 1000); 

    t_start = timer.getTime();
  }

  timer.elapsed();
  printf("# of total epoch = %lu\n", MAX_EPOCH);
}

vector<mat> getRandWeights(size_t input_dim, string structure, size_t output_dim) {

  auto dims = splitAsInt(structure, '-');
  dims.push_back(output_dim);
  dims.insert(dims.begin(), input_dim);
  for (size_t i=0; i<dims.size(); ++i)
    dims[i] += 1;

  size_t nWeights = dims.size() - 1;
  vector<mat> weights(nWeights);

  for (size_t i=0; i<nWeights; ++i) {
    float coeff = (2 * sqrt(6.0f / (dims[i] + dims[i+1]) ) );
    weights[i] = coeff * (rand(dims[i], dims[i+1]) - 0.5);
    printf("Initialize a weights[%lu] using %.4f x (rand(%3lu,%3lu) - 0.5)\n", i,
	coeff, dims[i], dims[i+1]);
  }

  CCE(cudaDeviceSynchronize());
  return weights;
}

size_t cnn_predict(const DNN& dnn, CNN& cnn, DataSet& data,
    ERROR_MEASURE errorMeasure) {

  size_t nError = 0;

  Batches batches(2048, data.size());
  for (Batches::iterator itr = batches.begin(); itr != batches.end(); ++itr) {
    mat fmiddle;
    cnn.feedForward(fmiddle, data.getX(*itr));
    mat prob = dnn.feedForward(fmiddle);
    nError += zeroOneError(prob, data.getY(*itr), errorMeasure);
  }

  return nError;
}

void cuda_profiling_ground() {
  mat x = randn(128, 128),
      h = randn(20, 20);

  perf::Timer timer;
  timer.start();
  cudaProfilerStart(); 
  
  mat z;
  for (int i=0; i<10000; ++i) {
    z = convn(x, h, "valid_shm");
  }

  CCE(cudaDeviceSynchronize());
  cudaProfilerStop();
  timer.elapsed();
}

