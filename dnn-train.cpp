#include <iostream>
#include <string>
#include <dnn.h>
#include <dnn-utility.h>
#include <cmdparser.h>
#include <rbm.h>
using namespace std;

void dnn_train(DNN& dnn, const DataSet& train, const DataSet& valid, size_t batchSize, ERROR_MEASURE errorMeasure);
bool isEoutStopDecrease(const std::vector<size_t> Eout, size_t epoch, size_t nNonIncEpoch);

int main (int argc, char* argv[]) {

  CmdParser cmd(argc, argv);

  cmd.add("training_set_file")
     .add("model_file", false);

  cmd.addGroup("Training options: ")
     .add("--rp", "perform random permutation at the start of each epoch", "false")
     .add("-v", "ratio of training set to validation set (split automatically)", "5")
     .add("--max-epoch", "number of maximum epochs", "100000")
     .add("--min-acc", "Specify the minimum cross-validation accuracy", "0.5")
     .add("--learning-rate", "learning rate in back-propagation", "0.01")
     .add("--variance", "the variance of normal distribution when initializing the weights", "0.01")
     .add("--batch-size", "number of data per mini-batch", "32")
     .add("--type", "choose one of the following:\n"
	"0 -- classfication\n"
	"1 -- regression", "0");

  cmd.addGroup("Structure of Neural Network: ")
     .add("--nodes", "specify the width(nodes) of each hidden layer seperated by \"-\":\n"
	"Ex: 1024-1024-1024 for 3 hidden layer, each with 1024 nodes. \n"
	"(Note: This does not include input and output layer)", "-1");

  cmd.addGroup("Pre-training options:")
     .add("--rescale", "Rescale each feature to [0, 1]", "false")
     .add("--slope-thres", "threshold of ratio of slope in RBM pre-training", "0.05")
     .add("--pre", "type of Pretraining. Choose one of the following:\n"
	"0 -- Random initialization (no pre-training)\n"
	"1 -- RBM (Restricted Boltzman Machine)\n"
	"2 -- Load from pre-trained model", "0")
     .add("-f", "when option --pre is set to 2, specify the filename of the pre-trained model", "train.dat.model");

  cmd.addGroup("Example usage: dnn-train data/train3.dat --nodes=16-8");

  if (!cmd.isOptionLegal())
    cmd.showUsageAndExit();

  string train_fn     = cmd[1];
  string model_fn     = cmd[2];
  string structure    = cmd["--nodes"];
  int ratio	      = cmd["-v"];
  size_t batchSize    = cmd["--batch-size"];
  float learningRate  = cmd["--learning-rate"];
  float variance      = cmd["--variance"];
  float minValidAcc   = cmd["--min-acc"];
  size_t maxEpoch     = cmd["--max-epoch"];
  size_t preTraining  = cmd["--pre"];
  bool rescale        = cmd["--rescale"];
  bool randperm	      = cmd["--rp"];
  float slopeThres    = cmd["--slope-thres"];
  string pre_model_fn = cmd["-f"];

  if (model_fn.empty())
    model_fn = train_fn.substr(train_fn.find_last_of('/') + 1) + ".model";

  DataSet data(train_fn, rescale);
  data.shuffleFeature();
  data.showSummary();

  DataSet train, valid;
  data.splitIntoTrainingAndValidationSet(train, valid, data, ratio);

  // Set configurations
  Config config;
  config.variance = variance;
  config.learningRate = learningRate;
  config.minValidAccuracy = minValidAcc;
  config.maxEpoch = maxEpoch;
  config.setDimensions(structure, data);
  config.print();

  DNN dnn;
  // Initialize Deep Neural Network
  switch (preTraining) {
    case 0:
      dnn.init(config.dims);
      break;

    case 1:
      dnn.init(rbminit(data, getDimensionsForRBM(data, structure), slopeThres));
      break;

    case 2:
      assert(!pre_model_fn.empty());
      printf("Loading pre-trained model from file: \"%s\"\n", pre_model_fn.c_str());
      dnn = DNN(pre_model_fn);
      break;

    default:
      return -1;
  }

  dnn.setConfig(config);

  ERROR_MEASURE err = CROSS_ENTROPY;
  // Start Training
  dnn_train(dnn, train, valid, batchSize, err);

  // Save the model
  dnn.save(model_fn);

  return 0;
}

void dnn_train(DNN& dnn, const DataSet& train, const DataSet& valid, size_t batchSize, ERROR_MEASURE errorMeasure) {

  printf("Training...\n");
  perf::Timer timer;
  timer.start();

  vector<mat> O(dnn.getNLayer());

  size_t Ein;
  size_t MAX_EPOCH = dnn.getConfig().maxEpoch, epoch;
  std::vector<size_t> Eout;
  Eout.reserve(MAX_EPOCH);

  size_t nTrain = train.getX().getRows(),
	 nValid = valid.getX().getRows();

  size_t nBatch = nTrain / batchSize,
         remained = nTrain - nBatch * batchSize;

  mat fout;

  if (remained > 0)
    ++nBatch;

  for (epoch=0; epoch<MAX_EPOCH; ++epoch) {

    if (dnn.getConfig().randperm)
      const_cast<DataSet&>(train).shuffleFeature();

    for (size_t b=0; b<nBatch; ++b) {

      size_t offset = b*batchSize;
      size_t nData = batchSize;

      if (b == nBatch - 1)
	nData = min(remained - 1, batchSize);

      // Copy a batch of data from training data to O[0]
      mat fin(nData, train.getX().getCols());
      memcpy2D(fin, train.getX(), offset, 0, nData, train.getX().getCols(), 0, 0);

      dnn.feedForward(fout, fin);

      mat error = dnn.getError(train.getProb(), fout, offset, nData, errorMeasure);

      dnn.backPropagate(error, fin, fout);
      dnn.update(dnn.getConfig().learningRate);
    }

    dnn.feedForward(fout, valid.getX());
    Eout.push_back(zeroOneError(fout, valid.getY(), errorMeasure));
  
    dnn.feedForward(fout, train.getX());
    Ein = zeroOneError(fout, train.getY(), errorMeasure);

    float trainAcc = 1.0f - (float) Ein / nTrain;

    if (trainAcc < 0.5) {
      cout << "."; cout.flush();
      continue;
    }

    float validAcc= 1.0f - (float) Eout[epoch] / nValid;
    if (validAcc > dnn.getConfig().minValidAccuracy && isEoutStopDecrease(Eout, epoch, dnn.getConfig().nNonIncEpoch))
      break;

    printf("Epoch #%lu: Training Accuracy = %.4f %% ( %lu / %lu ), Validation Accuracy = %.4f %% ( %lu / %lu )\n",
      epoch, trainAcc * 100, nTrain - Ein, nTrain, validAcc * 100, nValid - Eout[epoch], nValid); 

    dnn.adjustLearningRate(trainAcc);
  }

  // Show Summary
  printf("\n%ld epochs in total\n", epoch);
  timer.elapsed();

  dnn.feedForward(fout, train.getX());
  Ein = zeroOneError(fout, train.getY(), errorMeasure);

  printf("[   In-Sample   ] ");
  showAccuracy(Ein, train.getY().size());
  printf("[ Out-of-Sample ] ");
  showAccuracy(Eout.back(), valid.getY().size());
}

bool isEoutStopDecrease(const std::vector<size_t> Eout, size_t epoch, size_t nNonIncEpoch) {

  for (size_t i=0; i<nNonIncEpoch; ++i) {
    if (epoch - i > 0 && Eout[epoch] > Eout[epoch - i])
      return false;
  }

  return true;
}

