#include <iostream>
#include <string>
#include <dnn.h>
// #include <dnn-utility.h>
#include <cmdparser.h>
#include <batch.h>
using namespace std;

void printLabels(const mat& prob, FILE* fid, int base);
FILE* openFileOrStdout(const string& filename);

int main (int argc, char* argv[]) {

  CmdParser cmd(argc, argv);

  cmd.add("testing_set_file")
    .add("model_file")
    .add("output_file", false);

  cmd.addGroup("Feature options:")
     .add("--input-dim", "specify the input dimension (dimension of feature).\n"
	 "0 for auto detection.")
     .add("--normalize", "Feature normalization: \n"
	"0 -- Do not normalize.\n"
	"1 -- Rescale each dimension to [0, 1] respectively.\n"
	"2 -- Normalize to standard score. z = (x-u)/sigma .", "0")
     .add("--base", "Label id starts from 0 or 1 ?", "0");

  cmd.addGroup("Options:")
    .add("--acc", "calculate prediction accuracy", "true")
    .add("--prob", "output posterior probabilities if true\n"
	"0 -- Do not output posterior probabilities. Output class-id.\n"
	"1 -- Output posterior probabilities. (range in [0, 1]) \n"
	"2 -- Output natural log of posterior probabilities. (range in [-inf, 0])", "0")
    .add("--silent", "Suppress all log messages", "false");

  cmd.addGroup("Example usage: dnn-predict test3.dat train3.dat.model");

  if (!cmd.isOptionLegal())
    cmd.showUsageAndExit();

  string test_fn    = cmd[1];
  string model_fn   = cmd[2];
  string output_fn  = cmd[3];

  size_t input_dim  = cmd["--input-dim"];
  NormType n_type   = (NormType) (int) cmd["--normalize"];
  string n_filename = cmd["--nf"];
  int base	    = cmd["--base"];

  int output_type   = cmd["--prob"];
  bool silent	    = cmd["--silent"];
  bool calcAcc	    = cmd["--acc"];

  DataSet test(test_fn, input_dim, base);
  test.loadPrecomputedStatistics(n_filename);
  test.setNormType(n_type);

  ERROR_MEASURE errorMeasure = CROSS_ENTROPY;

  DNN dnn(model_fn);

  size_t nError = 0;

  FILE* fid = openFileOrStdout(output_fn);

  Batches batches(1024, test.size());
  for (Batches::iterator itr = batches.begin(); itr != batches.end(); ++itr) {
    mat prob = dnn.feedForward(test.getX(*itr));

    if (calcAcc && !silent)
      nError += zeroOneError(prob, test.getY(*itr), errorMeasure);

    if (calcAcc && output_fn.empty() && output_type == 0)
      continue;

    switch (output_type) {
      case 0: printLabels(prob, fid, base); break;
      case 1: prob.print(fid);	      break;
      case 2: log(prob).print(fid);   break;
    }
  }

  if (fid != stdout)
    fclose(fid);

  if (calcAcc && !silent)
    showAccuracy(nError, test.size());

  return 0;
}

// DO NOT USE device_matrix::print()
// (since labels should be printed as integer)
void printLabels(const mat& prob, FILE* fid, int base) {
  auto h_labels = copyToHost(posteriorProb2Label(prob));
  for (size_t i=0; i<h_labels.size(); ++i)
    fprintf(fid, "%d\n", (int) h_labels[i] + base);
}

FILE* openFileOrStdout(const string& filename) {

  FILE* fid = filename.empty() ? stdout : fopen(filename.c_str(), "w");

  if (fid == NULL) {
    fprintf(stderr, "Failed to open output file");
    exit(-1);
  }

  return fid;
}
