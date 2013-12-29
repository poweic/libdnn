#include <iostream>
#include <string>
#include <dnn.h>
#include <dnn-utility.h>
#include <cmdparser.h>
using namespace std;

void saveFeature(string fn, const vector<float>& X, size_t nData, size_t dim) {

  FILE* fout = fopen(fn.c_str(), "w");

  for (size_t i=0; i<nData; ++i) {
    fprintf(fout, "train_%d [\n  ", i);
    for (size_t j=0; j<dim; ++j) {
      fprintf(fout, "%g ", X[j*nData + i]);
    }
    fprintf(fout, "]\n");
  }

  fclose(fout);
}

int main (int argc, char* argv[]) {

  CmdParser cmd(argc, argv);

  cmd.addGroup("Input / Output:")
     .add("svm_format_file_in")
     .add("kaldi_format_file_out");

  if (!cmd.isOptionLegal())
    cmd.showUsageAndExit();

  string file_in = cmd[1];
  string file_out= cmd[2];

  DataSet data;
  getFeature(file_in, data);
  showSummary(data);

  data.X.resize(data.X.getRows(), data.X.getCols() - 1);

  DataSet train, valid;
  splitIntoTrainingAndValidationSet(train, valid, data, 5);

  vector<float> X = copyToHost(train.X);

  saveFeature(file_out, copyToHost(train.X), train.X.getRows(), train.X.getCols());
  saveFeature(file_out + ".label", copyToHost(train.y), train.y.getRows(), train.y.getCols());

  file_out += ".valid";
  saveFeature(file_out, copyToHost(valid.X), valid.X.getRows(), valid.X.getCols());
  saveFeature(file_out + ".label", copyToHost(valid.y), valid.y.getRows(), valid.y.getCols());

  return 0;
}

