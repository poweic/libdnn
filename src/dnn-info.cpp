#include <iostream>
#include <string>
#include <dnn.h>
#include <cmdparser.h>
using namespace std;

int main (int argc, char* argv[]) {

  CmdParser cmd(argc, argv);

  cmd.add("model_file");

  cmd.addGroup("Example usage: dnn-info train.dat.model");

  if (!cmd.isOptionLegal())
    cmd.showUsageAndExit();

  string model_fn = cmd[1];

  DNN dnn(model_fn);

  dnn.status();

  return 0;
}
