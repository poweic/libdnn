// Copyright 2013-2014 [Author: Po-Wei Chou]
// 
// Licensed under the Apache License, Version 2.0 (the "License");
// you may not use this file except in compliance with the License.
// You may obtain a copy of the License at
// 
//     http://www.apache.org/licenses/LICENSE-2.0
// 
// Unless required by applicable law or agreed to in writing, software
// distributed under the License is distributed on an "AS IS" BASIS,
// WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
// See the License for the specific language governing permissions and
// limitations under the License.

#include <iostream>
#include <string>
#include <dnn.h>
#include <cmdparser.h>
using namespace std;

int main (int argc, char* argv[]) {

  CmdParser cmd(argc, argv);

  cmd.add("input-model")
     .add("output-model", false);

  cmd.addGroup("Options: ")
     .add("--layer", "Specify which layers to copy (or dump). "
	 "Ex: 1:2:9 means only print out layer 1, 2 and 9.", "all");

  cmd.addGroup("Example usage: dnn-print train.dat.model");

  if (!cmd.isOptionLegal())
    cmd.showUsageAndExit();

  string input_model_fn  = cmd[1];
  string output_model_fn = cmd[2];
  string layer		 = cmd["--layer"];

  FILE* fid = output_model_fn == "" ? stdout : fopen(output_model_fn.c_str(), "w");
  if (!fid)
    throw std::runtime_error(RED_ERROR + "Cannot open file " + output_model_fn);

  DNN dnn(input_model_fn);

  const auto& t = dnn.getTransforms();

  vector<size_t> layerIds;

  if (layer == "all") {
    for (size_t i=0; i<t.size(); ++i)
      layerIds.push_back(i);
  }
  else
    layerIds = splitAsInt(layer, ':');

  for (auto l : layerIds) { 
    if (l < t.size())
      t[l]->write(fid);
    else
      cerr << YELLOW_WARNING << "Model does not have transform[" << l << "]" << endl;
  }

  if (fid != stdout)
    fclose(fid);

  return 0;
}
