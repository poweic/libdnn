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

#include <dataset.h>
#include <utility.h>
#include <cmdparser.h>
using namespace std;

int main (int argc, char* argv[]) {

  CmdParser cmd(argc, argv);
  cmd.printArgs();

  cmd.add("data-in")
     .add("statistics-out", false);

  cmd.addGroup("Feature options:")
     .add("--input-dim", "specify the input dimension (dimension of feature).\n"
	 "0 for auto detection.")
     .add("--normalize", "Feature normalization: \n"
	"0 -- Do not normalize.\n"
	"1 -- Rescale each dimension to [0, 1] respectively.\n"
	"2 -- Normalize to standard score. z = (x-u)/sigma .");

  cmd.addGroup("Example usage: data-statistics data/train3.dat train3.stat --input-dim 351 --normalize 2");

  if (!cmd.isOptionLegal())
    cmd.showUsageAndExit();

  string data_fn = cmd[1];
  string stat_fn = cmd[2];

  size_t input_dim    = cmd["--input-dim"];
  NormType n_type     = (NormType) (int) cmd["--normalize"];

  DataSet data(data_fn, input_dim);
  data.normalize(n_type);
  data.showSummary();

  FILE* fid = stat_fn.empty() ? stdout : fopen(stat_fn.c_str(), "w");

  if (!fid)
    throw std::runtime_error(RED_ERROR + "Cannot open file to write:" + stat_fn);

  data.getNormalizer()->print(fid);

  if (fid != stdout)
    fclose(fid);

  return 0;
}
