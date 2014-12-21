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
#include <nnet.h>
#include <dnn-utility.h>
#include <cmdparser.h>
using namespace std;

int main (int argc, char* argv[]) {

  CmdParser cmd(argc, argv);
  cmd.printArgs();

  cmd.add("model_in")
     .add("model_out");

  cmd.addGroup("Example usage: dnn-transpose a.mdl aT.mdl");

  if (!cmd.isOptionLegal())
    cmd.showUsageAndExit();

  string model_in     = cmd[1];
  string model_out    = cmd[2];

  // Load model
  NNet nnet(model_in);

  if (AffineTransform* T = dynamic_cast<AffineTransform*>(nnet.getTransforms()[0])) {
    mat M(351, 351);
    memcpy2D(M, T->get_w(), 0, 0, 351, 351, 0, 0);
    M = ~M;
    memcpy2D(T->get_w(), M, 0, 0, 351, 351, 0, 0);
  }

  nnet.save(model_out);

  return 0;
}
