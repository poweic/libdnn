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
#include <cmdparser.h>
using namespace std;

int main (int argc, char* argv[]) {

  CmdParser cmd(argc, argv);

  cmd.add("model_file");

  cmd.addGroup("Example usage: dnn-info train.dat.xml");

  if (!cmd.isOptionLegal())
    cmd.showUsageAndExit();

  string model_fn = cmd[1];

  NNet nnet(model_fn);

  nnet.status();

  return 0;
}
