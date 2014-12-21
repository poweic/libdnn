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

#include <config.h>

Config::Config():
    learningRate(0.01),
    maxEpoch(1024),
    batchSize(32),
    trainValidRatio(5),
    nNonIncEpoch(6),
    minValidAccuracy(0),
    randperm(false) {
  }

void Config::print() const {

  printf(".________________________________________.\n");
  printf("|                                        |\n");
  printf("|             Configurations             |\n");
  printf("|______________________________._________|\n");
  printf("|                              |         |\n");
  printf("| learning rate                | %7g |\n", learningRate);
  printf("| maxEpoch                     | %7lu |\n", maxEpoch);
  printf("| batchSize                    | %7lu |\n", batchSize);
  printf("| training / validation        | %7lu |\n", trainValidRatio);
  printf("| minimun validation accuracy  | %7g |\n", minValidAccuracy);
  printf("| random permuation each epoch | %7s |\n", randperm?"true":"false");
  printf("|______________________________|_________|\n");
  printf("\33[34m[Note]\33[0m When the accuracy on validation set doesn't go up"
      "for %lu epochs, the training procedure would stop.\n\n", nNonIncEpoch);

}
