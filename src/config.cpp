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
    variance(0.01),
    batchSize(32),
    trainValidRatio(5),
    nNonIncEpoch(6),
    minValidAccuracy(0),
    randperm(false) {
  }

void Config::print() const {

  printf("| learning rate                  | %9g |\n", learningRate);
  printf("| maxEpoch                       | %9lu |\n", maxEpoch);
  printf("| std for random init            | %9g |\n", variance);
  printf("| batchSize                      | %9lu |\n", batchSize);
  printf("| training / validation          | %9lu |\n", trainValidRatio);
  printf("| minimun validation accuracy    | %9g |\n", minValidAccuracy);
  printf("| random permuation each epoch   | %9s |\n", randperm?"true":"false");
  printf("+--------------------------------+-----------+\n");
  printf("\33[34m[Note]\33[0m When the accuracy on validation set doesn't go up"
      "for %lu epochs, the training procedure would stop.\n\n", nNonIncEpoch);

}
