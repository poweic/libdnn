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

#include <batch.h>

Batches::Batches(size_t batchSize, size_t totalSize):
  _batchSize(batchSize), _totalSize(totalSize),
  _begin(0, _batchSize, _totalSize),
  _end(-1, _batchSize, _totalSize) { }


void swap(Batches::iterator& lhs, Batches::iterator& rhs) {
  std::swap(lhs._batchSize, rhs._batchSize);
  std::swap(lhs._totalSize, rhs._totalSize);
  std::swap(lhs._batch, rhs._batch);
}
