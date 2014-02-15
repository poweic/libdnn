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
