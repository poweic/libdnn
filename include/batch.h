#ifndef __BATCH_H_
#define __BATCH_H_

class Batches {
public:
  Batches(size_t batchSize, size_t totalSize):
    _batchSize(batchSize), _totalSize(totalSize),
    _begin(0, _batchSize, _totalSize),
    _end(-1, _batchSize, _totalSize) {
  }

  class iterator;

  class Batch {
    public:
      size_t offset;
      size_t nData;

      friend class iterator;
    private:
      Batch(size_t offset, size_t nData): offset(offset), nData(nData) {}
  };

  class iterator {
    public:
      iterator(const iterator& source):
	_batchSize(source._batchSize), _totalSize(source._totalSize) {
	  _batch = new Batch(source._batch->offset, source._batch->nData);
      }
      ~iterator() { delete _batch; }

      iterator& operator = (iterator rhs);

      iterator& operator ++ ()	  { _batch->offset += _batchSize; return *this; }
      iterator operator ++ (int)  { _batch->offset += _batchSize; return *this; }

      Batch* operator -> () const {
	if (_batch->offset + _batch->nData >= _totalSize)
	  _batch->nData = _totalSize - _batch->offset;

	return _batch;
      }

      bool operator == (const iterator& rhs) { return _batch->offset == rhs._batch->offset; }
      bool operator != (const iterator& rhs) { return _batch->offset != rhs._batch->offset; }

      friend class Batches;
      friend void swap(iterator& lhs, iterator& rhs);

    private:
      size_t _batchSize;
      size_t _totalSize;

      Batch* _batch;

      iterator(size_t index, size_t batchSize, size_t totalSize):
	_batchSize(batchSize), _totalSize(totalSize) {
	  size_t offset = index;
	  if (offset == -1)
	    offset = ceil((float) _totalSize / _batchSize) * _batchSize;

	  _batch = new Batch(offset, _batchSize);
      }
  };

  const iterator& begin() const { return _begin; }
  const iterator& end() const { return _end; }

  size_t size() const { return ceil((float) _totalSize / _batchSize); }

private:
  size_t _batchSize;
  size_t _totalSize;
  iterator _begin;
  iterator _end;
};

void swap(Batches::iterator& lhs, Batches::iterator& rhs) {
  std::swap(lhs._batchSize, rhs._batchSize);
  std::swap(lhs._totalSize, rhs._totalSize);
  std::swap(lhs._batch, rhs._batch);
}


#endif // __BATCH_H_
