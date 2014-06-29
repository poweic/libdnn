#ifndef __HOST_MATRIX_H_
#define __HOST_MATRIX_H_

template <typename T>
class host_matrix {
public:
  host_matrix():_rows(0), _cols(0), _capacity(0), _data(NULL) {}

  host_matrix(size_t r, size_t c): _rows(r), _cols(c), _capacity(r*c), _data(NULL) {
    _data = new T[_capacity];
    memset(_data, 0, sizeof(T) * _capacity);
  }

  host_matrix(const host_matrix<T>& source): _rows(source._rows), _cols(source._cols), _capacity(_rows * _cols), _data(NULL) {
    _data = new T[_capacity];
    memcpy(_data, source._data, sizeof(T) * _capacity);
  }

  host_matrix(const device_matrix<T>& source): _rows(source.getRows()), _cols(source.getCols()), _capacity(_rows * _cols), _data(NULL) {

    _data = new T[_capacity];
    CCE(cudaMemcpy(_data, source.getData(), sizeof(float) * size(), cudaMemcpyDeviceToHost));
    CCE(cudaDeviceSynchronize());
  }

  ~host_matrix() {
    if (_data != NULL)
      delete [] _data;
  }

  operator device_matrix<T> () const {
    device_matrix<T> d(_rows, _cols);
    CCE(cudaMemcpy(d.getData(), _data, sizeof(float) * size(), cudaMemcpyHostToDevice));
    CCE(cudaDeviceSynchronize());
    return d;
  }

  host_matrix& operator = (host_matrix<T> rhs) {
    swap(*this, rhs);
    return *this;
  }

  void resize(size_t r, size_t c, T x) {
    this->resize(r, c);
    this->fillwith(x);
  }

  void resize(size_t r, size_t c) {
    if (_rows == r && _cols == c)
      return;

    _rows = r;
    _cols = c;

    if (r * c <= _capacity)
      return;
    
    if (_data != NULL)
      delete [] _data;

    _capacity = _rows * _cols;
    _data = new T[_capacity];
    memset(_data, 0, sizeof(T) * _capacity);
  }

  void reserve(size_t capacity) {
    if (capacity <= _capacity)
      return;

    _capacity = capacity;
    if (_data == NULL) {
      _data = new T[_capacity];
      return;
    }

    T* buffer = new T[_capacity];
    memset(buffer, 0, sizeof(T) * _capacity);
    memcpy(buffer, _data, sizeof(T) * size());
    delete [] _data;
    _data = buffer;
  }

  T& operator() (size_t i, size_t j) {
    return _data[j * _rows + i];
  }
  const T& operator() (size_t i, size_t j) const {
    return _data[j * _rows + i];
  }

  T& operator[] (size_t idx) {
    return _data[idx];
  }

  const T& operator[] (size_t idx) const {
    return _data[idx];
  }

  host_matrix<T> operator ~ () const {
    host_matrix<T> t(_cols, _rows);

    for (size_t i=0; i<t._rows; ++i)
      for (size_t j=0; j<t._cols; ++j)
	t(i, j) = (*this)(j, i);

    return t;
  }

  void fillwith(T value) {
    std::fill(_data, _data + size(), value);
  }

  void print(FILE* fid = stdout) const {
    for (size_t i=0; i<_rows; ++i) {
      for (size_t j=0; j<_cols; ++j)
	fprintf(fid, "%.5f ", _data[j * _rows + i]);
      fprintf(fid, "\n");
    }

    fprintf(fid, "rows = %lu, cols = %lu, capacity = %lu\n", _rows, _cols, _capacity);
  }

  size_t size() const { return _rows * _cols; }
  size_t getRows() const { return _rows; }
  size_t getCols() const { return _cols; }
  T* getData() const { return _data; }

  template <typename S>
  friend void swap(host_matrix<S>& lhs, host_matrix<S>& rhs);
  
private:
  size_t _rows;
  size_t _cols;
  size_t _capacity;

  T* _data;
};

template <typename T>
void swap(host_matrix<T>& lhs, host_matrix<T>& rhs) {
  using std::swap;
  swap(lhs._rows, rhs._rows);
  swap(lhs._cols, rhs._cols);
  swap(lhs._capacity, rhs._capacity);
  swap(lhs._data, rhs._data);
}

/*void self_test() {
  host_matrix<float> A(21, 34);
  size_t n = 0;
  for (size_t i=0; i<A.getRows(); ++i)
    for (size_t j=0; j<A.getCols(); ++j)
      A(i, j) = ++n;
  A.print();

  host_matrix<float> B;
  B = A;
  B.print();

  host_matrix<float> C = B;
  C.print();

  host_matrix<float> D(A);
  D.print();

  D.resize(11, 24);
  D.print();

  D.resize(21, 40);
  D.print();

  B.reserve(21*40);
  B.resize(21, 40);
  B.print();
}*/

#endif // __HOST_MATRIX_H_
