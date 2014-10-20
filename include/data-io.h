#ifndef __DATA_IO_H_
#define __DATA_IO_H_

#include <iostream>
#include <fstream>
#include <sstream>
#include <string>

#include <device_matrix.h>
#include <host_matrix.h>

typedef device_matrix<float> mat;
typedef host_matrix<float> hmat;

using namespace std;

struct BatchData {
    hmat x, y;
};

class DataStream {
public: 
  DataStream();
  DataStream(const string& filename);
  DataStream(const DataStream& src);

  size_t size() const;
  string get_filename() const;

  virtual DataStream* clone() const = 0;
  virtual void rewind() = 0;
  virtual void init(size_t start = 0, size_t end = -1) = 0;
  virtual BatchData read(int N, size_t dim, size_t base) = 0;

  static DataStream* create(const string& filename, size_t start = 0, size_t end = -1);

  enum {BEGIN = 0, END = -1};

protected:
  string _filename;
  size_t _size;
};

class KaldiStream : public DataStream {
public:
  KaldiStream();
  KaldiStream(const string& filename);
  KaldiStream(const KaldiStream& src);
  ~KaldiStream();

  KaldiStream& operator = (KaldiStream that);

  virtual DataStream* clone() const;
  virtual void init(size_t start = 0, size_t end = -1);
  virtual void rewind();
  virtual BatchData read(int N, size_t dim, size_t base);

  string get_feature_command() const;
  string get_label_command() const;

  friend void swap(KaldiStream& a, KaldiStream& b);

private:
  int _remained;

  FILE* _ffid;
  FILE* _lfid;
};

class BasicStream : public DataStream {
public:
  BasicStream();
  BasicStream(const string& filename, size_t start = 0, size_t end = -1);
  BasicStream(const BasicStream& src);
  ~BasicStream();

  BasicStream& operator = (BasicStream that);

  virtual DataStream* clone() const;
  virtual void init(size_t start = 0, size_t end = -1);
  virtual void rewind();
  virtual BatchData read(int N, size_t dim, size_t base);

  BatchData readSparseFeature(int N, size_t dim, size_t base);
  BatchData readDenseFeature(int N, size_t dim, size_t base);

  friend void swap(BasicStream& a, BasicStream& b);

  string getline();

public:
  bool _sparse;
  size_t _line_number;
  size_t _start;
  size_t _end;

  ifstream _fs;
};

size_t count_lines(const string& fn);
std::istream& go_to_line(std::istream& file, unsigned long num);

#endif // __DATA_IO_H_
