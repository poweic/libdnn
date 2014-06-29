#ifndef __DATA_IO_H_
#define __DATA_IO_H_

#include <iostream>
#include <fstream>
#include <string>

using namespace std;

class DataStream {
public:
  DataStream();
  DataStream(const string& filename, size_t start = 0, size_t end = -1);
  DataStream(const DataStream& src);
  ~DataStream();

  DataStream& operator = (DataStream that);

  friend void swap(DataStream& a, DataStream& b);

  string get_filename() const;
  size_t get_line_number() const;
  bool is_pipe() const { return _is_pipe; }
  void init(const string& filename, size_t start, size_t end);

  string getline();
  void rewind();

public:
  size_t _nLines;
  size_t _line_number;
  string _filename;

  int _remained;

  FILE* _feat_ps;
  FILE* _label_ps;

  string _feat_command;
  string _label_command;

  ifstream _fs;

  bool _is_pipe;

  size_t _start, _end;
};

size_t count_lines(const string& fn);
std::istream& go_to_line(std::istream& file, unsigned long num);

#endif // __DATA_IO_H_
