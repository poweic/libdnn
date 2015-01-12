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
  BatchData(): multi_label(false) {}
  hmat x, y;
  bool multi_label;
};

class InputStream {
public: 
  InputStream();
  InputStream(const string& filename);
  InputStream(const InputStream& src);

  virtual InputStream* clone() const = 0;
  virtual void rewind() = 0;
  virtual string getline() = 0;

  string get_fn() const { return _fn; }

  bool empty() const { return _fn.empty(); }

  enum {BEGIN = 0, END = -1};

protected:
  string _fn;    // Filename
};

class FileStream : public InputStream {
public:
  FileStream();
  FileStream(const string& filename, size_t start, size_t end);
  FileStream(const FileStream& src);
  ~FileStream();

  void init();
  istream& GoToLine(istream& file, unsigned long num);
  void setRange(size_t start, size_t end);
  FileStream& operator = (const FileStream& that) = delete;

  virtual InputStream* clone() const;
  virtual void rewind();

  static size_t CountLines(const string& filename);

  friend void swap(FileStream& a, FileStream& b);

  virtual string getline();

public:
  bool _sparse;
  size_t _line_number;
  size_t _start;
  size_t _end;

  ifstream _fs;  // fs for feature file's ifstream
};

class PipeStream : public InputStream {
public:
  PipeStream();
  PipeStream(const string& filename);
  PipeStream(const PipeStream& src);
  ~PipeStream();

  void init();
  PipeStream& operator = (const PipeStream& that) = delete;

  virtual InputStream* clone() const;
  virtual void rewind();
  virtual string getline();

  FILE* get_fp() { return _fp; };

  friend void swap(PipeStream& a, PipeStream& b);

protected:
  FILE* _fp;
};

/*
 * File Parser: Sparse Format, Dense Format, KaldiArchive, KaldiLabel
 *
 * */

class IFileParser {
public:
  IFileParser();
  IFileParser(const IFileParser& src);

  virtual IFileParser* clone() const = 0;

  virtual void read(hmat* x, int N, size_t dim, hmat* y = nullptr, size_t base = 0) = 0;
  virtual void setRange(size_t start, size_t end);

  virtual void rewind() { _is->rewind(); }

  enum Format {
    Sparse,
    Dense,
    KaldiArchive,
    KaldiLabel,
    Unknown   /* if filename is empty */
  };

  static Format GetFormat(const string& filename);
  static IFileParser* create(const string& filename, IFileParser::Format format, size_t size);

protected:
  InputStream* _is;
};

class SparseParser : public IFileParser {
public:
  SparseParser();
  SparseParser(const string& filename, size_t start, size_t end);

  virtual IFileParser* clone() const;

  virtual void read(hmat* x, int N, size_t dim, hmat* y = nullptr, size_t base = 0);
};

class DenseParser : public IFileParser {
public:
  DenseParser();
  DenseParser(const string& filename, size_t start, size_t end);

  virtual IFileParser* clone() const;

  virtual void read(hmat* x, int N, size_t dim, hmat* y = nullptr, size_t base = 0);
};

class KaldiArchiveParser : public IFileParser {
public:
  KaldiArchiveParser();
  KaldiArchiveParser(const string& filename);

  virtual IFileParser* clone() const;
  virtual void read(hmat* x, int N, size_t dim, hmat* y = nullptr, size_t base = 0);
  virtual void rewind();

  static size_t CountLines(const string& filename);
private:
  int _remained;
};

class KaldiLabelParser : public IFileParser {
public:
  KaldiLabelParser();
  KaldiLabelParser(const string& filename);

  virtual IFileParser* clone() const;
  virtual void read(hmat* x, int N, size_t dim, hmat* y = nullptr, size_t base = 0);
  virtual void rewind();

  static size_t CountLines(const string& filename);
private:
  vector<size_t> _remained;
};

#endif // __DATA_IO_H_
