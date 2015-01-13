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

#include <utility.h>
#include <data-io.h>
#include <stdexcept>
#include <algorithm>
#include <cassert>

#define CFRE(x) { if (x == 0) throw std::runtime_error(RED_ERROR + "Failed when read features. "); }

/* \brief Factory method for class InputStream
 *
 * */

InputStream::InputStream() {
}

InputStream::InputStream(const string& filename) : _fn(filename) {
}

InputStream::InputStream(const InputStream& src) : _fn(src._fn) {
}

/* Other Utility Functions 
 *
 * */
bool isFileSparse(string fn) {
  ifstream fin(fn.c_str());
  string line;
  std::getline(fin, line);
  fin.close();
  return line.find(':') != string::npos;
}

/* \brief constructor of class FileStream
 *
 * */
FileStream::FileStream(): _sparse(true), _line_number(0), _start(0), _end(-1) {
}

FileStream::FileStream(const string& filename, size_t start, size_t end):
  InputStream(filename), _sparse(isFileSparse(filename)), _line_number(0), _start(start), _end(end) {
  this->init();
  this->setRange(start, end);
}

FileStream::FileStream(const FileStream& src) : InputStream(src),
  _sparse(src._sparse), _line_number(src._line_number), _start(src._start), _end(src._end) {
  this->init();
  this->setRange(_start, _end);
}

FileStream::~FileStream() {
  _fs.close();
}

void FileStream::init() {
  if (_fs.is_open())
    _fs.close();

  // Read from normal file
  _fs.open(_fn.c_str());

  if (!_fs.is_open())
    throw std::runtime_error(RED_ERROR + "Cannot load file: " + _fn);
}

istream& FileStream::GoToLine(istream& file, unsigned long num) {
  file.seekg(std::ios::beg);
  
  if (num == 0)
    return file;

  for(size_t i=0; i < num; ++i)
    file.ignore(std::numeric_limits<std::streamsize>::max(),'\n');

  return file;
}

void FileStream::setRange(size_t start, size_t end) {
  _start = start;

  GoToLine(_fs, _start);
  _line_number = _start;
}

InputStream* FileStream::clone() const {
  return new FileStream(*this);
}

string FileStream::getline() {
  string line;

  if ( _line_number >= _end )
    this->rewind();

  if ( !std::getline(_fs, line) ) {
    this->rewind();
    std::getline(_fs, line);
  }

  ++_line_number;

  return line;
}

void FileStream::rewind() {
  _fs.clear();
  GoToLine(_fs, _start);
  _line_number = _start;
}

size_t FileStream::CountLines(const string& fn) {
  clog << "Loading file: \33[32m" << fn << "\33[0m (try to find out how many data) ...";
  clog.flush();
  
  std::ifstream fin(fn.c_str()); 
  size_t N = std::count(std::istreambuf_iterator<char>(fin), 
      std::istreambuf_iterator<char>(), '\n');
  fin.close();

  clog << "\t\33[32m[Done]\33[0m" << endl;
  return N;
}

void swap(FileStream& a, FileStream& b) { 
  std::swap(a._sparse, b._sparse);
  std::swap(a._line_number, b._line_number);
  std::swap(a._start, b._start);
  std::swap(a._end, b._end);
}

/*
 * implementation of PipeStream goes here.
 *
 * */
string read_uid(FILE* fid) {
  char buffer[512];
  int result = fscanf(fid, "%s ", buffer);

  return (result != 1) ? "" : string(buffer);
}

PipeStream::PipeStream(): _fp(nullptr) {
}

PipeStream::PipeStream(const string& filename): InputStream(filename), _fp(nullptr) {
  this->init();
}

PipeStream::PipeStream(const PipeStream& src): _fp(src._fp) {
}

PipeStream::~PipeStream() {
  if (_fp)
    pclose(_fp);
}

void PipeStream::init() {

  if (_fp) {
    pclose(_fp);
    _fp = nullptr;
  }

  _fp = popen(this->get_fn().c_str(), "r");

  if (!_fp)
    throw runtime_error(RED_ERROR + "Failed to open pipe: \"" + this->get_fn() + "\"");
}

InputStream* PipeStream::clone() const {
  return new PipeStream(*this);
}

void PipeStream::rewind() {
  this->init();
}

string PipeStream::getline() {
  char* line = nullptr;
  size_t len = 0;
  ssize_t read = ::getline(&line, &len, _fp);

  return (read != -1) ? line : "";
}

void swap(PipeStream& a, PipeStream& b) {
  std::swap(a._fp, b._fp);
}

/*
 * Implementation of SparseParser goes here.
 *
 * */

IFileParser::IFileParser(): _is(nullptr) {

}

IFileParser::IFileParser(const IFileParser& src): _is(nullptr) {
  if (src._is)
    _is = src._is->clone();
}

void IFileParser::setRange(size_t start, size_t end) {
  auto ptr = dynamic_cast<FileStream*>(_is);

  if (ptr != nullptr)
    ptr->setRange(start, end);
}

IFileParser::Format IFileParser::GetFormat(const string& filename) {
  if (filename.empty())
    return Unknown;

  if (filename.size() > 4 && filename.substr(0, 4) == "ark:")
    return KaldiArchive;
  else if (filename.size() > 4 && filename.substr(0, 4) == "scp:")
    return KaldiLabel;
  else if (isFileSparse(filename))
    return Sparse;
  else
    return Dense;
}

IFileParser* IFileParser::create(const string& filename, IFileParser::Format format, size_t size) {
  switch (format) {
    case Sparse:
      return new SparseParser(filename, 0, size);
    case Dense:
      return new DenseParser(filename, 0, size);
    case KaldiArchive:
      return new KaldiArchiveParser(filename.substr(4));
    case KaldiLabel:
      return new KaldiLabelParser(filename.substr(4));
    case Unknown:
      return nullptr;
  }
}

/*
 * Implementation of SparseParser goes here.
 *
 * */
SparseParser::SparseParser() {
}

SparseParser::SparseParser(const string& filename, size_t start, size_t end) {
  IFileParser::_is = new FileStream(filename, start, end);
}

IFileParser* SparseParser::clone() const {
  return new SparseParser(*this);
}

void SparseParser::read(hmat* x, int N, size_t dim, hmat* y, size_t base) {

  string token;

  for (int i=0; i<N; ++i) {
    stringstream ss(IFileParser::_is->getline());

    // string token can be either "idx:value" (with ':') or "label" (without ':')
    while (ss >> token) {
      auto tokens = splitAsInt(token, ':');

      // if it's label, there's no ':' in string token => tokens.size() == 1
      if (tokens.size() == 1) {
	if (y)
	  y->get(i) = tokens[0] - base;
      }
      else {
	size_t idx = tokens[0];
	size_t value = tokens[1];

	if (idx == 0)
	  throw std::runtime_error(RED_ERROR + "Index in sparse format should"
	      " be started from 1 instead of 0. ( like 1:... not 0:... )");

	x->get(i, idx - 1) = value;
      }

    }
  }
}

/*
 * Implementation of DenseParser goes here.
 *
 * */
DenseParser::DenseParser() {
}

DenseParser::DenseParser(const string& filename, size_t start, size_t end) {
  IFileParser::_is = new FileStream(filename, start, end);
}

IFileParser* DenseParser::clone() const {
  return new DenseParser(*this);
}

void DenseParser::read(hmat* x, int N, size_t dim, hmat* y, size_t base) {
  string token;

  for (int i=0; i<N; ++i) {
    size_t j = 0;
    stringstream ss(IFileParser::_is->getline());

    ss >> token;
    if (y)
      y->get(i) = stof(token) - base;
    else
      x->get(i, j++) = stof(token);

    while (ss >> token)
      x->get(i, j++) = stof(token);
  }
}

/*!
 * Implementation of KaldiArchiveParser goes here.
 *
 * */

KaldiArchiveParser::KaldiArchiveParser(): _remained(0) {
}

KaldiArchiveParser::KaldiArchiveParser(const string& filename): _remained(0) {
  IFileParser::_is = new PipeStream(filename);
  clog << "Reading feature from \33[33m\"" << IFileParser::_is->get_fn() << "\"\33[0m" << endl;
}

IFileParser* KaldiArchiveParser::clone() const {
  return new KaldiArchiveParser(*this);
}

void KaldiArchiveParser::read(hmat* x, int N, size_t dim, hmat* y, size_t base) {

  // Read kaldi feature
  FILE* fp = dynamic_cast<PipeStream*>(IFileParser::_is)->get_fp();
  int counter = 0;

  while (true) {

    if (_remained == 0) {
      string uid = read_uid(fp);

      char s[6]; 
      int frame;
      int dimension;

      CFRE(fread((void*) s, 6, 1, fp));
      CFRE(fread((void*) &frame, 4, 1, fp));
      CFRE(fread((void*) s, 1, 1, fp));
      CFRE(fread((void*) &dimension, 4, 1, fp));

      if (dimension != (int) dim)
	throw std::runtime_error(RED_ERROR + "feature dimension in kaldi archive (=" +
	    to_string(dimension) + ") does not match --input-dim (=" +
	    to_string(dim) + ").");

      _remained = frame;
    }

    for(int i = 0; i < _remained; i++) {
      for(int j = 0; j < (int) dim; j++)
	CFRE(fread((void*) &(x->get(counter, j)), sizeof(float), 1, fp));

      if (++counter == N) {
	_remained -= i + 1;
	return;
      }
    }

    _remained = 0;
  }
}

void KaldiArchiveParser::rewind() {
  IFileParser::rewind();
  this->_remained = 0;
}

size_t KaldiArchiveParser::CountLines(const string& filename) {

  // Use wc to count # of features
  string wc_count_lines = filename + "| feat-to-len --print-args=false ark:- ark,t:- | cut -f 2 -d ' '";

  size_t totalLength = 0, length = 0;

  FILE* fid = popen(wc_count_lines.c_str(), "r");
  if (!fid)
    throw runtime_error(RED_ERROR + "Failed to open " + filename);

  while (fscanf(fid, "%lu", &length) == 1)
    totalLength += length;
  pclose(fid);

  return totalLength;
}


/*!
 * Implementation of KaldiLabelParser goes here.
 *
 * */

KaldiLabelParser::KaldiLabelParser() {
}

KaldiLabelParser::KaldiLabelParser(const string& filename) {
  IFileParser::_is = new PipeStream(filename);
  clog << "Reading label from \33[33m\"" << IFileParser::_is->get_fn() << "\"\33[0m" << endl;
}

IFileParser* KaldiLabelParser::clone() const {
  return new KaldiLabelParser(*this);
}

void KaldiLabelParser::read(hmat* x, int N, size_t dim, hmat* y, size_t base) {

  int counter = 0;

  while (counter < N) {
    if (_remained.empty()) {
      string line = IFileParser::_is->getline();
      size_t pos = line.find_first_of(' ') + 1;
      _remained = splitAsInt(rtrim(line.substr(pos)), ' ');
    }

    // Read kaldi Label
    while (!_remained.empty()) {

      x->get(counter) = _remained[0];
      _remained.erase(_remained.begin());

      if (++counter == N)
	return;
    }
  }
}

void KaldiLabelParser::rewind() {
  IFileParser::rewind();
  this->_remained.clear();
}

size_t KaldiLabelParser::CountLines(const string& filename) {

  // Use wc to count # of features
  string wc_count_lines = filename + "| cut -f 2- -d ' ' | wc -w";

  size_t totalLength;

  FILE* fid = popen(wc_count_lines.c_str(), "r");
  if (!fid)
    throw runtime_error(RED_ERROR + "Failed to open " + filename);

  if (fscanf(fid, "%lu", &totalLength) != 1)
    throw std::runtime_error(RED_ERROR + "Failed to count number of labels");
  pclose(fid);

  return totalLength;
}
