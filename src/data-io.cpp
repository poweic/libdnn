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
#define DEBUG_STR(x) ("\33[33m"#x"\33[0m = " + to_string(x) + "\t")

/* \brief Factory method for class DataStream
 *
 * */

DataStream::DataStream() : _size(0) {
}

DataStream::DataStream(const string& filename) : _filename(filename), _size(0) {
}

DataStream::DataStream(const DataStream& src) : _filename(src._filename), _size(src._size) {
}

DataStream* DataStream::create(const string& filename, size_t start, size_t end) {
  // Find if filename contains "ark:" as prefix. if yes, we'll read it from pipe.
  bool is_kaldi = (filename.size() > 4 && filename.substr(0, 4) == "ark:");

  if (is_kaldi)
    return new KaldiStream(filename);
  else
    return new BasicStream(filename, start, end);
}

size_t DataStream::size() const {
  return _size;
}

string DataStream::get_filename() const {
  return _filename;
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

/* \brief constructor of class BasicStream
 *
 * */
BasicStream::BasicStream(): _sparse(true), _line_number(0), _start(0), _end(-1) {
}

BasicStream::BasicStream(const string& filename, size_t start, size_t end):
  DataStream(filename), _sparse(isFileSparse(filename)), _line_number(0), _start(start), _end(end) {
  this->init(start, end);
}

BasicStream::BasicStream(const BasicStream& src) : DataStream(src),
  _sparse(src._sparse), _line_number(src._line_number), _start(src._start), _end(src._end) {
  this->init(_start, _end);
}

BasicStream::~BasicStream() {
  _fs.close();
}

BasicStream& BasicStream::operator = (BasicStream that) {
  swap(*this, that);
  return *this;
}

void BasicStream::init(size_t start, size_t end) {

  _start = start;
  _end = end;
  _line_number = _start;

  // Read from normal file
  if (_fs.is_open())
    _fs.close();

  _fs.open(DataStream::_filename.c_str());

  if (!_fs.is_open())
    throw std::runtime_error(RED_ERROR + "Cannot load file: " + DataStream::_filename);

  if (DataStream::_size == 0)
    DataStream::_size = count_lines(DataStream::_filename);

  go_to_line(_fs, _start);

  _end = min(DataStream::_size, _end);
  DataStream::_size = min(DataStream::_size, _end - _start);
}

DataStream* BasicStream::clone() const {
  return new BasicStream(*this);
}

BatchData BasicStream::read(int N, size_t dim, size_t base) {

  if (_sparse)
    return this->readSparseFeature(N, dim, base);
  else
    return this->readDenseFeature(N, dim, base);
}

BatchData BasicStream::readSparseFeature(int N, size_t dim, size_t base) {

  BatchData data;
  data.x.resize(N, dim + 1, 0);
  data.y.resize(N, 1, 0);

  string token;

  for (int i=0; i<N; ++i) {
    stringstream ss(this->getline());
  
    ss >> token;
    data.y[i] = stof(token);

    while (ss >> token) {
      size_t pos = token.find(':');
      if (pos == string::npos)
	continue;

      size_t j = stof(token.substr(0, pos));

      if (j == 0)
	throw std::runtime_error(RED_ERROR + "Index in sparse format should"
	    " be started from 1 instead of 0. ( like 1:... not 0:... )");

      float value = stof(token.substr(pos + 1));

      data.x(i, j - 1) = value;
    }
  
    // FIXME I'll remove it and move this into DNN. Since bias is only need by DNN,
    // not by CNN or other classifier.
    data.x(i, dim) = 1;
  }

  for (int i=0; i<N; ++i)
    data.y[i] -= base;

  return data;
}

BatchData BasicStream::readDenseFeature(int N, size_t dim, size_t base) {

  BatchData data;
  data.x.resize(N, dim + 1, 0);
  data.y.resize(N, 1, 0);
  
  string token;

  for (int i=0; i<N; ++i) {
    stringstream ss(this->getline());
  
    ss >> token;
    data.y[i] = stof(token);

    size_t j = 0;
    while (ss >> token)
      data.x(i, j++) = stof(token);

    // FIXME I'll remove it and move this into DNN. Since bias is only need by DNN,
    // not by CNN or other classifier.
    data.x(i, dim) = 1;
  }

  for (int i=0; i<N; ++i)
    data.y[i] -= base;
  
  return data;
}

string BasicStream::getline() {
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

void BasicStream::rewind() {
  _fs.clear();
  go_to_line(_fs, _start);
  _line_number = _start;
}

void swap(BasicStream& a, BasicStream& b) { 
  std::swap(a._sparse, b._sparse);
  std::swap(a._line_number, b._line_number);
  std::swap(a._start, b._start);
  std::swap(a._end, b._end);
}

/*
 * implementation of Kaldi DataStream
 *
 * */
string read_uid(FILE* fid) {
  char buffer[512];
  int result = fscanf(fid, "%s ", buffer);

  return (result != 1) ? "" : string(buffer);
}

KaldiStream::KaldiStream(): _remained(0), _ffid(nullptr), _lfid(nullptr) {
}

KaldiStream::KaldiStream(const string& filename): DataStream(filename),
  _remained(0), _ffid(nullptr), _lfid(nullptr) {
  this->init();
}

KaldiStream::KaldiStream(const KaldiStream& src): _remained(src._remained),
  _ffid(src._ffid), _lfid(src._lfid) {
}

KaldiStream::~KaldiStream() {
  if (_ffid) pclose(_ffid);
  if (_lfid) pclose(_lfid);
}

KaldiStream& KaldiStream::operator = (KaldiStream that) {
  swap(*this, that);
  return *this;
}

void KaldiStream::init(size_t start, size_t end) {

  clog << "Reading feature from \33[33m\"" << this->get_feature_command() << "\"\33[0m" << endl;
  clog << "Reading label from \33[33m\"" << this->get_label_command() << "\"\33[0m" << endl;

  // Use wc to count # of features
  // "| feat-to-len --print-args=false ark:- ark,t:- | cut -f 2 -d ' '";
  string wc_count_words = this->get_label_command() + "| cut -f 2- -d ' ' | wc -w";

  FILE* fid = popen(wc_count_words.c_str(), "r");
  if (fscanf(fid, "%lu", &(DataStream::_size)) != 1)
    throw std::runtime_error(RED_ERROR + "Failed to count number of labels");
  pclose(fid);

  clog << util::blue("[Info]") << " Found " << util::green(to_string(DataStream::_size))
       << " labels in \"" << this->get_label_command() << "\"" << endl;

  _ffid = popen(this->get_feature_command().c_str(), "r");

  if (! (this->get_label_command().empty()) )
    _lfid = popen(this->get_label_command().c_str(), "r");
}

DataStream* KaldiStream::clone() const {
  return new KaldiStream(*this);
}

BatchData KaldiStream::read(int N, size_t dim, size_t base) {
  BatchData data;
  data.x.resize(N, dim + 1, 0);
  data.y.resize(N, 1, 0);

  // Read kaldi feature
  FILE* &fis = this->_ffid;
  FILE* &lis = this->_lfid;

  int counter = 0;
  int& r = this->_remained;

  while (true) {

    if (r == 0) {
      string uid1, uid2;
      uid1 = read_uid(fis);

      if (lis != nullptr) {
	uid2 = read_uid(lis);

	if (uid1.empty() or uid2.empty()) {
	  this->rewind();
	  uid1 = read_uid(fis);
	  uid2 = read_uid(lis);
	}

	if (uid1 != uid2)
	  throw std::runtime_error(RED_ERROR + "uid1 != uid2 (\"" + uid1 + "\" != \"" + uid2 + "\")");
      }

      char s[6]; 
      int frame;
      int dimension;

      CFRE(fread((void*) s, 6, 1, fis));
      CFRE(fread((void*) &frame, 4, 1, fis));
      CFRE(fread((void*) s, 1, 1, fis));
      CFRE(fread((void*) &dimension, 4, 1, fis));

      if (dimension != (int) dim)
	throw std::runtime_error(RED_ERROR + "feature dimension in kaldi archive (=" +
	    to_string(dimension) + ") does not match --input-dim (=" +
	    to_string(dim) + ").");

      r = frame;
    }

    for(int i = 0; i < r; i++) {
      for(int j = 0; j < (int) dim; j++)
	CFRE(fread((void*) &data.x(counter, j), sizeof(float), 1, fis));
      data.x(counter, dim) = 1;

      if (lis != nullptr) {
	size_t y;
	CFRE(fscanf(lis, "%lu", &y));
	data.y[counter] = y;
      }

      if (++counter == N) {
	r -= i + 1;
	return data;
      }
    }

    r = 0;
  }

  return data;
}

void KaldiStream::rewind() {
  pclose(_ffid);
  _ffid = popen(this->get_feature_command().c_str(), "r");

  if (_lfid != nullptr)
    pclose(_lfid);

  if (! (this->get_label_command().empty()) )
    _lfid = popen(this->get_label_command().c_str(), "r");
  
  this->_remained = 0;
}

string KaldiStream::get_feature_command() const {
  size_t pos = DataStream::_filename.find_first_of(",");

  /*if (pos == string::npos)
    throw runtime_error(RED_ERROR + "Please specify feature and label like ark:feat.ark,label.ark");*/

  return DataStream::_filename.substr(4, pos - 4);
}

string KaldiStream::get_label_command() const {
  size_t pos = DataStream::_filename.find_first_of(",");

  /*if (pos == string::npos)
    throw runtime_error(RED_ERROR + "Please specify feature and label like ark:feat.ark,label.ark");*/

  if (pos == string::npos)
    return "";
  else
    return DataStream::_filename.substr(pos + 1);
}

void swap(KaldiStream& a, KaldiStream& b) {
  std::swap(a._remained, b._remained);
  std::swap(a._ffid, b._ffid);
  std::swap(a._lfid, b._lfid);
}

/* 
 * Other utilities
 *
 * */
size_t count_lines(const string& fn) {

  clog << "Loading file: \33[32m" << fn << "\33[0m (try to find out how many data) ...";
  clog.flush();
  
  std::ifstream fin(fn.c_str()); 
  size_t N = std::count(std::istreambuf_iterator<char>(fin), 
      std::istreambuf_iterator<char>(), '\n');
  fin.close();

  clog << "\t\33[32m[Done]\33[0m" << endl;
  return N;
}

std::istream& go_to_line(std::istream& file, unsigned long num){
  file.seekg(std::ios::beg);
  
  if (num == 0)
    return file;

  for(size_t i=0; i < num; ++i)
    file.ignore(std::numeric_limits<std::streamsize>::max(),'\n');

  return file;
}
