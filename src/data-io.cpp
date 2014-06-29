#include <data-io.h>
#include <stdexcept>
#include <algorithm>
#include <cassert>

#define RED_ERROR "\33[31m[Error]\33[0m "
/* \brief constructor of class DataStream
 *
 * */
DataStream::DataStream(): _nLines(0), _line_number(0), _remained(0), _is_pipe(false), _start(0), _end(-1) {
}

DataStream::DataStream(const string& filename, size_t start, size_t end) : _nLines(0), _remained(0) {
  this->init(filename, start, end);
}

DataStream::DataStream(const DataStream& src) : _nLines(src._nLines),
    _line_number(src._line_number), _filename(src._filename),
    _is_pipe(src._is_pipe), _start(src._start), _end(src._end) {
  this->init(_filename, _start, _end);
}

DataStream::~DataStream() {
  _fs.close();
}

DataStream& DataStream::operator = (DataStream that) {
  swap(*this, that);
  return *this;
}

void DataStream::init(const string& filename, size_t start, size_t end) {

  _filename = filename;
  _start = start;
  _end = end;
  _line_number = _start;

  // Find if filename contains "ark:" as prefix. if yes, we'll read it from pipe.
  _is_pipe = (filename.size() > 4 && filename.substr(0, 4) == "ark:");

  if (_is_pipe) {
    assert(RED_ERROR "read from pipe is NOT ready yet!");
    // Read from pipe
    size_t pos = filename.find_first_of(",");
    
    if (pos == string::npos)
      throw runtime_error(RED_ERROR "Please specify feature and label like ark:feat.ark,label.ark");

    _feat_command = filename.substr(4, pos - 4);
    _label_command = filename.substr(pos + 1);

    printf("Reading feature from \33[33m\"%s\"\33[0m\n", _feat_command.c_str());
    printf("Reading label from \33[33m\"%s\"\33[0m\n", _label_command.c_str());

    // Use wc to count # of features
    string wc_count_lines = _feat_command + "| feat-to-len ark:- ark,t:- | cut -f 2 -d ' '";
    FILE* fid = popen(wc_count_lines.c_str(), "r");

    _nLines = 0;
    int x;
    while (fscanf(fid, "%d", &x) == 1) _nLines += x;
    pclose(fid);

    fclose(stderr);

    // FIXME _end is useless when the input is from pipe
    _end = -999;

    _feat_ps = popen(_feat_command.c_str(), "r");
    _label_ps = popen(_label_command.c_str(), "r");
  }
  else {
    // Read from normal file
    if (_fs.is_open())
      _fs.close();

    _fs.open(_filename.c_str());

    if (!_fs.is_open())
      throw std::runtime_error("\33[31m[Error]\33[0m Cannot load file: " + filename);

    if (_nLines == 0)
      _nLines = count_lines(_filename);

    go_to_line(_fs, _start);

    _end = min(_nLines, _end);
    _nLines = min(_nLines, _end - _start);
  }
}

string DataStream::getline() {
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

void DataStream::rewind() {
  if (_is_pipe) {
    pclose(_feat_ps);
    _feat_ps = popen(_feat_command.c_str(), "r");

    pclose(_label_ps);
    _label_ps = popen(_label_command.c_str(), "r");
  }
  else {
    _fs.clear();
    go_to_line(_fs, _start);
    _line_number = _start;
  }
}

string DataStream::get_filename() const {
  return _filename;
}

size_t DataStream::get_line_number() const {
  return _nLines;
}

void swap(DataStream& a, DataStream& b) { 
  std::swap(a._nLines, b._nLines);
  std::swap(a._line_number, b._line_number);
  std::swap(a._filename, b._filename);
  // std::swap(a._fs, b._fs);
  // std::swap(a._feat_ps, b._feat_ps);
  // std::swap(a._label_ps, b._label_ps);
  std::swap(a._feat_command, b._feat_command);
  std::swap(a._label_command, b._label_command);
  std::swap(a._is_pipe, b._is_pipe);
  std::swap(a._start, b._start);
  std::swap(a._end, b._end);
}

/* 
 * Other utilities
 *
 * */
size_t count_lines(const string& fn) {
  printf("Loading file: \33[32m%s\33[0m (try to find out how many data) ...", fn.c_str());
  fflush(stdout);
  
  std::ifstream fin(fn.c_str()); 
  size_t N = std::count(std::istreambuf_iterator<char>(fin), 
      std::istreambuf_iterator<char>(), '\n');
  fin.close();

  printf("\t\33[32m[Done]\33[0m\n");
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
