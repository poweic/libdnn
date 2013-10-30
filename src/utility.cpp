#include <utility.h>

string int2str(int n) {
  char buf[32];
  sprintf(buf, "%d", n);
  return string(buf);
}

double str2double(const string& str) {
  return (double) atof(str.c_str());
}

float str2float(const string& str) {
  return atof(str.c_str());
}

int str2int(const string& str) {
  return atoi(str.c_str());
}

vector<string>& split(const string &s, char delim, vector<string>& elems) {
  stringstream ss(s);
  string item;
  while(getline(ss, item, delim))
    elems.push_back(item);
  return elems;
}

vector<string> split(const string &s, char delim) {
  vector<string> elems;
  return split(s, delim, elems);
}

vector<size_t> randperm(size_t N) {
  vector<size_t> perm(N);
  string result = exec("seq 0 " + int2str(N) + " | shuf");

  vector<string> numbersInStr = split(result, '\n');

  foreach (i, perm)
    perm[i] = (size_t) str2int(numbersInStr[i]);

  return perm;
}

string replace_all(const string& str, const string &token, const string &s) {
  string result(str);
  size_t pos = 0;
  while((pos = result.find(token, pos)) != string::npos) {
    result.replace(pos, token.size(), s);
    pos += s.size();
  }
  return result;
}

bool isInt(string str) {
  int n = str2int(str);
  string s = int2str(n);

  return (s == str);
}

string join(const vector<string>& arr, string token) {
  string str;
  for (size_t i=0; i<arr.size() - 1; ++i)
      str += arr[i] + token;
  str += arr[arr.size() - 1];
  return str;
}

std::string exec(std::string cmd) {
  FILE* pipe = popen(cmd.c_str(), "r");
  if (!pipe)
    return "ERROR";

  char buffer[128];
  std::string result = "";

  try {
    while(!feof(pipe)) {
      if(fgets(buffer, 128, pipe) != NULL)
	result += buffer;
    }
  } catch (...) {
    std::cerr << "[Warning] Exception caught in " << __FUNCTION__ << endl;
  }

  pclose(pipe);
  return result;
}

void doPause() {
  cout << "Press "BLUE"[ Enter ]"COLOREND" to continue...";
  cin.clear(); 
  cin.ignore();
}

namespace bash {
  vector<string> ls(string path) {
    return split(exec("ls " + path), '\n');
  }
}
