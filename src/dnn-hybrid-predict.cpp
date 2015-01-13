#include <iostream>
#include <string>
#include <dnn.h>
#include <cmdparser.h>
#include <batch.h>
using namespace std;

void printLabels(const mat& prob, FILE* fid, int base);
FILE* openFileOrStdout(const string& filename);
vector<size_t> getS2gMapping(string s2g_fn);
mat getGroupIdUsingHiddenOutputs(const DNN& dnn, const mat& fin, const mat& centroid);
mat loadCentroidInCSV(string filename);
vector<FeatureTransform*> loadFdlrTransforms(string fdlrs_fn);

hmat senonePosteriorToGroupPosterior(const hmat& sPost, const vector<size_t>& s2g, size_t N);
size_t getGroupNumber(const vector<size_t>& s2g);
mat loadLogPriorProbability(string fn);

mat computeEnhancedPosterior(DNN& dnn, const vector<FeatureTransform*>& fts,
    const mat& score, const BatchData& data, const mat& gPost, const float ratio);
vector<mat> computePosteriorForEachGroup(DNN& dnn, const vector<FeatureTransform*>& fts, const BatchData& data);

int main (int argc, char* argv[]) {

  CmdParser cmd(argc, argv);
  cmd.printArgs();

  cmd.add("testing_set_file")
    .add("model_file")
    .add("output_file", false);

  cmd.addGroup("Feature options:")
     .add("--input-dim", "specify the input dimension (dimension of feature).\n"
	 "0 for auto detection.")
     .add("--normalize", "Feature normalization: \n"
	"0 -- Do not normalize.\n"
	"1 -- Rescale each dimension to [0, 1] respectively.\n"
	"2 -- Normalize to standard score. z = (x-u)/sigma .", "0")
     .add("--prior", "prior probability for each classes", "")
     .add("--nf", "Load pre-computed statistics from file", "")
     .add("--fdlrs", "fDLR affine transform matrices. Ex: exp/#.fdlr")
     .add("--s2g", "file containing senone to group id mapping")
     .add("--ratio", "score' = r * score + (1-r) * sum(others)", "1")
     .add("--base", "Label id starts from 0 or 1 ?", "0");

  cmd.addGroup("Options:")
    .add("--acc", "calculate prediction accuracy", "true")
    .add("--prob", "output posterior probabilities if true\n"
	"0 -- Do not output posterior probabilities. Output class-id.\n"
	"1 -- Output posterior probabilities. (range in [0, 1]) \n"
	"2 -- Output natural log of posterior probabilities. (range in [-inf, 0])", "0")
    .add("--dump", "dump hidden outputs of layer #n. Ex: --dump 0 will dump the"
	"output of 1st hidden layer. --dump 0 will dump the original input", "-1")
    .add("--silent", "Suppress all log messages", "false");

  cmd.addGroup("Hardward options:")
     .add("--cache", "specify cache size (in MB) in GPU used by cuda matrix.", "16");

  cmd.addGroup("Example usage: dnn-predict test3.dat train3.dat.model");

  if (!cmd.isOptionLegal())
    cmd.showUsageAndExit();

  string test_fn    = cmd[1];
  string model_fn   = cmd[2];
  string output_fn  = cmd[3];

  size_t input_dim  = cmd["--input-dim"];
  NormType n_type   = (NormType) (int) cmd["--normalize"];
  string n_filename = cmd["--nf"];
  string prior_fn   = cmd["--prior"];
  // string centroid_fn= cmd["--centroid"];
  string s2g_fn     = cmd["--s2g"];
  float ratio	    = cmd["--ratio"];
  string fdlrs_fn   = cmd["--fdlrs"];
  int base	    = cmd["--base"];

  int output_type   = cmd["--prob"];
  int dump	    = cmd["--dump"];
  bool silent	    = cmd["--silent"];
  bool calcAcc	    = cmd["--acc"];

  size_t cache_size   = cmd["--cache"];
  CudaMemManager<float>::setCacheSize(cache_size);

  // TODO
  auto s2g = getS2gMapping(s2g_fn);
  auto fts = loadFdlrTransforms(fdlrs_fn);

  mat log_prior = loadLogPriorProbability(prior_fn);
  clog << YELLOW_WARNING << "Load prior prob from: \33[32m" << prior_fn << "\33[0m" << endl;

  if (!prior_fn.empty() && output_type == 1)
    clog << YELLOW_WARNING << "Prior probabilities is not inplemented with --prob 1 ('cause I'm lazy)" << endl;

  size_t nGroups = getGroupNumber(s2g);
  if (nGroups != fts.size())
    clog << YELLOW_WARNING << "# of groups (" << nGroups << ") != # of transform matrix (" << fts.size() << ")" << endl;

  DataSet test(test_fn, input_dim, base);
  test.setNormType(n_type, n_filename);

  DNN dnn(model_fn);

  size_t nError = 0;

  FILE* fid = openFileOrStdout(output_fn);

  mat log_priors;

  Batches batches(1024, test.size());
  for (Batches::iterator itr = batches.begin(); itr != batches.end(); ++itr) {
    auto data = test[itr];
    mat prob = dnn.feedForward(data.x, dump);

    // TODO
    // (1) change senonePosteriorToGroupPosterior to matrix multiplication:
    //	   (1024 x 3445) x (3445 x 32) = 1024 x 32
    if (ratio != 1) {
      auto gPost = (mat) senonePosteriorToGroupPosterior(prob, s2g, nGroups);
      prob = computeEnhancedPosterior(dnn, fts, prob, data, gPost, ratio);
    }
    // =====================================================

    if (calcAcc && !silent)
      nError += zeroOneError(prob, data.y, CROSS_ENTROPY);

    if (calcAcc && output_fn.empty() && output_type == 0)
      continue;

    switch (output_type) {
      case 0:
	printLabels(prob, fid, base);
	break;

      case 1:
	prob.print(fid, 7);
	break;

      case 2: 
	if (log_priors.getRows() != prob.getRows())
	  log_priors = mat(prob.getRows(), 1, 1) * log_prior;

	if (!prior_fn.empty())
	  (log(prob) - log_priors).print(fid, 7);
	else
	  log(prob).print(fid, 7);
	break;
    }
  }

  if (fid != stdout)
    fclose(fid);

  if (calcAcc && !silent)
    showAccuracy(nError, test.size());

  return 0;
}

mat loadLogPriorProbability(string fn) {
  if (fn.empty())
    return mat();

  mat prior(fn);
  double sum = ((hmat) (prior * mat(prior.getCols(), 1, 1)))[0];
  return log(prior / sum);
}

vector<FeatureTransform*> loadFdlrTransforms(string fdlrs_fn) {

  vector<FeatureTransform*> fts;

  for (size_t i=0; i<fdlrs_fn.size(); ++i)
    if (fdlrs_fn[i] == '#')
      fdlrs_fn[i] = '*';

  auto filenames = split(exec("ls -1v " + fdlrs_fn), '\n');

  FILE* fid;
  for (size_t i=0; i<filenames.size(); ++i) {
    fid = fopen(filenames[i].c_str(), "r");

    if (!fid)
      throw std::runtime_error(RED_ERROR + "Cannot load file " + filenames[i]);

    // fprintf(stderr, "Loading fDLR matrix: \33[32m%s\33[0m\n", filenames[i].c_str());
    fts.push_back(FeatureTransform::create(fid));

    fclose(fid);
  }

  return fts;
}

vector<size_t> getS2gMapping(string s2g_fn) {
  vector<size_t> s2g;

  ifstream fin(s2g_fn.c_str());

  size_t g;
  while (fin >> g)
    s2g.push_back(g);

  fin.close();
  
  return s2g;
}

mat getGroupIdUsingHiddenOutputs(const DNN& dnn, const mat& fin, const mat& centroid) {

  mat y;

  const auto& t = dnn.getTransforms();

  t[0]->feedForward(y, fin);

  // t.size() - 1 means last hidden outputs
  for (size_t i=1; i<t.size() - 2; ++i)
    t[i]->feedForward(y, y);

  y.resize(y.getRows(), y.getCols() - 1);

  mat p = pdist(y, centroid);
  p *= -1;  // Find minimum distance rather maximum

  return posteriorProb2Label(p);
}

mat loadCentroidInCSV(string filename) {
  string tmp = ".tmp.csv-to-ssv";
  string cmd = "cat " + filename + " | tr ',' ' ' > " + tmp;
  exec(cmd);

  mat centroid(tmp);

  exec("rm " + tmp);
  return centroid;
}

// DO NOT USE device_matrix::print()
// (since labels should be printed as integer)
void printLabels(const mat& prob, FILE* fid, int base) {
  auto h_labels = copyToHost(posteriorProb2Label(prob));
  for (size_t i=0; i<h_labels.size(); ++i)
    fprintf(fid, "%d\n", (int) h_labels[i] + base);
}

FILE* openFileOrStdout(const string& filename) {

  FILE* fid = filename.empty() ? stdout : fopen(filename.c_str(), "w");

  if (fid == NULL) {
    fprintf(stderr, "Failed to open output file");
    exit(-1);
  }

  return fid;
}

size_t getGroupNumber(const vector<size_t>& s2g) {
  size_t N = 0;

  for (auto x : s2g)
    N = std::max(N, x);

  return N + 1;
}


hmat senonePosteriorToGroupPosterior(const hmat& sPost, const vector<size_t>& s2g, size_t N) {

  size_t rows = sPost.getRows(),
	 cols = sPost.getCols();

  hmat gPost(rows, N);

  for (size_t i=0; i<rows; ++i) {
    for (size_t j=0; j<cols; ++j)
      gPost(i, s2g[j]) += sPost(i, j);
  }

  return gPost;
}

vector<mat> computePosteriorForEachGroup(DNN& dnn, const vector<FeatureTransform*>& fts, const BatchData& data) {

  vector<mat> posts;

  auto& t = dnn.getTransforms();
  auto* original = t[0];
  for (size_t i=0; i<fts.size(); ++i) {
    t[0] = fts[i];
    posts.push_back(dnn.feedForward(data.x));
  }

  t[0] = original;

  return posts;
}

mat computeEnhancedPosterior(DNN& dnn, const vector<FeatureTransform*>& fts,
    const mat& score, const BatchData& data, const mat& gPost, const float ratio) { 

  size_t rows = score.getRows(),
	 cols = score.getCols();

  mat new_score(rows, cols, 0);

  mat coeff(rows, 1);

  auto& t = dnn.getTransforms();
  auto* original = t[0];
  for (size_t i=0; i<fts.size(); ++i) {
    memcpy2D(coeff, gPost, 0, i, rows, 1, 0, 0);

    t[0] = fts[i];
    new_score += row_multiply(dnn.feedForward(data.x), coeff);
  }
  t[0] = original;

  // Interpolate the original score with other scores
  new_score = ratio * score + ( 1 - ratio ) * new_score;

  return new_score;
}
