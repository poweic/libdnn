#include <feature-transform.h>
#include <config.h>
#include <utility.h>
#include <dnn-utility.h>

class NNet {
public:

  NNet();
  NNet(const string& model_fn);
  ~NNet();

  void init(const string &structure);

  mat feedForward(const mat& fin) const;
  void feedForward(mat& fout, const mat& fin);
  void backPropagate(mat& error, const mat& fin, const mat& fout,
      float learning_rate);

  void feedBackward(mat& error, const mat& delta);

  void read(const string &fn);
  void save(const string &fn) const;

  virtual size_t getInputDimension() const;
  virtual size_t getOutputDimension() const;

  std::vector<FeatureTransform*>& getTransforms() { return _transforms; }

  void status() const;

  bool is_cnn_dnn_boundary(size_t i) const;

  void setDropout(bool flag);
  void setConfig(const Config& config);
  Config getConfig() const;

  friend ostream& operator << (ostream& os, const NNet& nnet);

private:

  void weight_initialize();

  std::vector<FeatureTransform*> _transforms;
  std::vector<mat > _houts;

  /* Hidden Outputs: outputs of each hidden layers
   * The first element in the std::vector (i.e. _houts[0])
   * is the output of first hidden layer. 
   * ( Note: this means no input data will be kept in _houts. )
   * ( Also, no output data will be kept in _houts. )
   * */
  Config _config;
};

ostream& operator << (ostream& os, const NNet& nnet);
