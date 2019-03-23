//
// Created by YAN Yuchen on 11/8/2018.
//

#include "../../dyana.hpp"
#include <vector>
using namespace tg;
using namespace std;
class XorModel {
  dyana::linear_layer fc1;
  dyana::linear_layer fc2;
public:
  EASY_SERIALIZABLE(fc1, fc2)
  XorModel(const XorModel&) = default;
  XorModel(XorModel&&) = default;
  XorModel &operator=(const XorModel&) = default;
  XorModel &operator=(XorModel&&) = default;
  XorModel():fc1(2),fc2(1) {
  }

  dyana::tensor forward(bool x, bool y) {
    auto input = dyana::tensor({x?(float)1:(float)0, y?(float)1:(float)0});
    return fc2.predict(dyana::tanh(fc1.predict(input)));
  }

  bool predict(bool x, bool y) {
    return forward(x, y).as_scalar() > 0.0;
  }

  dyana::tensor compute_loss(bool x, bool y, bool oracle) {
    auto oracle_expr = dyana::tensor(oracle?(float)1:(float)0);
    return dyana::binary_log_loss(dyana::logistic(forward(x, y)), oracle_expr);
  }
};

int main() {
  dyana::initialize(1, dyana::trainer_type::SIMPLE_SGD, 0.1);

  typedef vector<bool> datum_type;

  cout << "reading training data" << endl;

  // the xor training dataset
  vector<datum_type> data({{false, false, false}, {false, true, true}, {true, false, true}, {true, true, false}});

  cout << "training" <<endl;
  XorModel model;
  dyana::fit<datum_type>(500, data, [&](const datum_type& datum){
    return model.compute_loss(datum[0], datum[1], datum[2]);
  });

  cout << "predicting" <<endl;
  for(const auto& datum:data) {
    cout << datum[0] << " " << datum[1] << " => "<< model.predict(datum[0], datum[1]) <<endl;
  }

  cout << "saving model" <<endl;
  stringstream ss;
  cereal::BinaryOutputArchive(ss) << model;

  cout << "loading model" <<endl;
  XorModel new_model;
  cereal::BinaryInputArchive(ss) >> new_model;

  cout << "predicting using loaded model" <<endl;
  for(const auto& datum:data) {
    cout << datum[0] << " " << datum[1] << " => "<< new_model.predict(datum[0], datum[1]) <<endl;
  }
}
