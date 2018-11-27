//
// Created by YAN Yuchen on 11/8/2018.
//

#include "../../dy.hpp"
#include <vector>
using namespace tg;
using namespace std;
class XorModel {
public:
  XorModel(const XorModel&) = default;
  XorModel(XorModel&&) = default;
  XorModel &operator=(const XorModel&) = default;
  XorModel &operator=(XorModel&&) = default;
  XorModel():fc1(4),fc2(1) {
  }

  dy::tensor forward(bool x, bool y) {
    auto input = dy::tensor({x?1.0:0.0, y?1.0:0.0});
    return fc2.predict(dy::tanh(fc1.predict(input)));
  }

  bool predict(bool x, bool y) {
    return forward(x, y).as_scalar() > 0.0;
  }

  dy::tensor compute_loss(bool x, bool y, bool oracle) {
    auto oracle_expr = dy::tensor(oracle?1.0:0.0);
    return dy::binary_log_loss(dy::logistic(forward(x, y)), oracle_expr);
  }

  EASY_SERIALZABLE(fc1, fc2)
private:
  dy::linear_layer fc1;
  dy::linear_layer fc2;
};

int main() {
  dy::initialize(1, dy::trainer_type::SIMPLE_SGD, 1);

  typedef vector<bool> datum_type;

  cout << "reading training data" << endl;

  // the xor training dataset
  vector<datum_type> data({{false, false, false}, {false, true, true}, {true, false, true}, {true, true, false}});

  cout << "training" <<endl;
  XorModel model;
  dy::fit<datum_type>(1000, data, [&](const datum_type& datum){
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
