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

  dynet::Expression forward(bool x, bool y) {
    auto input = dy::const_expr({x?1.0:0.0, y?1.0:0.0});
    return fc2.forward(dynet::tanh(fc1.forward(input)));
  }

  bool predict(bool x, bool y) {
    return dy::as_scalar(forward(x, y)) > 0.0;
  }

  dynet::Expression compute_loss(bool x, bool y, bool oracle) {
    auto oracle_expr = dy::const_expr(oracle?1.0:0.0);
    return dynet::binary_log_loss(dynet::logistic(forward(x, y)), oracle_expr);
  }

  EASY_SERIALZABLE(fc1, fc2)
private:
  dy::linear_layer fc1;
  dy::linear_layer fc2;
};

int main() {
  dy::initialize();

  typedef vector<bool> datum_type;

  cout << "reading training data" << endl;

  // the xor training dataset
  vector<datum_type> data({{false, false, false}, {false, true, true}, {true, false, true}, {true, true, false}});

  // duplicate the data 10000 times to make a BIG dataset
  vector<datum_type> big_data;
  for(unsigned epoch=0; epoch<10000; epoch++) {
    std::copy(data.begin(), data.end(), std::back_inserter(big_data));
  }

  cout << "parallel training" <<endl;
  XorModel model;
  const unsigned NUM_WORKERS = 4;
  dy::mp_train<datum_type>(NUM_WORKERS, big_data, [&](const datum_type& datum){
    return model.compute_loss(datum[0], datum[1], datum[2]);
  });

  cout << "predicting" <<endl;
  for(const auto& datum:data) {
    dy::renew_cg();
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
    dy::renew_cg();
    cout << datum[0] << " " << datum[1] << " => "<< new_model.predict(datum[0], datum[1]) <<endl;
  }
}
