#include <iostream>
#include "dyana.hpp"

using namespace std;

class xor_model {
  dyana::linear_dense_layer dense0;
  dyana::linear_dense_layer dense1;
public:
  EASY_SERIALIZABLE(dense0, dense1)

  xor_model():dense0(2), dense1(1) {}
  xor_model(const xor_model&) = default;
  xor_model(xor_model&&) noexcept = default;
  xor_model &operator=(const xor_model&) = default;
  xor_model &operator=(xor_model&&) noexcept = default;

  dyana::tensor operator()(const dyana::tensor &x, const dyana::tensor &y) {
    auto t = dyana::concatenate({x, y});
    t = dyana::logistic(dense0(t));
    return dyana::logistic(dense1(t));
  }

  bool operator()(bool x, bool y) {
    dyana::tensor numeric_result = operator()((dyana::tensor)x, (dyana::tensor)y);
    return numeric_result.as_scalar() > 0.5;
  }

  dyana::tensor compute_loss(bool x, bool y, bool oracle) {
    dyana::tensor numeric_result = operator()((dyana::tensor)x, (dyana::tensor)y);
    return dyana::binary_log_loss(numeric_result, (dyana::tensor)oracle);
  }
};

int main() {
  dyana::initialize();

  vector<bool> input0s{true, true, false, false};
  vector<bool> input1s{true, false, true, false};
  vector<bool> oracles{false, true, true, false};

  xor_model my_model;

  dyana::adam_trainer trainer(0.1);
  trainer.num_epochs = 100;
  trainer.num_workers = 4;
  trainer.train(my_model, input0s, input1s, oracles);


  cout << "input0" << "," << "input1" << "," << "output" <<endl;
  for(const auto &[input0, input1]:dyana::zip(input0s, input1s)) {
    cout << input0 << "," << input1 << "," << my_model(input0, input1) <<endl;
  }

  return 0;
}
