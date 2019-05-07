#include <iostream>
#include "dyana.hpp"

using namespace std;

class xor_model {
  dyana::linear_dense_layer fc_0;
  dyana::linear_dense_layer fc_1;
public:
  EASY_SERIALIZABLE(fc_0, fc_1)

  xor_model(const xor_model&) = default;
  xor_model(xor_model&&) noexcept = default;
  xor_model &operator=(const xor_model&) = default;
  xor_model &operator=(xor_model&&) noexcept = default;

  xor_model():fc_0(4), fc_1(1) {}

  dyana::tensor operator()(const dyana::tensor &x, const dyana::tensor &y) {
    auto t = dyana::concatenate({x, y});
    t = dyana::tanh(fc_0(t));
    return dyana::logistic(fc_1(t));
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

  vector<bool> x0s{true, true, false, false};
  vector<bool> x1s{true, false, true, false};
  vector<bool> oracles{false, true, true, false};

  xor_model my_model;

  dyana::adam_trainer trainer(0.1);
  trainer.num_workers = 10;
  trainer.num_epochs = 100;
  trainer.train(my_model, x0s, x1s, oracles);

  for(const auto &[x0, x1]:dyana::zip(x0s, x1s)) {
    cout << my_model(x0, x1) <<endl;
  }

  return 0;
}
