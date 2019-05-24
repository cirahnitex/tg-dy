#include <iostream>
#include <vector>
#include <dyana.hpp>
#include <stdexcept>
#include <algorithm>


class xor_model {
  dyana::linear_dense_layer dense0;
  dyana::linear_dense_layer dense1;
public:
  EASY_SERIALIZABLE(dense0, dense1);

  xor_model():dense0(2), dense1(1) {}
  // ouput of dim layer dense0 is called hiddend units. to solve XOR, at least need 2.
  xor_model(const xor_model&) = default;
  xor_model(xor_model&&) noexcept = default;
  xor_model &operator = (const xor_model&) = default;
  xor_model &operator = (xor_model&&) = default;

  // a transduce method to stacking two sigmoid dense layers historically named as logistic
  // S-function is a kind of logistc function = 1/(1+exp[-x])

  dyana::tensor operator()(const dyana::tensor &x, const dyana::tensor &y) {
    auto t = dyana::concatenate({x,y});
    t = dyana::logistic(dense0(t));
    return dyana::logistic(dense1(t));
  }


  // wraper of the tensor function
  bool operator()(bool x, bool y) {
    dyana::tensor numeric_result = operator()((dyana::tensor)x, (dyana::tensor)y);
    //std::cout << numeric_result;
    return numeric_result.as_scalar()>0.5;
  }

  // binary log loss
  // y ln(y hat) + (1 - y) ln(1 - y hat)
  dyana::tensor compute_loss(bool x, bool y, bool oracle) {
    dyana::tensor numeric_result = operator()((dyana::tensor)x, (dyana::tensor)y);
    return dyana::binary_log_loss(numeric_result, (dyana::tensor)oracle);
  }

};


int main(int argc, char** argv) {


  dyana::initialize();

  // training
  // test
  std::vector<bool> input0s{true, true, false, false};
  std::vector<bool> input1s{true, false, true, false};
  std::vector<bool> oracles{false, true, true, false};

  xor_model test;

  dyana::adam_trainer trainer(0.1); // train rate
  trainer.num_epochs = 100;
  trainer.num_workers = 4;
  trainer.train(test, input0s, input1s, oracles);
  /*
  std::cout << "input0" << "," << "input1" << "," << "output" << std::endl;
  for(const auto &[input0, input1]:dyana::zip(input0s, input1s)) {
      using namespace std;
      std::cout << input0 << "," << input1 << "," << my_model(input0, input1) << std::endl;
  }
  */

  return 0;

}


