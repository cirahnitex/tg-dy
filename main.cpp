#include <iostream>
#include "dyana.hpp"
#include <xml_archive.hpp>
#include <sstream>
#include <vector>
#include <thread>
using namespace std;

class xor_model {
  dyana::binary_readout_model bo;
public:
  bool operator()(bool x, bool y) {
    auto combined = dyana::concatenate({dyana::tensor(x), dyana::tensor(y)});
    return bo(combined);
  }
  dyana::tensor compute_loss(bool x, bool y, bool oracle) {

    auto combined = dyana::concatenate({dyana::tensor(x), dyana::tensor(y)});

    auto ret = bo.compute_loss(combined, oracle);

    return ret;
  }
};

vector<bool> input0s{true, true, false, false};
vector<bool> input1s{true, false, true, false};
vector<bool> oracles{true, true, true, false};


int main() {
  dyana::initialize();

  xor_model my_model;

  dyana::simple_sgd_trainer trainer(0.01);
  trainer.num_epochs = 10;
  trainer.num_workers = 1;
  trainer.train_reporting_dev_score(my_model, input0s, input1s, oracles, input0s, input1s, oracles);

  cout << "input0" << "," << "input1" << "," << "output" <<endl;
  for(auto&& t:dyana::zip(input0s, input1s)) {
    cout << std::get<0>(t) << "," << std::get<1>(t) << "," << my_model(std::get<0>(t), std::get<1>(t)) <<endl;
  }

  return 0;
}
