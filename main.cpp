#include <iostream>
#include "dyana.hpp"
#include <sstream>
#include <vector>
#include "dyana_timer.hpp"

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

vector<bool> input0s{true, true, false, false, true, true, false, false};
vector<bool> input1s{true, false, true, false, true, false, true, false};
vector<bool> oracles{true, true, true, false, true, true, true, false};


int main() {

  // todo: test out new trainer
  dyana::initialize(512);

  xor_model model;

  using datum_t = tuple<bool, bool, bool>;

  dyana::simple_sgd_trainer trainer(0.1);
  trainer.set_num_epochs(10);
  trainer.set_learning_rate_scheduler([&](unsigned datum_idx, unsigned epoch_idx) {
    cout << "datum idx: "<< datum_idx << endl;
    cout << "epoch idx: "<< epoch_idx << endl;
    return 0.1;
  });
  trainer.set_batch_size(4);
  trainer.train<datum_t>([&](const datum_t& datum) {
    auto&& [x, y, oracle] = datum;
    return model.compute_loss(x, y, oracle);
  }, dyana::zip(input0s, input1s, oracles));

  dyana::parallel_map<bool>(input0s, [&](const bool& x) {
    return dyana::zeros({4}).as_vector();
  }, 4, 2);

  return 0;
}
