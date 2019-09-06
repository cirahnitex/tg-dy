#include <iostream>
#include "dyana.hpp"
#include <xml_archive.hpp>
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
  dyana::initialize();

  static dyana::parameter W({32,32});
  static dyana::parameter b({32});
  auto t = (W * dyana::zeros(dyana::Dim({32},1)) + b);
  dyana::_cg().print_graphviz();
  t.as_vector();
  return 0;
}
