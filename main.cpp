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

  auto loss1 = my_model.compute_loss(true, true, true);
  auto loss2 = my_model.compute_loss(true, false, true);

  cout << dyana::_cg().nodes.size() <<endl;
  cout << dyana::_cg().parameter_nodes.size() <<endl;


  return 0;
}
