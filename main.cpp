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
  dyana::initialize(512);

  vector<string> choices = {"a", "b", "c"};

  dynet::LookupParameter lookup = dyana::_pc()->add_lookup_parameters(4, {8});

  cout << "make CG" << endl;
  auto emb = dynet::tanh(dynet::lookup(dyana::_cg(), lookup, vector<unsigned>{1}));

  cout << "compute" << endl;
  dyana::_cg().forward(emb);

  int* my_int = new int;
  delete my_int;
  return 0;
}
