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

  string GIVE = "give";
  string ME = "me";

  dyana::embedding_lookup lookup_table(4, initializer_list<string>{GIVE, ME});

  auto result = lookup_table(GIVE).as_vector();

  for(auto&& v:result) {
    cout << v << " ";
  }
  cout << endl;

  return 0;
}
