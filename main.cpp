#include <iostream>
#include "dyana.hpp"
#include <xml_archive.hpp>
#include <sstream>
#include <vector>
#include "dyana_timer.hpp"
#include "dyana_parallel_map.hpp"
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

  // create some data
  vector<float> xs{1,2,3,4,5};

  // create a square fn
  auto square_returning_vector = [](float x) {
    return vector<float>{x*x};
  };

  // call parallel_map
  auto ys = dyana::parallel_map<float>(xs, square_returning_vector, 1, 2);

  // print results
  for(auto&& y:ys) {
    cout << y[0] << endl;
  }

  return 0;
}
