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

struct simple_timer {
  std::clock_t start;
  simple_timer(): start(std::clock()){}
  ~simple_timer() {
    std::cout << (std::clock() - start) / (CLOCKS_PER_SEC / 1000) << "ms" <<std::endl;
  }
};


int main() {
  dyana::initialize();


  return 0;
}
