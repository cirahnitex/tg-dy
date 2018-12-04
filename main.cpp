#include <iostream>
#include "dy.hpp"

using namespace tg;
using namespace std;


int main() {
  dy::initialize(1, dy::trainer_type::SIMPLE_SGD);

  unordered_set<string> l0_vocab = {
    "a", "b", "c", "d", "e", "f", "g","h","i","j","k","l","m","n"
  };
  unordered_set<string> l1_vocab = {
    "A", "B", "C", "D", "E", "F", "G","H","I","J","K","L","M","N"
  };
  dy::bi_lookup_readout bilr(3, l0_vocab, l1_vocab);

  typedef pair<string, string> datum;
  vector<datum> data({
    make_pair("a","A"),
    make_pair("b","B"),
    make_pair("c","C")
  });
  dy::fit<datum>(1000, data, [&](const datum& datum){
    auto loss_computer = bilr.create_readout_with_loss_computer({"a","b","c"}, {"A","B","C"});
    return loss_computer(datum.first, datum.second).second;
  });

  return 0;
}
