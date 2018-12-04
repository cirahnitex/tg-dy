#include <iostream>
#include "dy.hpp"
#include <chrono>
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
//  auto x = bilr.create_lookup_with_loss_computer({"a","b","c"}, {"A","B","C"});
  typedef pair<string, string> datum;
  vector<datum> data({
    make_pair("a","A"),
    make_pair("b","B"),
    make_pair("c","C")
  });
  auto start = std::chrono::steady_clock::now();
  dy::fit<datum>(1000, data, [&](const datum& datum){
    auto loss_computer = bilr.create_lookup_with_loss_computer({"a", "b", "c"}, {"A", "B", "C"});
    return loss_computer(datum.first, datum.second).second;
  });
  auto duration = std::chrono::duration_cast<std::chrono::milliseconds>
    (std::chrono::steady_clock::now() - start);
  cout << duration.count()<<endl;

  for(const string& l0:l0_vocab) {
    cout << "l0:"<<l0<<" translation:"<<bilr.translate_l0_to_l1_slow(l0,1).front().first<<endl;
  }

  return 0;
}
