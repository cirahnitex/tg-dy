#include <iostream>
#include "dy.hpp"
#include "dy_training_framework.hpp"
#include <dynet/lstm.h>
#include <dynet/training.h>
#include <functional>
#include <word2vec.hpp>
#include <ECMAScript_string_utils.hpp>
#include <cereal/archives/binary.hpp>
#include <cereal/types/vector.hpp>
#include <sstream>

using namespace tg;
using namespace std;
using namespace util;

template<class T>
void print_helper(const T& x, std::ostream& os=std::cout) {
  for(const auto& t:x) {
    os << t << " ";
  }
  os <<endl;
}


int main() {
  dy::initialize();
  std::unordered_set<string> vocab;
  vocab.insert("turn");
  vocab.insert("left");
  vocab.insert("right");
  dy::readout_layer ro(vocab);
  for(unsigned i=0;i<10; i++) {
    cout << ro.random_readout(dy::zeros({3})) <<endl;
  }


  return 0;
}
