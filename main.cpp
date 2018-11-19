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
  auto z = dy::zeros({15});
  cout << dy::as_tensor(z) << endl;
  return 0;
}
