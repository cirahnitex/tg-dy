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
  auto arr1 = dy::const_expr({1,2,3});
  auto arr2 = dy::const_expr({3,2,1});
  auto arr3 = dy::const_expr({1,3,2});
  print_helper(dy::as_vector(dy::max({arr1, arr2,arr3})));

  return 0;
}
