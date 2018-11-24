#include <iostream>
#include "dy.hpp"
#include "dy_training_framework.hpp"

#include <functional>
#include <word2vec.hpp>
#include <ECMAScript_string_utils.hpp>
#include <sstream>

using namespace tg;
using namespace std;

template<class T>
void print_helper(const T& x, std::ostream& os=std::cout) {
  for(const auto& t:x) {
    os << t << " ";
  }
  os <<endl;
}

dy::linear_layer fc1;
dy::linear_layer fc2;

int main() {
  dy::initialize();

  dy::linear_layer fc1(4);
  dy::linear_layer fc2(1);
  dy::tensor x({1,0});
  x = fc1.forward(x);
  x = dy::tanh(x);
  x = fc2.forward(x);
  cout << (x.as_scalar() > 0)  <<endl;

  return 0;
}
