#include <iostream>
#include "dy.hpp"

using namespace tg;
using namespace std;


int main() {
  dy::initialize();

  dy::tensor W({1, 2, 3, 4, 5, 6}, {3, 2});
  dy::tensor x({-1, 0});
  auto c = W*x;
  for(const auto val:c.as_vector()) {
    cout << val <<endl;
  }
  return 0;
}
