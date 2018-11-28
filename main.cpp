#include <iostream>
#include "dy.hpp"

using namespace tg;
using namespace std;


int main() {
  dy::initialize();

  dy::tensor arr({0,1,2,3});
  auto picked = dy::pick_range(arr, 1, 3).as_vector();
  for(const auto& val:picked) {
    cout << val <<endl;
  }
  return 0;
}
