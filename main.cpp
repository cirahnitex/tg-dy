#include <iostream>
#include "dy.hpp"

using namespace tg;
using namespace std;


int main() {
  dy::initialize();

  dy::tensor arr({0,1,2,3});
  auto picked = arr.select_rows({1,3});
  cout << picked.dim() << endl;
  for(const auto& val:picked.as_vector()) {
    cout << val <<endl;
  }
  return 0;
}
