#include <iostream>
#include "dy.hpp"

using namespace tg;
using namespace std;

dy::tensor max1d(const dy::tensor& x) {
  if(x.dim().nd != 1) throw runtime_error("max1d only works with 1D-tensor");
  vector<float> values = x.as_vector();
  auto max_iter = std::max_element(values.begin(), values.end());
  unsigned index_of_max_element = std::distance(max_iter, values.begin());
  return dy::pick(x, index_of_max_element);
}


int main() {
  dy::initialize();

  // define an input
  dy::tensor x({1,2});

  auto y = max1d(x);

  // get the output as std::vector
  std::vector<float> values = y.as_vector();

  // outputs 1 number
  for(auto val:values) { cout << val <<" "; }
  cout << endl;

  return 0;
}
