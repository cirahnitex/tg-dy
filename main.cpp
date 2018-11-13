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
using namespace dynet;
using namespace util;


/**
 * normalized a vector
 * \param x vector to normalize
 * \return the normalized vector
 */
dynet::Expression vector_norm(const dynet::Expression &x) {
  auto length = dynet::l2_norm(x);
  static const auto epsilon = dy::const_expr(0.0001);
  return dynet::cdiv(x, length + epsilon);
}

int main() {
  dy::initialize();

  const auto big_num = dy::const_expr({-110,-100,0,0,0,0,0,90,100});

  const auto sigmoid = dy::logistic(big_num);
  for(const auto& num:dy::as_vector(sigmoid)) {
    cout << num << " ";
  }
  cout <<endl;

  return 0;
}
