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

  std::stringstream ss; // any stream can be used

  auto p = dy::add_parameters({1});
  cout << (bool)p.p <<endl;


  {
    dy::linear_layer fc(1);

    vector<float> input({4,5});
    auto output = dy::as_scalar(fc.forward(dy::const_expr(input)));
    cout << output <<endl;
    cereal::BinaryOutputArchive(ss).operator()(fc); // Write the data to the archive

  } // archive goes out of scope, ensuring all contents are flushed

  {

    dy::linear_layer fc;
    cereal::BinaryInputArchive(ss).operator()(fc); // Read the data from the archive

    vector<float> input({4,5});
    auto output = dy::as_scalar(fc.forward(dy::const_expr(input)));
    cout << output <<endl;

  }

  return 0;
}
