#include <iostream>
#include <vector>
#include <dyana.hpp>
#include <stdexcept>
#include <algorithm>

//using namespace std;
//using namespace dynet;

// do not use cpp math function since the training need those information

dyana::tensor normalize(const dyana::tensor& x);

dyana::tensor max1d(const dyana::tensor& x);

class tanh_dense_layer
{
  // tanh dense layer requires a linear dense layer under the hood
  dyana::linear_dense_layer fc_m;
public:
  // SL layer macro
  EASY_SERIALIZABLE(fc_m);

  tanh_dense_layer() = default; // default constructor
  tanh_dense_layer(const tanh_dense_layer&) = default; // copy constructor
  tanh_dense_layer(tanh_dense_layer&&) noexcept = default; // move constructor && gives rvalue

  tanh_dense_layer &operator = (const tanh_dense_layer&) = default;
  tanh_dense_layer &operator = (tanh_dense_layer&&) noexcept = default;

  explicit tanh_dense_layer(unsigned dim_out):fc_m(dim_out) {}

  explicit tanh_dense_layer(const dyana::linear_dense_layer& ori_layer): fc_m(ori_layer) {}

  dyana::tensor operator()(const dyana::tensor &x) {
    return dyana::tanh(fc_m(x));
  }

};


class xor_model {
  dyana::linear_dense_layer dense0;
  dyana::linear_dense_layer dense1;
public:
  EASY_SERIALIZABLE(dense0, dense1);

  xor_model():dense0(2), dense1(1) {}
  // ouput of dim layer dense0 is called hiddend units. to solve XOR, at least need 2.
  xor_model(const xor_model&) = default;
  xor_model(xor_model&&) noexcept = default;
  xor_model &operator = (const xor_model&) = default;
  xor_model &operator = (xor_model&&) = default;

  // a transduce method to stacking two sigmoid dense layers historically named as logistic
  // S-function is a kind of logistc function = 1/(1+exp[-x])

  dyana::tensor operator()(const dyana::tensor &x, const dyana::tensor &y) {
    auto t = dyana::concatenate({x,y});
    t = dyana::logistic(dense0(t));
    return dyana::logistic(dense1(t));
  }


  // wraper of the tensor function
  bool operator()(bool x, bool y) {
    dyana::tensor numeric_result = operator()((dyana::tensor)x, (dyana::tensor)y);
    //std::cout << numeric_result;
    return numeric_result.as_scalar()>0.5;
  }

  // binary log loss
  // y ln(y hat) + (1 - y) ln(1 - y hat)
  dyana::tensor compute_loss(bool x, bool y, bool oracle) {
    dyana::tensor numeric_result = operator()((dyana::tensor)x, (dyana::tensor)y);
    return dyana::binary_log_loss(numeric_result, (dyana::tensor)oracle);
  }

};


int main(int argc, char** argv) {

  /*dynet::initialize(argc, argv);
  // create pc and trainer
  dynet::ParameterCollection pc;
  dynet::SimpleSGDTrainer trainer(pc);

  // defines flow of information
  dynet::ComputationGraph cg;

  // training matrix?
  dynet::Expression W = parameter(cg, pc.add_parameters({1,3}));

  // input and output?
  vector<dynet::real> x_values(3);
  dynet::Expression x = input(cg, {3}, &x_values);
  dynet::real y_value;
  dynet::Expression y = input(cg, &y_value);
  // prediction from x
  dynet::Expression y_pred = logistic(W*x);

  // calculate departure?
  dynet::Expression l = binary_log_loss(y_pred, y);
  cg.print_graphviz();
  // example
  x_values = {0.5, 0.3, 0.7};
  y_value = 1.0;

  // forward propagates values through computation graph, return loss
  dynet::real loss = as_scalar(cg.forward(l));
  cg.backward(l);
  trainer.update();

std::cout << loss;
    */

  dyana::initialize();


  /*
  dyana::tensor W({1,2,3,4,5,6},{3,2});
  dyana::tensor x({-1,0});

  dyana::tensor y = W*x;

  y = normalize(y);

  std::vector<float> values = y.as_vector();

  for (auto val:values) std::cout << val << " ";
  std::cout << std::endl;

  std::cout << "Max value: " << max1d(y).as_scalar() << std::endl;
  */


  // linear dense layer
  /*
  dyana::linear_dense_layer dense(2); // dim 2 output linear layer

  tanh_dense_layer test_tanh(dense);

  dyana::tensor x({-1, 0, 1, 2, 3}); //input

  dyana::tensor y = dense(x); // apply

  dyana::tensor ty = test_tanh(x);

  std::vector<float> values = y.as_vector();

  for (auto val: values) std::cout << val << " ";
  std::cout << std::endl;

  for (auto val : ty.as_vector()) std::cout << val << " ";
  std::cout << std::endl;
  */
  // you will see different result as the layer is initialized randomly
  // the layer only take output dimension, when this has been applied first time, it will be initialized


  // will see error by running also:
  /*
  dyana::tensor x2({1,2});
  dyana::tensor y2 = dense(x2);
  for (auto val: y2.as_vector()) std::cout << val << " ";
  std::cout << std::endl;
  */
  //terminate called after throwing an instance of 'std::runtime_error' what():  linear dense layer: input dimension mismatch. expected 5, got 2 Aborted (core dumped)



  // training
  // test
  std::vector<bool> input0s{true, true, false, false};
  std::vector<bool> input1s{true, false, true, false};
  std::vector<bool> oracles{false, true, true, false};

  xor_model test;

  dyana::adam_trainer trainer(0.1); // train rate
  trainer.num_epochs = 100;
  trainer.num_workers = 4;
  trainer.train(test, input0s, input1s, oracles);
  /*
  std::cout << "input0" << "," << "input1" << "," << "output" << std::endl;
  for(const auto &[input0, input1]:dyana::zip(input0s, input1s)) {
      using namespace std;
      std::cout << input0 << "," << input1 << "," << my_model(input0, input1) << std::endl;
  }
  */

  return 0;

}




dyana::tensor normalize(const dyana::tensor& x) {
  dyana::tensor modulus = dyana::l2_norm(x);
  return dyana::cdiv(x, modulus + 1e-6);
}

// e.g. of good forward backward implement

dyana::tensor max1d(const dyana::tensor& x) {
  if(x.dim().nd !=1) throw std::runtime_error("max1d only works with rank 1 tensor");
  std::vector<float> values = x.as_vector();
  auto max_iter = std::max_element(values.begin(), values.end());
  unsigned index_of_max_element = std::distance(max_iter, values.begin());
  return x.at(index_of_max_element);
}


// layer concept
// linear dense layer
// custom layer

// eg tanh dense layer



