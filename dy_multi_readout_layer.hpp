//
// Created by YAN Yuchen on 11/12/2018.
//

#ifndef DYNET_WRAPPER_DY_MULTI_READOUT_LAYER_HPP
#define DYNET_WRAPPER_DY_MULTI_READOUT_LAYER_HPP
#include <vector>
#include <unordered_set>
#include <string>
#include "dy_common.hpp"
#include "dy_linear_layer.hpp"
namespace tg {
  namespace dy {
    class multi_readout_layer {
    public:
      multi_readout_layer() = default;
      multi_readout_layer(const multi_readout_layer&) = default;
      multi_readout_layer(multi_readout_layer&&) = default;
      multi_readout_layer &operator=(const multi_readout_layer&) = default;
      multi_readout_layer &operator=(multi_readout_layer&&) = default;
      explicit multi_readout_layer(const std::unordered_set<std::string>& labels):labels(labels.begin(), labels.end()),fc(labels.size()) {
      }
      std::unordered_set<std::string> readout(const dy::Tensor &x) {
        const auto evidences = fc.forward(x).as_vector();
        std::unordered_set<std::string> ret;
        for (unsigned i = 0; i < labels.size(); i++) {
          const auto &label = labels[i];
          if (evidences[i] > 0) {
            ret.insert(label);
          }
        }
        return ret;
      }
      dy::Tensor compute_loss(const dy::Tensor& x, const std::unordered_set<std::string>& oracle) {
        using namespace std;
        vector<float> oracle_float;
        for(const auto& label:labels) {
          oracle_float.push_back(oracle.count(label)>0?(float)1:(float)0);
        }
        return dynet::binary_log_loss(dynet::logistic(fc.forward(x)), dy::Tensor(oracle_float));
      }
      EASY_SERIALZABLE(labels, fc)
    private:
      std::vector<std::string> labels;
      linear_layer fc;
    };
  }
}
#endif //DYNET_WRAPPER_DY_MULTI_READOUT_LAYER_HPP
