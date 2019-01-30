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
    class multi_readout_model {
    public:
      multi_readout_model() = default;
      multi_readout_model(const multi_readout_model&) = default;
      multi_readout_model(multi_readout_model&&) = default;
      multi_readout_model &operator=(const multi_readout_model&) = default;
      multi_readout_model &operator=(multi_readout_model&&) = default;
      explicit multi_readout_model(const std::unordered_set<std::string>& labels):labels(labels.begin(), labels.end()),fc(labels.size()) {
      }
      std::unordered_set<std::string> readout(const dy::tensor &x) {
        const auto evidences = fc.predict(x).as_vector();
        std::unordered_set<std::string> ret;
        for (unsigned i = 0; i < labels.size(); i++) {
          const auto &label = labels[i];
          if (evidences[i] > 0) {
            ret.insert(label);
          }
        }
        return ret;
      }
      dy::tensor compute_loss(const dy::tensor& x, const std::unordered_set<std::string>& oracle) {
        using namespace std;
        vector<float> oracle_float;
        for(const auto& label:labels) {
          oracle_float.push_back(oracle.count(label)>0?(float)1:(float)0);
        }
        return dynet::binary_log_loss(dynet::logistic(fc.predict(x)), dy::tensor(oracle_float));
      }
      EASY_SERIALIZABLE(labels, fc)
    private:
      std::vector<std::string> labels;
      linear_layer fc;
    };
  }
}
#endif //DYNET_WRAPPER_DY_MULTI_READOUT_LAYER_HPP
