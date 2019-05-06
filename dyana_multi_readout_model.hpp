//
// Created by YAN Yuchen on 11/12/2018.
//

#ifndef DYANA_MULTI_READOUT_LAYER_HPP
#define DYANA_MULTI_READOUT_LAYER_HPP

#include <vector>
#include <unordered_set>
#include <string>
#include "dyana_common.hpp"
#include "dyana_linear_layer.hpp"

namespace dyana {
  /**
   * a multi-readout model does the following when predicting:
   * * takes an embedding
   * * predicts zero or many labels
   * this is equivalent to have a binary readout for every possible label
   */
  class multi_readout_model {
  public:
    multi_readout_model() = default;

    multi_readout_model(const multi_readout_model &) = default;

    multi_readout_model(multi_readout_model &&) = default;

    multi_readout_model &operator=(const multi_readout_model &) = default;

    multi_readout_model &operator=(multi_readout_model &&) = default;

    /**
     * construct a multi-readout model
     * \param labels the set of all possible labels
     */
    explicit multi_readout_model(const std::unordered_set<std::string> &labels) : labels(labels.begin(), labels.end()),
                                                                                  fc(labels.size()) {
    }

    /**
     * perform a prediction
     * \param x the embedding
     * \return predicted labels
     */
    std::unordered_set<std::string> readout(const dyana::tensor &x) {
      const auto evidences = fc.operator()(x).as_vector();
      std::unordered_set<std::string> ret;
      for (unsigned i = 0; i < labels.size(); i++) {
        const auto &label = labels[i];
        if (evidences[i] > 0) {
          ret.insert(label);
        }
      }
      return ret;
    }

    /**
     * computes the loss given training data
     * \param x the embedding
     * \param oracle the golden answer
     * \return the loss
     */
    dyana::tensor compute_loss(const dyana::tensor &x, const std::unordered_set<std::string> &oracle) {
      using namespace std;
      vector<float> oracle_float;
      for (const auto &label:labels) {
        oracle_float.push_back(oracle.count(label) > 0 ? (float) 1 : (float) 0);
      }
      return dynet::binary_log_loss(dynet::logistic(fc.operator()(x)), dyana::tensor(oracle_float));
    }

    EASY_SERIALIZABLE(labels, fc)

  private:
    std::vector<std::string> labels;
    linear_dense_layer fc;
  };
}

#endif //DYANA_MULTI_READOUT_LAYER_HPP
