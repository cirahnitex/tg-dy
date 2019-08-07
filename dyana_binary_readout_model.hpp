//
// Created by YAN Yuchen on 1/22/2019.
//

#ifndef DYANA_BINARY_READOUT_MODEL_HPP
#define DYANA_BINARY_READOUT_MODEL_HPP

#include "dyana_common.hpp"
#include "dyana_serialization_helper.hpp"
#include "dyana_linear_layer.hpp"


namespace dyana {
  /*
   * a binary transduce model does the following when predicting
   * * takes an embedding (tensor<X>)
   * * predicts a true/false answer
   */
  class binary_readout_model {
    linear_dense_layer fc;
  public:
    EASY_SERIALIZABLE(fc)

    binary_readout_model() : fc(1) {};

    binary_readout_model(const binary_readout_model &) = default;

    binary_readout_model(binary_readout_model &&) = default;

    binary_readout_model &operator=(const binary_readout_model &) = default;

    binary_readout_model &operator=(binary_readout_model &&) = default;

    operator bool() const {
      return (bool)fc;
    }

    /**
     * performs prediction
     * \param x tensor<X> the embedding
     * \return the answer
     */
    bool operator()(const tensor &x) {
      return fc(x).as_scalar() > 0;
    }

    /**
     * compute loss
     * \param x tensor<X> the embedding
     * \param oracle the golden answer
     * \return the loss
     */
    tensor compute_loss(const tensor &x, bool oracle) {
      if(oracle) {
        return -dyana::log_sigmoid(fc(x));
      }
      else {
        return -dyana::log_sigmoid(-fc(x));
      }
    }
  };
}


#endif //DYANA_BINARY_READOUT_MODEL_HPP
