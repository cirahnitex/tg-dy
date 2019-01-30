//
// Created by YAN Yuchen on 1/22/2019.
//

#ifndef DYNET_WRAPPER_DY_BINARY_READOUT_MODEL_HPP
#define DYNET_WRAPPER_DY_BINARY_READOUT_MODEL_HPP

#include "dy_common.hpp"
#include "dy_serialization_helper.hpp"
#include "dy_linear_layer.hpp"

namespace tg {
  namespace dy {
    class binary_readout_model {
      linear_layer fc;
    public:
      EASY_SERIALIZABLE(fc)
      binary_readout_model():fc(1){};
      binary_readout_model(const binary_readout_model&) = default;
      binary_readout_model(binary_readout_model&&) = default;
      binary_readout_model &operator=(const binary_readout_model&) = default;
      binary_readout_model &operator=(binary_readout_model&&) = default;
      bool predict(const tensor &x) {
        return fc.predict(x).as_scalar() > 0;
      }

      tensor compute_loss(const tensor &x, bool oracle) {
        return dy::binary_log_loss(dy::logistic(fc.predict(x)), oracle);
      }
    };
  }
}

#endif //DYNET_WRAPPER_DY_BINARY_READOUT_MODEL_HPP
