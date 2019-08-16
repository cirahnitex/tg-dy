//
// Created by Dekai WU and YAN Yuchen on 20190816.
//

#ifndef DYANA_CPP_DYANA_DROPOUT_LAYER_HPP
#define DYANA_CPP_DYANA_DROPOUT_LAYER_HPP
#include "dyana_common.hpp"
#include "dyana_operations.hpp"
#include "dyana_serialization_helper.hpp"
#include "dyana_mp_train.hpp"

namespace dyana {
  /**
   * a layer that performs dropout operation
   * this layer has no effect if not training
   */
  class dropout_layer {
    float dropout_rate;
  public:
    EASY_SERIALIZABLE(dropout_rate)
    dropout_layer(const dropout_layer&) = default;
    dropout_layer(dropout_layer&&) noexcept = default;
    dropout_layer &operator=(const dropout_layer&) = default;
    dropout_layer &operator=(dropout_layer&&) noexcept = default;
    dropout_layer(float dropout_rate = 0.9):dropout_rate(dropout_rate) {};
    dyana::tensor operator()(const dyana::tensor& x) {
      return dyana::is_training()?dyana::dropout(x, dropout_rate):x;
    }
  };
}

#endif
