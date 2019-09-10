//
// Created by YAN Yuchen on 1/22/2019.
//

#ifndef DYANA_NORMALIZATION_LAYER_HPP
#define DYANA_NORMALIZATION_LAYER_HPP

#include "dyana_common.hpp"
#include "dyana_serialization_helper.hpp"

namespace dyana {
  class normalization_layer {
    parameter g;
    parameter b;

    void ensure_init(const tensor &x) {
      if ((bool)g) return; // ensure that this lazy initialization is only executed once
      auto dim = x.dim()[0];
      g = parameter({dim});
      g.set_values(std::vector<float>(dim, (float)1));
      b = parameter({dim});
      b.set_values(std::vector<float>(dim, (float)0));
    }

  public:
    EASY_SERIALIZABLE(g, b)

    normalization_layer() = default;

    normalization_layer(const normalization_layer &) = default;

    normalization_layer(normalization_layer &&) = default;

    normalization_layer &operator=(const normalization_layer &) = default;

    normalization_layer &operator=(normalization_layer &&) = default;

    tensor operator()(const tensor &x) {
      ensure_init(x);
      return dynet::layer_norm(x, tensor(g), tensor(b));
    }
  };
}

#endif //DYANA_NORMALIZATION_LAYER_HPP
