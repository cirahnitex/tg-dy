//
// Created by YAN Yuchen on 1/22/2019.
//

#ifndef DYNET_WRAPPER_DY_NORMALIZATION_LAYER_HPP
#define DYNET_WRAPPER_DY_NORMALIZATION_LAYER_HPP
#include "dy_common.hpp"
#include "dy_serialization_helper.hpp"
namespace tg {
  namespace dy {
    class normalization_layer {
      Parameter g;
      Parameter b;
      void ensure_init(const tensor& x) {
        if(g.p) return; // ensure that this lazy initialization is only executed once
        auto dim = x.dim()[0];
        g = dy::add_parameters({dim});
        b = dy::add_parameters({dim});
      }
    public:
      EASY_SERIALZABLE(g, b)
      normalization_layer() = default;
      normalization_layer(const normalization_layer&) = default;
      normalization_layer(normalization_layer&&) = default;
      normalization_layer &operator=(const normalization_layer&) = default;
      normalization_layer &operator=(normalization_layer&&) = default;
      tensor predict(const tensor& x) {
        ensure_init(x);
        return dynet::layer_norm(x, tensor(g), tensor(b));
      }
    };
  }
}
#endif //DYNET_WRAPPER_DY_NORMALIZATION_LAYER_HPP
