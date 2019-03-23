//
// Created by YAN Yuchen on 1/22/2019.
//

#ifndef DYANA_NORMALIZATION_LAYER_HPP
#define DYANA_NORMALIZATION_LAYER_HPP
#include "dyana_common.hpp"
#include "dyana_serialization_helper.hpp"
namespace tg {
  namespace dyana {
    class normalization_layer {
      parameter g;
      parameter b;
      void ensure_init(const tensor& x) {
        if(g.is_nil()) return; // ensure that this lazy initialization is only executed once
        auto dim = x.dim()[0];
        g = parameter({dim});
        b = parameter({dim});
      }
    public:
      EASY_SERIALIZABLE(g, b)
      normalization_layer() = default;
      normalization_layer(const normalization_layer&) = default;
      normalization_layer(normalization_layer&&) = default;
      normalization_layer &operator=(const normalization_layer&) = default;
      normalization_layer &operator=(normalization_layer&&) = default;
      tensor predict(const tensor& x) {
        ensure_init(x);
        return dynet::layer_norm(x, tensor(g), tensor(b));
      }
      tensor operator()(const tensor& x) {return predict(x);}
    };
  }
}
#endif //DYANA_NORMALIZATION_LAYER_HPP
