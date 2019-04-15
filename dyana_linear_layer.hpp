//
// Created by YAN Yuchen on 4/25/2018.
//

#ifndef FRAME_ANALYSIS_DYNET_LINEARLAYER_HPP
#define FRAME_ANALYSIS_DYNET_LINEARLAYER_HPP

#include <dynet/dynet.h>
#include <dynet/expr.h>
#include "dyana_common.hpp"
#include "dyana_operations.hpp"
#include "dyana_serialization_helper.hpp"

namespace tg {
  namespace dyana {
    class linear_layer {
    public:
      linear_layer() = default;

      linear_layer(const linear_layer &) = default;

      linear_layer(linear_layer &&) = default;

      linear_layer &operator=(const linear_layer &) = default;

      linear_layer &operator=(linear_layer &&) = default;

      explicit linear_layer(unsigned dim_out)
        : dim_in(0), dim_out(dim_out), W(), b({dim_out}) {
      }

      dyana::tensor transduce(const dyana::tensor &x) {
        ensure_init(x);
        if(x.dim()[0] != dim_in) throw std::runtime_error("linear dense layer: input dimension mismatch. expected " + std::to_string(dim_in) + ", got " + std::to_string(x.dim()[0]));
        return W * x + b;
      }

      dyana::tensor
      predict_given_output_positions(const dyana::tensor &x, const std::vector<unsigned> &output_positions) {
        ensure_init(x);
        auto selected_W = dyana::tensor(W).select_rows(output_positions);
        auto selected_b = dyana::tensor(b).select_rows(output_positions);
        return selected_W * x + selected_b;
      }

      EASY_SERIALIZABLE(dim_in, dim_out, W, b)

    private:
      unsigned dim_in;
      unsigned dim_out;
      dyana::parameter W;
      dyana::parameter b;

      void ensure_init(const dyana::tensor &input) {
        if (dim_in != 0) return;
        dim_in = input.dim()[0];
        W = parameter({dim_out, dim_in});
      }
    };

    class dense_layer {
    public:
      enum ACTIVATION {
        IDENTITY, SIGMOID, TANH, RELU
      };
    private:
      linear_layer linear;
      ACTIVATION activation;
    public:

      EASY_SERIALIZABLE(linear, activation)

      dense_layer() = default;

      dense_layer(const dense_layer &) = default;

      dense_layer(dense_layer &&) noexcept = default;

      dense_layer &operator=(const dense_layer &) = default;

      dense_layer &operator=(dense_layer &&) noexcept = default;

      dense_layer(unsigned dim_out, ACTIVATION activation = IDENTITY) : linear(dim_out), activation(activation) {
      }

      dyana::tensor apply_activation(const dyana::tensor &x) {
        switch (activation) {
          case IDENTITY:
            return x;
          case SIGMOID:
            return dyana::logistic(x);
          case TANH:
            return dyana::tanh(x);
          case RELU:
            return dyana::rectify(x);
          default:
            return x;
        }
      }


      dyana::tensor transduce(const dyana::tensor &x) {
        return apply_activation(linear.transduce(x));
      }

      dyana::tensor
      transduce_given_output_positions(const dyana::tensor &x, const std::vector<unsigned> &output_positions) {
        return apply_activation(linear.predict_given_output_positions(x, output_positions));
      }
    };
  }
}


#endif //FRAME_ANALYSIS_DYNET_LINEARLAYER_HPP
