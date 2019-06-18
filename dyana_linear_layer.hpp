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


  namespace dyana {
    class linear_dense_layer {
    public:
      linear_dense_layer() = default;

      linear_dense_layer(const linear_dense_layer &) = default;

      linear_dense_layer(linear_dense_layer &&) = default;

      linear_dense_layer &operator=(const linear_dense_layer &) = default;

      linear_dense_layer &operator=(linear_dense_layer &&) = default;

      explicit linear_dense_layer(unsigned dim_out)
        : dim_in(0), dim_out(dim_out), W(), b({dim_out}) {
      }

      operator bool() const {
        return dim_out != 0;
      }

      dyana::tensor operator()(const dyana::tensor &x) {
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
  }



#endif //FRAME_ANALYSIS_DYNET_LINEARLAYER_HPP
