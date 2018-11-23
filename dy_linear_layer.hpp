//
// Created by YAN Yuchen on 4/25/2018.
//

#ifndef FRAME_ANALYSIS_DYNET_LINEARLAYER_HPP
#define FRAME_ANALYSIS_DYNET_LINEARLAYER_HPP

#include <dynet/dynet.h>
#include <dynet/expr.h>
#include "dy_common.hpp"
#include "dy_operations.hpp"
#include "dy_serialization_helper.hpp"
namespace tg {
  namespace dy {
    class linear_layer {
    public:
      linear_layer() = default;
      linear_layer(const linear_layer&) = default;
      linear_layer(linear_layer&&) = default;
      linear_layer &operator=(const linear_layer&) = default;
      linear_layer &operator=(linear_layer&&) = default;
      explicit linear_layer(unsigned dim_out)
          : dim_in(0), dim_out(dim_out), W(), b(add_parameters({dim_out})) {
      }

      dy::Tensor operator()(const dy::Tensor& x) {
        return forward(x);
      }

      dy::Tensor forward(const dy::Tensor& x) {
        ensure_init(x);
        return W*x+b;
      }

      dy::Tensor forward_given_output_positions(const dy::Tensor& x, const std::vector<unsigned> output_positions) {
        ensure_init(x);
        auto selected_W = dynet::select_rows(dy::Tensor(W), output_positions);
        auto selected_b = dynet::reshape(dynet::pick(dy::Tensor(b), output_positions), {(unsigned)output_positions.size()});
        return selected_W * x + selected_b;
      }

      EASY_SERIALZABLE(dim_in, dim_out, W, b)

    private:
      unsigned dim_in;
      unsigned dim_out;
      dy::Parameter W;
      dy::Parameter b;

      void ensure_init(const dy::Tensor& input) {
        if(dim_in != 0) return;
        dim_in = input.dim()[0];
        W = add_parameters({dim_out, dim_in});
      }
    };
  }
}


#endif //FRAME_ANALYSIS_DYNET_LINEARLAYER_HPP
