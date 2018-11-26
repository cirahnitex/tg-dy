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

      dy::tensor operator()(const dy::tensor& x) {
        return forward(x);
      }

      dy::tensor forward(const dy::tensor& x) {
        ensure_init(x);
        return W*x+b;
      }

      dy::tensor forward_given_output_positions(const dy::tensor& x, const std::vector<unsigned> output_positions) {
        ensure_init(x);
        auto selected_W = dy::select_rows(dy::tensor(W), output_positions);
        auto selected_b = dy::reshape(dy::pick(dy::tensor(b), output_positions), {(unsigned)output_positions.size()});
        return selected_W * x + selected_b;
      }

      EASY_SERIALZABLE(dim_in, dim_out, W, b)

    private:
      unsigned dim_in;
      unsigned dim_out;
      dy::Parameter W;
      dy::Parameter b;

      void ensure_init(const dy::tensor& input) {
        if(dim_in != 0) return;
        dim_in = input.dim()[0];
        W = add_parameters({dim_out, dim_in});
      }
    };
  }
}


#endif //FRAME_ANALYSIS_DYNET_LINEARLAYER_HPP
