//
// Created by YAN Yuchen on 4/25/2018.
//

#ifndef FRAME_ANALYSIS_DYNET_LINEARLAYER_HPP
#define FRAME_ANALYSIS_DYNET_LINEARLAYER_HPP

#include <dynet/dynet.h>
#include <dynet/expr.h>
#include "dy_common.hpp"
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
          : dim_in(0), dim_out(dim_out), W(), b() {
      }

      dy::Expression operator()(const dy::Expression& x) {
        return forward(x);
      }

      dy::Expression forward(const dy::Expression& x) {
        ensure_init(x);
        return dy::expr(W)*x+dy::expr(b);
      }

      dy::Expression forward_given_output_positions(const dy::Expression& x, const std::vector<unsigned> output_positions) {
        ensure_init(x);
        auto selected_W = dynet::select_rows(dy::expr(W), output_positions);
        auto selected_b = dynet::reshape(dynet::pick(dy::expr(b), output_positions), {(unsigned)output_positions.size()});
        return selected_W * x + selected_b;
      }

      EASY_SERIALZABLE(dim_in, dim_out, W, b)

    private:
      unsigned dim_in;
      unsigned dim_out;
      dynet::Parameter W;
      dynet::Parameter b;

      void ensure_init(const dy::Expression& input) {
        if(dim_in != 0) return;
        dim_in = input.dim()[0];
        W = add_parameters({dim_out, dim_in});
        b = add_parameters({dim_out});
      }
    };
  }
}


#endif //FRAME_ANALYSIS_DYNET_LINEARLAYER_HPP
