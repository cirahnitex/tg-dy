//
// Created by YAN Yuchen on 9/6/2018.
//

#include <dynet/dynet.h>
#include <dynet/expr.h>
#include "dy_common.hpp"
#include "dy_serialization_helper.hpp"

namespace tg {
  namespace dy {
    class conv2d_layer {
    public:
      conv2d_layer() = default;

      conv2d_layer(const conv2d_layer &) = default;

      conv2d_layer(conv2d_layer &&) = default;

      conv2d_layer &operator=(const conv2d_layer &) = default;

      conv2d_layer &operator=(conv2d_layer &&) = default;

      conv2d_layer(unsigned output_channels,
                   unsigned filter_height, unsigned filter_width, unsigned stride_between_rows = 1,
                   unsigned stride_between_columns = 1, bool with_bias = false, bool disable_padding = true) :
        input_channels(0), output_channels(output_channels), filter_height(filter_height),
        filter_width(filter_width),
        stride_between_rows(stride_between_rows), stride_between_columns(stride_between_columns),
        with_bias(with_bias), disable_padding(disable_padding),
        filter(),
        bias() {
        if (with_bias) bias = add_parameters({output_channels});
      }

      dynet::Dim calculate_output_dimension(unsigned input_height, unsigned input_width) {
        if (disable_padding) {
          return dynet::Dim({
                              (unsigned)ceil(float(input_height - filter_height + 1) / float(stride_between_rows)),
                              (unsigned)ceil(float(input_width - filter_width + 1) / float(stride_between_columns)),
                              output_channels
                            });
        } else {
          return dynet::Dim({
                              (unsigned)ceil(float(input_height) / float(stride_between_rows)),
                              (unsigned)ceil(float(input_width) / float(stride_between_columns)),
                              output_channels
                            });
        }
      }

      dynet::Expression forward(const dynet::Expression& x) {
        ensure_init(x);
        if(with_bias) {
          return dynet::conv2d(x, dy::expr(filter), dy::expr(bias), {stride_between_rows, stride_between_columns}, disable_padding);
        }
        else {
          return dynet::conv2d(x, dy::expr(filter), {stride_between_rows, stride_between_columns}, disable_padding);
        }
      }

      EASY_SERIALZABLE(input_channels, output_channels, filter_height, filter_width, stride_between_rows, stride_between_columns, with_bias, disable_padding, filter, bias)
    private:
      unsigned input_channels;
      unsigned output_channels;
      unsigned filter_height;
      unsigned filter_width;
      unsigned stride_between_rows;
      unsigned stride_between_columns;
      bool with_bias;
      bool disable_padding;
      dynet::Parameter filter;
      dynet::Parameter bias;

      void ensure_init(const dynet::Expression& x) {
        if(input_channels > 0) return;
        input_channels = x.dim()[2];
        filter = add_parameters({filter_height, filter_width, input_channels, output_channels});
      }
    };

    class conv1d_layer {
    public:
      static dynet::Expression reshape_to_conv2d_compatible(const dynet::Expression &x) {
        using namespace std;
        const auto& dim = x.dim();
        return dynet::reshape(dynet::transpose(x), dynet::Dim({dim[1], 1, dim[0]},x.dim().batch_elems()));
      }
      static dynet::Expression reshape_to_conv1d_compatible(const dynet::Expression &x) {
        const auto& dim = x.dim();
        return dynet::reshape(x, dynet::Dim({dim[2], dim[0]}, dim.batch_elems()));
      }

      conv1d_layer() = default;
      conv1d_layer(const conv1d_layer&) = default;
      conv1d_layer(conv1d_layer&&) = default;
      conv1d_layer &operator=(const conv1d_layer&) = default;
      conv1d_layer &operator=(conv1d_layer&&) = default;

      conv1d_layer(unsigned output_channels,
                   unsigned filter_length, unsigned stride = 1,
                   bool with_bias = false, bool disable_padding = true) :
        input_channels(0), output_channels(output_channels), filter_length(filter_length),
        stride(stride), with_bias(with_bias), disable_padding(disable_padding),
        filter(),
        bias() {
        if (with_bias) bias = add_parameters({output_channels});
      }

      unsigned calculate_output_length(unsigned input_length) {
        if (disable_padding) {
          return (unsigned)ceil(float(input_length - filter_length + 1) / float(stride));
        } else {
          return (unsigned)ceil(float(input_length) / float(stride));
        }
      }

      dynet::Expression forward(const dynet::Expression& x) {
        using namespace std;
        ensure_init(x);
        if(with_bias) {
          return reshape_to_conv1d_compatible(
            dynet::conv2d(reshape_to_conv2d_compatible(x), dy::expr(filter), dy::expr(bias), {stride, 1}, disable_padding));
        }
        else {
          return reshape_to_conv1d_compatible(
            dynet::conv2d(reshape_to_conv2d_compatible(x), dy::expr(filter), {stride, 1}, disable_padding));
        }
      }

      EASY_SERIALZABLE(input_channels, output_channels, filter_length,  stride, with_bias, disable_padding, filter, bias)
    private:
      unsigned input_channels;
      unsigned output_channels;
      unsigned filter_length;
      unsigned stride;
      bool with_bias;
      bool disable_padding;
      dynet::Parameter filter;
      dynet::Parameter bias;

      void ensure_init(const dynet::Expression& x) {
        if(input_channels > 0) return;
        input_channels = x.dim()[0];
        filter = add_parameters({filter_length, 1, input_channels, output_channels});
      }
    };

    dynet::Expression maxpooling2d(const dynet::Expression& x, unsigned window_width, unsigned window_height, unsigned stride_between_rows, unsigned stride_between_columns, bool disable_padding = true) {
      return dynet::maxpooling2d(x, {window_width, window_height}, {stride_between_rows, stride_between_columns}, disable_padding);
    }

    dynet::Expression maxpooling1d(const dynet::Expression& x, unsigned window_length, unsigned stride, bool disable_padding = true) {
      return conv1d_layer::reshape_to_conv1d_compatible(
        dynet::maxpooling2d(conv1d_layer::reshape_to_conv2d_compatible(x), {window_length, 1}, {stride, 1},
                            disable_padding));
    }
  }
}
