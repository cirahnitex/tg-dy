//
// Created by YAN Yuchen on 9/6/2018.
//

#include "dyana_common.hpp"
#include "dyana_serialization_helper.hpp"


namespace dyana {
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
      if (with_bias) bias = dyana::parameter({output_channels});
    }

    operator bool() {
      return output_channels > 0;
    }

    dyana::Dim calculate_output_dimension(unsigned input_height, unsigned input_width) {
      if (disable_padding) {
        return dyana::Dim({
                            (unsigned) ceil(float(input_height - filter_height + 1) / float(stride_between_rows)),
                            (unsigned) ceil(float(input_width - filter_width + 1) / float(stride_between_columns)),
                            output_channels
                          });
      } else {
        return dyana::Dim({
                            (unsigned) ceil(float(input_height) / float(stride_between_rows)),
                            (unsigned) ceil(float(input_width) / float(stride_between_columns)),
                            output_channels
                          });
      }
    }

    dyana::tensor operator()(const dyana::tensor &x) {
      ensure_init(x);
      if (with_bias) {
        return dynet::conv2d(x, dyana::tensor(filter), dyana::tensor(bias),
                             {stride_between_rows, stride_between_columns}, disable_padding);
      } else {
        return dynet::conv2d(x, dyana::tensor(filter), {stride_between_rows, stride_between_columns}, disable_padding);
      }
    }

    EASY_SERIALIZABLE(input_channels, output_channels, filter_height, filter_width, stride_between_rows,
                      stride_between_columns, with_bias, disable_padding, filter, bias)

  private:
    unsigned input_channels;
    unsigned output_channels;
    unsigned filter_height;
    unsigned filter_width;
    unsigned stride_between_rows;
    unsigned stride_between_columns;
    bool with_bias;
    bool disable_padding;
    dyana::parameter filter;
    dyana::parameter bias;

    void ensure_init(const dyana::tensor &x) {
      if (input_channels > 0) return;
      input_channels = x.dim()[2];
      filter = dyana::parameter({filter_height, filter_width, input_channels, output_channels});
    }
  };

  inline dyana::tensor
  maxpooling2d(const dyana::tensor &x, unsigned window_height, unsigned window_width, unsigned stride_between_rows, unsigned stride_between_columns, bool disable_padding = true) {
    return dynet::maxpooling2d(x, {window_height, window_width}, {stride_between_rows, stride_between_columns}, disable_padding);
  }

  inline dyana::tensor global_maxpooling2d(const dyana::tensor& x) {
    auto&& dim = x.dim();
    return dynet::reshape(dynet::maxpooling2d(x, {dim[0], dim[1]}, {1, 1}), {dim[2]});
  }
}
