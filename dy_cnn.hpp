//
// Created by YAN Yuchen on 9/6/2018.
//

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

      dy::Expression forward(const dy::Expression& x) {
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

      void ensure_init(const dy::Expression& x) {
        if(input_channels > 0) return;
        input_channels = x.dim()[2];
        filter = add_parameters({filter_height, filter_width, input_channels, output_channels});
      }
    };

    class conv1d_layer {
    public:
      static dy::Expression reshape_to_conv2d_compatible(const dy::Expression &x) {
        using namespace std;
        const auto& dim = x.dim();
        return dynet::reshape(dynet::transpose(x), dynet::Dim({dim[1], 1, dim[0]},x.dim().batch_elems()));
      }
      static dy::Expression reshape_to_conv1d_compatible(const dy::Expression &x) {
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

      dy::Expression forward(const dy::Expression& x) {
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

      void ensure_init(const dy::Expression& x) {
        if(input_channels > 0) return;
        input_channels = x.dim()[0];
        filter = add_parameters({filter_length, 1, input_channels, output_channels});
      }
    };

    dy::Expression maxpooling2d(const dy::Expression& x, unsigned window_width, unsigned window_height, unsigned stride_between_rows, unsigned stride_between_columns, bool disable_padding = true) {
      return dynet::maxpooling2d(x, {window_width, window_height}, {stride_between_rows, stride_between_columns}, disable_padding);
    }

    dy::Expression maxpooling1d(const dy::Expression& x, unsigned window_length, unsigned stride, bool disable_padding = true) {
      return conv1d_layer::reshape_to_conv1d_compatible(
        dynet::maxpooling2d(conv1d_layer::reshape_to_conv2d_compatible(x), {window_length, 1}, {stride, 1},
                            disable_padding));
    }

    std::vector<dy::Expression> my_maxpooling1d(const std::vector<dy::Expression>& xs, unsigned window_length, unsigned stride, bool disable_padding = true) {
      using namespace std;
      if(disable_padding) {
        const int bound = xs.size() - window_length;
        if(bound<0) {
          return std::vector<dy::Expression>({dy::max(xs)});
        }
        else {
          std::vector<dy::Expression> ys;
          for(int i=0; i<=bound; i+=stride) {
            std::vector<dy::Expression> inners(xs.begin()+i, xs.begin()+i+window_length);
            ys.push_back(dy::max(inners));
          }
          return ys;
        }
      }
      else {
        //TODO: support padding
        throw std::runtime_error("not implemented");
      }
    }

    class my_conv1d_layer {
    public:
      my_conv1d_layer(unsigned output_channels,
                   unsigned filter_length, unsigned stride = 1,
                   bool with_bias = false, bool disable_padding = true) :
        input_channels(0), output_channels(output_channels), filter_length(filter_length),
        stride(stride), with_bias(with_bias), disable_padding(disable_padding),
        filters(filter_length),
        bias() {
        if (with_bias) bias = add_parameters({output_channels});
      }
      std::vector<dy::Expression> forward(const std::vector<dy::Expression>& xs) {
        if(xs.empty()) return std::vector<dy::Expression>();
        this->ensure_init(xs[0]);
        if(disable_padding) {
          const int bound = xs.size() - filter_length;
          if(bound<0) {
            std::vector<dy::Expression> inners;
            for(unsigned i=0; i<xs.size(); i++) {
              inners.push_back(dy::expr(filters[i])*xs[i]);
            }
            if(with_bias) {
              return std::vector<dy::Expression>({dy::sum(inners)+dy::expr(bias)});
            }
            else {
              return std::vector<dy::Expression>({dy::sum(inners)});
            }
          }
          else {
            std::vector<dy::Expression> ys;
            for(int i=0; i<=bound; i+=stride) {
              std::vector<dy::Expression> inners;
              for(unsigned offset=0; offset<filter_length; offset++) {
                inners.push_back(dy::expr(filters[offset])*xs[bound+offset]);
              }
              if(with_bias) {
                ys.push_back(dy::sum(inners)+dy::expr(bias));
              }
              else {
                ys.push_back(dy::sum(inners));
              }
            }
            return ys;
          }
        }
        else {
          //TODO: support padding
          throw std::runtime_error("not implemented");
        }
      }
    private:
      unsigned input_channels;
      unsigned output_channels;
      unsigned filter_length;
      unsigned stride;
      bool with_bias;
      bool disable_padding;
      std::vector<dynet::Parameter> filters;
      dynet::Parameter bias;
      void ensure_init(const dy::Expression& x) {
        if(input_channels > 0) return;
        input_channels = x.dim()[0];
        for(auto& filter:filters) {
          filter = add_parameters({output_channels, input_channels});
        }
      }
    };
  }
}
