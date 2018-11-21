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

    dy::Expression maxpooling2d(const dy::Expression& x, unsigned window_width, unsigned window_height, unsigned stride_between_rows, unsigned stride_between_columns, bool disable_padding = true) {
      return dynet::maxpooling2d(x, {window_width, window_height}, {stride_between_rows, stride_between_columns}, disable_padding);
    }

    std::vector<dy::Expression> maxpooling1d(const std::vector<dy::Expression>& xs, unsigned window_length, unsigned stride, bool disable_padding = true) {
      if(xs.empty()) {throw std::runtime_error("cannot call maxpool1d on empty vector");}
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
        int pad_begin = (window_length - 1) / 2;
        int pad_end = window_length - 1 - pad_begin;
        int max_end = xs.size();
        std::vector<dy::Expression> ys;
        for(int i=0; i<max_end; i+=stride) {
          int begin = std::max(0, i-pad_begin);
          int end_ = std::min(max_end, i+pad_end);
          std::vector<dy::Expression> inners(xs.begin()+begin, xs.begin()+end_);
          ys.push_back(dy::max(inners));
        }
        return ys;
      }
    }

    class conv1d_layer {
    public:
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
        filters(filter_length),
        bias() {
        if (with_bias) bias = add_parameters({output_channels});
      }
      std::vector<dy::Expression> forward(const std::vector<dy::Expression>& xs) {
        if(xs.empty()) return std::vector<dy::Expression>();
        this->ensure_init(xs[0]);
        if(disable_padding) {return forward_no_padding(xs);}
        else {
          auto zeros = dy::zeros(xs[0].dim());
          unsigned pad_begin = (filter_length - 1) / 2;
          unsigned pad_end = filter_length - 1 - pad_begin;

          // zeros at beginning
          std::vector<Expression> padded(pad_begin, zeros);

          // values in the middle
          std::copy(xs.begin(), xs.end(), std::back_inserter(padded));

          // zeros at the end
          for(unsigned i=0; i<pad_end; i++) {
            padded.push_back(zeros);
          }
          return forward_no_padding(padded);
        }
      }
      EASY_SERIALZABLE(input_channels, output_channels, filter_length, stride, with_bias, disable_padding)
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
      std::vector<dy::Expression> forward_no_padding(const std::vector<dy::Expression>& xs) {
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
    };
  }
}
