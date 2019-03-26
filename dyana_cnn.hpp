//
// Created by YAN Yuchen on 9/6/2018.
//

#include "dyana_common.hpp"
#include "dyana_serialization_helper.hpp"

namespace tg {
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
        if (with_bias) bias = parameter({output_channels});
      }

      dyana::Dim calculate_output_dimension(unsigned input_height, unsigned input_width) {
        if (disable_padding) {
          return dyana::Dim({
                              (unsigned)ceil(float(input_height - filter_height + 1) / float(stride_between_rows)),
                              (unsigned)ceil(float(input_width - filter_width + 1) / float(stride_between_columns)),
                              output_channels
                            });
        } else {
          return dyana::Dim({
                              (unsigned)ceil(float(input_height) / float(stride_between_rows)),
                              (unsigned)ceil(float(input_width) / float(stride_between_columns)),
                              output_channels
                            });
        }
      }

      dyana::tensor transduce(const dyana::tensor &x) {
        ensure_init(x);
        if(with_bias) {
          return dynet::conv2d(x, dyana::tensor(filter), dyana::tensor(bias), {stride_between_rows, stride_between_columns}, disable_padding);
        }
        else {
          return dynet::conv2d(x, dyana::tensor(filter), {stride_between_rows, stride_between_columns}, disable_padding);
        }
      }

      EASY_SERIALIZABLE(input_channels, output_channels, filter_height, filter_width, stride_between_rows, stride_between_columns, with_bias, disable_padding, filter, bias)
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

      void ensure_init(const dyana::tensor& x) {
        if(input_channels > 0) return;
        input_channels = x.dim()[2];
        filter = parameter({filter_height, filter_width, input_channels, output_channels});
      }
    };

    inline dyana::tensor maxpooling2d(const dyana::tensor& x, unsigned window_width, unsigned window_height, unsigned stride_between_rows, unsigned stride_between_columns, bool disable_padding = true) {
      return dynet::maxpooling2d(x, {window_width, window_height}, {stride_between_rows, stride_between_columns}, disable_padding);
    }

    inline std::vector<dyana::tensor> maxpooling1d(const std::vector<dyana::tensor>& xs, unsigned window_length, unsigned stride, bool disable_padding = true) {
      if(xs.empty()) {throw std::runtime_error("cannot call maxpool1d on empty vector");}
      using namespace std;
      if(disable_padding) {
        const int bound = xs.size() - window_length;
        if(bound<0) {
          return std::vector<dyana::tensor>({dyana::max(xs)});
        }
        else {
          std::vector<dyana::tensor> ys;
          for(int i=0; i<=bound; i+=stride) {
            std::vector<dyana::tensor> inners(xs.begin()+i, xs.begin()+i+window_length);
            ys.push_back(dyana::max(inners));
          }
          return ys;
        }
      }
      else {
        int pad_begin = (window_length - 1) / 2;
        int pad_end = window_length - 1 - pad_begin;
        int max_end = xs.size();
        std::vector<dyana::tensor> ys;
        for(int i=0; i<max_end; i+=stride) {
          int begin = std::max(0, i-pad_begin);
          int end_ = std::min(max_end, i+pad_end);
          std::vector<dyana::tensor> inners(xs.begin()+begin, xs.begin()+end_);
          ys.push_back(dyana::max(inners));
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
        if (with_bias) bias = parameter({output_channels});
      }
      std::vector<dyana::tensor> transduce(const std::vector<dyana::tensor> &xs) {
        if(xs.empty()) return std::vector<dyana::tensor>();
        this->ensure_init(xs[0]);
        if(disable_padding) {return forward_no_padding(xs);}
        else {
          auto zeros = dyana::zeros(xs[0].dim());
          unsigned pad_begin = (filter_length - 1) / 2;
          unsigned pad_end = filter_length - 1 - pad_begin;

          // zeros at beginning
          std::vector<tensor> padded(pad_begin, zeros);

          // values in the middle
          std::copy(xs.begin(), xs.end(), std::back_inserter(padded));

          // zeros at the end
          for(unsigned i=0; i<pad_end; i++) {
            padded.push_back(zeros);
          }
          return forward_no_padding(padded);
        }
      }
      EASY_SERIALIZABLE(input_channels, output_channels, filter_length, stride, with_bias, disable_padding)
    private:
      unsigned input_channels;
      unsigned output_channels;
      unsigned filter_length;
      unsigned stride;
      bool with_bias;
      bool disable_padding;
      std::vector<parameter> filters;
      parameter bias;
      void ensure_init(const dyana::tensor& x) {
        if(input_channels > 0) return;
        input_channels = x.dim()[0];
        for(auto& filter:filters) {
          filter = parameter({output_channels, input_channels});
        }
      }
      std::vector<dyana::tensor> forward_no_padding(const std::vector<dyana::tensor>& xs) {
        const int bound = xs.size() - filter_length;
        if(bound<0) {
          std::vector<dyana::tensor> inners;
          for(unsigned i=0; i<xs.size(); i++) {
            inners.push_back(filters[i]*xs[i]);
          }
          if(with_bias) {
            return std::vector<dyana::tensor>({dyana::sum(inners)+bias});
          }
          else {
            return std::vector<dyana::tensor>({dyana::sum(inners)});
          }
        }
        else {
          std::vector<dyana::tensor> ys;
          for(int i=0; i<=bound; i+=stride) {
            std::vector<dyana::tensor> inners;
            for(unsigned offset=0; offset<filter_length; offset++) {
              inners.push_back(filters[offset]*xs[bound+offset]);
            }
            if(with_bias) {
              ys.push_back(dyana::sum(inners)+bias);
            }
            else {
              ys.push_back(dyana::sum(inners));
            }
          }
          return ys;
        }
      }
    };
  }
}
