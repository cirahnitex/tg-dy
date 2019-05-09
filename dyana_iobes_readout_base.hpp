//
// Created by YAN Yuchen on 5/8/2018.
//

#ifndef FRAME_ANALYSIS_DYNET_IOBES_READOUT_BASE_HPP
#define FRAME_ANALYSIS_DYNET_IOBES_READOUT_BASE_HPP

#include "dyana.hpp"
#include <dynet/dynet.h>
#include <chart.hpp>

namespace dyana {
  class iobes_readout_base {
  public:
    iobes_readout_base() = default;

    iobes_readout_base(const iobes_readout_base &) = default;

    iobes_readout_base(iobes_readout_base &&) = default;

    iobes_readout_base &operator=(const iobes_readout_base &) = default;

    iobes_readout_base &operator=(iobes_readout_base &&) = default;

    typedef tg::labeled_span<tg::span<unsigned>, std::string> labeled_span_type;

    template<typename RANGE_EXP0, typename RANGE_EXP1>
    iobes_readout_base(RANGE_EXP0 &&prefixes, RANGE_EXP1 &&labels) :
      ro_prefix(prefixes),
      ro_label(labels) {
      }

    /**
     * train the IOBES labeler
     * given sentence embeddings and oracle labeled span, compute loss
     * \param embeddings_in sentence embeddings, represented as a list of token embeddings
     * \param oracle true answer, labeled span
     * \return
     */
    dyana::tensor
    compute_loss(const std::vector<dyana::tensor> &embeddings_in, const std::vector<labeled_span_type> &oracle) {
      std::vector<dyana::tensor> ret;
      for (unsigned i = 0; i < embeddings_in.size(); i++) {
        std::string prefix_oracle, label_oracle;
        tie(prefix_oracle, label_oracle) = get_prefixed_label_at_token_index(i, oracle);

        // if the label is not NULL, both label and IOBES prefix need to be trained
        if (!label_oracle.empty()) {
          ret.push_back(ro_prefix.compute_loss(embeddings_in[i], prefix_oracle));
        }

        // if the label is NULL, only label need to be trained
        ret.push_back(ro_label.compute_loss(embeddings_in[i], label_oracle));
      }
      return dyana::sum(ret);
    }

    /**
     * predict labeled span given sentence embedding
     * \param embeddings_in embedding of each token in sentence
     * \return
     */
    std::vector<labeled_span_type> operator()(const std::vector<dyana::tensor> &embeddings_in) {
      enum {
        OUTSIDE, INSIDE
      } state = OUTSIDE;
      unsigned s_anchor = 0;
      std::string label_anchor;
      std::vector<labeled_span_type> ret;
      for (unsigned i = 0; i < embeddings_in.size(); ++i) {
        auto embedding = embeddings_in[i];
        auto prefix = ro_prefix.operator()(embedding);
        auto label = ro_label.operator()(embedding);

        if (state == OUTSIDE) {
          if (label.empty()) {

          } else if (prefix == "S") {
            ret.push_back(labeled_span_type(labeled_span_type::span_type(i, i + 1), label));
          } else {
            label_anchor = label;
            state = INSIDE;
            s_anchor = i;
          }
        } else {
          if (label.empty()) {
            ret.push_back(labeled_span_type(labeled_span_type::span_type(s_anchor, i), label_anchor));
            state = OUTSIDE;
          } else if (prefix == "S") {
            ret.push_back(labeled_span_type(labeled_span_type::span_type(s_anchor, i), label_anchor));
            ret.push_back(labeled_span_type(labeled_span_type::span_type(i, i + 1), label));
            state = OUTSIDE;
          } else if (label != label_anchor) {
            ret.push_back(labeled_span_type(labeled_span_type::span_type(s_anchor, i), label_anchor));
            label_anchor = label;
            s_anchor = i;
            if (prefix == "E") {
              ret.push_back(labeled_span_type(labeled_span_type::span_type(i, i + 1), label));
              state = OUTSIDE;
            }
          } else if (prefix == "I") {

          } else if (prefix == "B") {
            ret.push_back(labeled_span_type(labeled_span_type::span_type(s_anchor, i), label_anchor));
            label_anchor = label;
            s_anchor = i;
          } else if (prefix == "E") {
            ret.push_back(labeled_span_type(labeled_span_type::span_type(s_anchor, i + 1), label_anchor));
            state = OUTSIDE;
          }
        }
      }
      return ret;
    }

    EASY_SERIALIZABLE(ro_prefix, ro_label)

  protected:
    virtual std::pair<std::string, std::string> get_prefixed_label_at_token_index(unsigned index,
                                                                                  const std::vector<labeled_span_type> &labeled_spans) const = 0;

    dyana::readout_model ro_prefix;
    dyana::readout_model ro_label;
  };
}


#endif //FRAME_ANALYSIS_DYNET_IOBES_READOUT_BASE_HPP
