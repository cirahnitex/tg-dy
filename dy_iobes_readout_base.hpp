//
// Created by YAN Yuchen on 5/8/2018.
//

#ifndef FRAME_ANALYSIS_DYNET_IOBES_READOUT_BASE_HPP
#define FRAME_ANALYSIS_DYNET_IOBES_READOUT_BASE_HPP
#include "dy.hpp"
#include <dynet/dynet.h>
#include <srl_graph.hpp>

namespace tg {
  typedef srl_graph<std::size_t, std::size_t, std::string, std::string> web_srl_graph;
  namespace dy {
    class iobes_readout_base {
    public:
      iobes_readout_base() = default;
      iobes_readout_base(const iobes_readout_base&) = default;
      iobes_readout_base(iobes_readout_base&&) = default;
      iobes_readout_base &operator=(const iobes_readout_base&) = default;
      iobes_readout_base &operator=(iobes_readout_base&&) = default;
      typedef web_srl_graph::item_type labeled_span_type;
      iobes_readout_base(const std::unordered_set<std::string>& prefixes, const std::unordered_set<std::string>& labels):
          ro_prefix(prefixes),
          ro_label(std::unordered_set<std::string>({""}), labels)
      {}
      /**
       * train the IOBES labeler
       * given sentence embeddings and oracle labeled span, compute loss
       * \param embeddings_in sentence embeddings, represented as a list of token embeddings
       * \param oracle true answer, labeled span
       * \return
       */
      dy::tensor compute_loss(const std::vector<dy::tensor>& embeddings_in, const std::vector<web_srl_graph::item_type> oracle) {
        std::vector<dy::tensor> ret;
        for(unsigned i=0; i<embeddings_in.size(); i++) {
          std::string prefix_oracle, label_oracle;
          tie(prefix_oracle, label_oracle) = get_prefixed_label_at_token_index(i, oracle);

          // if the label is not NULL, both label and IOBES prefix need to be trained
          if(!label_oracle.empty()) {
            ret.push_back(ro_prefix.compute_loss(embeddings_in[i], prefix_oracle));
          }

          // if the label is NULL, only label need to be trained
          ret.push_back(ro_label.compute_loss(embeddings_in[i], label_oracle));
        }
        return dy::sum(ret);
      }

      /**
       * predict labeled span given sentence embedding
       * \param embeddings_in embedding of each token in sentence
       * \return
       */
      std::vector<web_srl_graph::item_type> predict(const std::vector<dy::tensor> &embeddings_in) {
        enum {OUTSIDE, INSIDE} state = OUTSIDE;
        unsigned s_anchor = 0;
        std::string label_anchor;
        std::vector<web_srl_graph::item_type> ret;
        for(unsigned i=0; i<embeddings_in.size(); ++i) {
          auto embedding = embeddings_in[i];
          auto prefix = ro_prefix.predict(embedding);
          auto label = ro_label.predict(embedding);

          if(state == OUTSIDE) {
            if(label.empty()) {

            }
            else if(prefix == "S") {
              ret.push_back(labeled_span_type(labeled_span_type::span_type(i,i+1), label));
            }
            else {
              label_anchor = label;
              state = INSIDE;
              s_anchor = i;
            }
          }
          else {
            if(label.empty()) {
              ret.push_back(labeled_span_type(labeled_span_type::span_type(s_anchor,i), label_anchor));
              state = OUTSIDE;
            }
            else if(prefix == "S") {
              ret.push_back(labeled_span_type(labeled_span_type::span_type(s_anchor,i), label_anchor));
              ret.push_back(labeled_span_type(labeled_span_type::span_type(i,i+1), label));
              state = OUTSIDE;
            }
            else if(label != label_anchor) {
              ret.push_back(labeled_span_type(labeled_span_type::span_type(s_anchor,i), label_anchor));
              label_anchor = label;
              s_anchor = i;
              if(prefix == "E") {
                ret.push_back(labeled_span_type(labeled_span_type::span_type(i,i+1), label));
                state = OUTSIDE;
              }
            }
            else if(prefix == "I") {

            }
            else if(prefix == "B") {
              ret.push_back(labeled_span_type(labeled_span_type::span_type(s_anchor,i), label_anchor));
              label_anchor = label;
              s_anchor = i;
            }
            else if(prefix == "E") {
              ret.push_back(labeled_span_type(labeled_span_type::span_type(s_anchor,i+1), label_anchor));
              state = OUTSIDE;
            }
          }
        }
        return ret;
      }

      EASY_SERIALZABLE(ro_prefix, ro_label)
    protected:
      virtual std::pair<std::string, std::string> get_prefixed_label_at_token_index(unsigned index, const std::vector<web_srl_graph::item_type> &labeled_spans) const = 0;
      dy::readout_model ro_prefix;
      dy::readout_model ro_label;
    };
  }

}



#endif //FRAME_ANALYSIS_DYNET_IOBES_READOUT_BASE_HPP
