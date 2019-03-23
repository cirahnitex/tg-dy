//
// Created by YAN Yuchen on 5/8/2018.
//

#ifndef FRAME_ANALYSIS_DYNET_IOE_READOUT_HPP
#define FRAME_ANALYSIS_DYNET_IOE_READOUT_HPP

#include "dyana_iobes_readout_base.hpp"
namespace tg {
  namespace dyana {
    class ioe_readout :public iobes_readout_base {
    public:
      ioe_readout() = default;
      ioe_readout(const ioe_readout&) = default;
      ioe_readout(ioe_readout&&) = default;
      ioe_readout &operator=(const ioe_readout&) = default;
      ioe_readout &operator=(ioe_readout&&) = default;
      ioe_readout(std::unordered_set<std::string>& tokens):
          iobes_readout_base({"I","E"}, tokens)
      {}
    protected:
      virtual std::pair<std::string, std::string> get_prefixed_label_at_token_index(unsigned index, const std::vector<labeled_span_type> &labeled_spans) const {
        for(auto itr = labeled_spans.begin(); itr!=labeled_spans.end(); ++itr) {
          if(index < itr->s()) continue;
          if(index + 1 == itr->t()) return make_pair("E", itr->label());
          if(index < itr->t()) return make_pair("I", itr->label());
        }
        return std::make_pair("O", "");
      }
    };
  }
}

#endif //FRAME_ANALYSIS_DYNET_IOE_READOUT_HPP
