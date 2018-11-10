//
// Created by YAN Yuchen on 5/8/2018.
//

#ifndef FRAME_ANALYSIS_DYNET_IOBES_READOUT_HPP
#define FRAME_ANALYSIS_DYNET_IOBES_READOUT_HPP
#include "dy_iobes_readout_base.hpp"

namespace tg {
  namespace dy {
    class iobes_readout :public iobes_readout_base{
    public:
      DECLARE_DEFAULT_CONSTRUCTORS(iobes_readout)
      iobes_readout(const std::vector<std::string>& tokens):
          iobes_readout_base({"I","B","E","S"}, tokens)
      {}
    protected:
      virtual std::pair<std::string, std::string> get_prefixed_label_at_token_index(unsigned index, const std::vector<labeled_span_type> &labeled_spans) const {
        for(auto itr = labeled_spans.begin(); itr!=labeled_spans.end(); ++itr) {
          if(index < itr->s()) continue;
          if(index == itr->s()) {
            if(index + 1 == itr->t()) return std::make_pair("S", itr->label());
            return std::make_pair("B", itr->label());
          }
          if(index + 1 == itr->t()) return make_pair("E", itr->label());
          if(index < itr->t()) return make_pair("I", itr->label());
        }
        return std::make_pair("O", "");
      }
    };
  }
}



#endif //FRAME_ANALYSIS_DYNET_IOBES_READOUT_HPP
