//
// Created by YAN Yuchen on 5/8/2018.
//

#ifndef FRAME_ANALYSIS_DYNET_IOB_READOUT_HPP
#define FRAME_ANALYSIS_DYNET_IOB_READOUT_HPP

#include "dyana_iobes_readout_base.hpp"

namespace dyana {
  class iob_readout : public iobes_readout_base {
  public:
    iob_readout() = default;

    iob_readout(const iob_readout &) = default;

    iob_readout(iob_readout &&) = default;

    iob_readout &operator=(const iob_readout &) = default;

    iob_readout &operator=(iob_readout &&) = default;

    template<typename RANGE_EXP>
    iob_readout(RANGE_EXP &&tokens) :
      iobes_readout_base({"I", "B"}, tokens) {}

  protected:
    virtual std::pair<std::string, std::string>
    get_prefixed_label_at_token_index(unsigned index, const std::vector<labeled_span_type> &labeled_spans) const {
      for (auto itr = labeled_spans.begin(); itr != labeled_spans.end(); ++itr) {
        if (index < itr->s()) continue;
        if (index == itr->s()) return make_pair("B", itr->label());
        if (index < itr->t()) return make_pair("I", itr->label());
      }
      return std::make_pair("O", "");
    }
  };
}


#endif //FRAME_ANALYSIS_DYNET_IOB_READOUT_HPP
