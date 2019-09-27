//
// Created by Dekai WU and YAN Yuchen on 20190927.
//

#ifndef DYANA_CPP_DYANA_PARALLEL_MAP_HPP
#define DYANA_CPP_DYANA_PARALLEL_MAP_HPP

#include <dynet/dynet.h>
#include <functional>
#include <utility>
#include <dynet/mp.h>
#include "dyana_common.hpp"
#include "dyana_event_emitter.hpp"

namespace dyana {

  template<typename ITEM>
  class _parallel_map_learner : private dynet::mp::ILearner<std::pair<unsigned, ITEM>, float> {
  public:
    using numbered_item_t = std::pair<unsigned, ITEM>;

    _parallel_map_learner(const std::vector<ITEM>& items, std::vector<dynet::Parameter>& output_container,
                          const std::function<std::vector<float>(const ITEM&)>& fn, unsigned ret_dim,
                          unsigned num_workers)
      :
      fn(fn), output_container(output_container) {

      std::vector<numbered_item_t> numbered_items;
      for (unsigned i = 0; i < items.size(); i++) {
        numbered_items.emplace_back(i, items[i]);
      }

      dynet::ParameterCollection _;
      dynet::SimpleSGDTrainer trainer(_);

      {
        multiprocessing_guard __;
        dynet::mp::run_multi_process(num_workers, this, &trainer, numbered_items, {}, 1,
                                     numbered_items.size(), numbered_items.size());
      }

    }

  private:
    virtual float LearnFromDatum(const numbered_item_t& numbered_item, bool learn) {
      if (dyana::tensor::get_exprs_counter() != 0) {
        throw std::runtime_error(
          "NO GLOBAL TENSOR. All dyana::Tensor instances must be cleaned up before training on a new Datum. Otherwise severe memory leak will occur while training.");
      }

      // compute output on the item
      std::vector<float> ret = fn(numbered_item.second);

      // store the output into parameter collection
      output_container[numbered_item.first].set_value(ret);

      return 1;
    }

    virtual void SaveModel() {}

    std::function<std::vector<float>(const ITEM&)> fn;
    std::vector<dynet::Parameter>& output_container;
  };

  /**
   * perform array map, computed in parallel.
   *
   * note that when the callback function involves dyana computations, you cannot use
   * simple multithread based parallel computation, because
   * the underlying dynet library limits to only one CG per process (don't know why the heck it is)
   *
   * so a hacky way to perform parallel computation is to represent the parellel computation as "training a model",
   * in which each worker stores its computation results by "updating model parameters"
   *
   * because of such hacky implemenation, the callback function can only return vector<float>
   * in addition, the returned vector<float> for each item must have exactly the same size
   *
   * also because of such hacky implementation, you will see a message like "1       loss = 5 (0.000424s)"
   * printed to stderr. Please ignore that.
   *
   * \tparam ITEM item type
   * \param items list of items to compute
   * \param fn the callback function
   * \param ret_dim the size of each result
   * \param num_workers number of worker processes to spawn
   * \return a list of result
   */
  template<typename ITEM>
  std::vector<std::vector<float>>
  parallel_map(const std::vector<ITEM>& items, const std::function<std::vector<float>(const ITEM&)>& fn,
               unsigned ret_dim, unsigned num_workers) {

    if (items.empty()) return {};

    if (num_workers <= 1) {
      std::vector<std::vector<float>> ret;
      for (auto&& item:items) {
        ret.push_back(fn(item));
      }
      return ret;
    }

    dynet::ParameterCollection pc;
    std::vector<dynet::Parameter> output_container;
    for (unsigned i = 0; i < items.size(); i++) {
      output_container.push_back(pc.add_parameters({ret_dim}));
    }

    _parallel_map_learner<ITEM> _learner(items, output_container, fn, ret_dim, num_workers);

    std::vector<std::vector<float>> ret;
    for (auto&& p:output_container) {
      ret.push_back(dynet::as_vector(*p.values()));
    }

    return ret;
  }
}

#endif //DYANA_CPP_DYANA_PARALLEL_MAP_HPP
