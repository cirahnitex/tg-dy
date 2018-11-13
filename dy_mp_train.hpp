//
// Created by YAN Yuchen on 5/5/2018.
//

#ifndef DYNET_WRAPPER_DY_MP_TRAIN_HPP
#define DYNET_WRAPPER_DY_MP_TRAIN_HPP

#include <dynet/dynet.h>
#include <functional>
#include <dynet/mp.h>
#include "dy_common.hpp"

namespace tg {
  namespace dy {
    /**
     * internal helper class for function mp_train
     * \tparam DATUM represents a single data item
     */
    template<typename DATUM>
  class _mp_train_learner :private dynet::mp::ILearner<DATUM, unsigned> {
    public:
      _mp_train_learner(unsigned num_workers, const std::vector<DATUM> &data, std::function<dynet::Expression(const DATUM &)> compute_loss, std::function<void(const std::exception&, const DATUM&)> on_error) :
          compute_loss(compute_loss), on_error(on_error) {
        if(data.empty()) return;
        compute_loss(data[0]); // for its side-effect only. to ensure that all lazy-initialized layers has been initialized before going parallel
        if(num_workers > 1) {
          dynet::mp::run_mp_minibatch(num_workers, this, data);
        }
        else {
          for(const auto& datum:data) {
            LearnFromDatum(datum, true);
          }
        }
      }

      virtual ~_mp_train_learner() {}

    private:
      virtual unsigned LearnFromDatum(const DATUM &datum, bool learn) {
        dy::renew_cg();
        try {
          dynet::Expression loss = compute_loss(datum);
          dy::train_on_loss(loss);
        }
        catch (const std::exception &e)
        {
          on_error(e, datum);
        }
        return 0;
      }

      virtual void SaveModel() {}

      std::function<dynet::Expression(const DATUM &)> compute_loss;
      std::function<void(const std::exception&, const DATUM&)> on_error;
    };

    /**
     * data-parallel training.
     * this function returns after all the data have finished training
     * \tparam DATUM represents a single piece of data
     * \param num_workers number of parallel processes. 1 means single process.
     * \param data the list of data
     * \param compute_loss how to compute loss given a datum
     * \param onerror what to do when an error occured on a datum
     */
    template<typename DATUM>
    void mp_train(unsigned num_workers, const std::vector<DATUM> &data, std::function<dynet::Expression(const DATUM &)> compute_loss, std::function<void(const std::exception&, const DATUM&)> on_error) {
      _mp_train_learner<DATUM>(num_workers, data, compute_loss, on_error);
    }
    /**
     * data-parallel training.
     * this function returns after all the data have finished training
     * \tparam DATUM represents a single piece of data
     * \param num_workers number of parallel processes. 1 means single process.
     * \param data the list of data
     * \param compute_loss how to compute loss given a datum
     */
    template<typename DATUM>
    void mp_train(unsigned num_workers, const std::vector<DATUM> &data, std::function<dynet::Expression(const DATUM &)> compute_loss) {
      _mp_train_learner<DATUM>(num_workers, data, compute_loss, [](const std::exception& e, const DATUM& d){
        std::cerr << "skipped datum because of exception" << std::endl;
        std::cerr << e.what() << std::endl;
      });
    }
  }
}

#endif //DYNET_WRAPPER_DY_MP_TRAIN_HPP
