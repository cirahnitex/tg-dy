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
  class _mp_train_learner :private dynet::mp::ILearner<DATUM, float> {
    public:
      _mp_train_learner(unsigned num_workers, unsigned num_epoches, const std::vector<DATUM> &training_set,  const std::vector<DATUM> &dev_set, std::function<dy::Tensor(const DATUM &)> compute_loss, std::function<void(const std::exception&, const DATUM&)> on_error, std::function<void()> save) :
          compute_loss(compute_loss), on_error(on_error), save(save) {
        if(training_set.empty()) return;
        compute_loss(training_set[0]); // for its side-effect only. to ensure that all lazy-initialized layers has been initialized before going parallel
        if(num_workers<=1) {
          dynet::mp::run_single_process(this, &dy::trainer(), training_set, dev_set,num_epoches,dev_set.size(),dev_set.size(), 1);
        }
        else {
          dynet::mp::run_multi_process(num_workers, this, &dy::trainer(), training_set, dev_set,num_epoches,dev_set.size(),dev_set.size());
        }

      }

      virtual ~_mp_train_learner() {}

    private:
      virtual float LearnFromDatum(const DATUM &datum, bool learn) {
        if(dy::Tensor::get_exprs_counter()!=0) {
          throw std::runtime_error("NO GLOBAL TENSOR. All dy::Tensor instances must be cleaned up before training on a new Datum. Otherwise severe memory leak will occur while training.");
        }
        try {
          dy::Tensor loss = compute_loss(datum);
          float ret = loss.as_scalar();
          if(learn) dy::cg().backward(loss);
          return ret;
        }
        catch (const std::exception &e)
        {
          on_error(e, datum);
        }
        return 0;
      }

      virtual void SaveModel() {save();}

      std::function<dy::Tensor(const DATUM &)> compute_loss;
      std::function<void(const std::exception&, const DATUM&)> on_error;
      std::function<void()> save;
    };

    /**
     * data-parallel training.
     * this function returns after all the data have finished training
     * \tparam DATUM type of a single datum
     * \param num_workers number of parallel processes. 1 means single process.
     * \param num_epoches number of epoches
     * \param training_set all training data
     * \param dev_set all dev data
     * \param compute_loss a function that accepts a datum and returns the loss
     * \param on_error how to report an exception
     * \param on_save how to save your model
     */
    template<typename DATUM>
    void fit(unsigned num_workers, unsigned num_epoches, const std::vector<DATUM> &training_set,
             const std::vector<DATUM> &dev_set, std::function<dy::Tensor(const DATUM &)> compute_loss,
             std::function<void(const std::exception &, const DATUM &)> on_error, std::function<void()> save) {
      _mp_train_learner<DATUM>(num_workers, num_epoches, training_set, dev_set, compute_loss, on_error, save);
    }

    /**
     * data-parallel training.
     * this function returns after all the data have finished training
     * \tparam DATUM type of a single datum
     * \param num_workers number of parallel processes. 1 means single process.
     * \param num_epoches number of epoches
     * \param training_set all training data
     * \param dev_set all dev data
     * \param compute_loss a function that accepts a datum and returns the loss
     * \param on_error how to report an exception
     */
    template<typename DATUM>
    void fit(unsigned num_workers, unsigned num_epoches, const std::vector<DATUM> &training_set,
             const std::vector<DATUM> &dev_set, std::function<dy::Tensor(const DATUM &)> compute_loss,
             std::function<void(const std::exception &, const DATUM &)> on_error) {
      _mp_train_learner<DATUM>(num_workers, num_epoches, training_set, dev_set, compute_loss, on_error, [](){});
    }

    /**
     * data-parallel training.
     * this function returns after all the data have finished training
     * \tparam DATUM type of a single datum
     * \param num_workers number of parallel processes. 1 means single process.
     * \param num_epoches number of epoches
     * \param training_set all training data
     * \param dev_set all dev data
     * \param compute_loss a function that accepts a datum and returns the loss
     */
    template<typename DATUM>
    void fit(unsigned num_workers, unsigned num_epoches, const std::vector<DATUM> &training_set,
             const std::vector<DATUM> &dev_set, std::function<dy::Tensor(const DATUM &)> compute_loss) {
      _mp_train_learner<DATUM>(num_workers, num_epoches, training_set, dev_set, compute_loss, [](const std::exception& e, const DATUM& d){
        std::cerr << "skipped datum because of exception" << std::endl;
        std::cerr << e.what() << std::endl;
      }, [](){});
    }
  }
}

#endif //DYNET_WRAPPER_DY_MP_TRAIN_HPP
