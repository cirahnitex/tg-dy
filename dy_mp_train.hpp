//
// Created by YAN Yuchen on 5/5/2018.
//

#ifndef DYNET_WRAPPER_DY_MP_TRAIN_HPP
#define DYNET_WRAPPER_DY_MP_TRAIN_HPP

#include <dynet/dynet.h>
#include <functional>
#include <dynet/mp.h>
#include "dy_common.hpp"

namespace dynet {
  namespace mp {
    template<class D, class S>
    void run_single_process_patched(ILearner<D, S>* learner, Trainer* trainer, const std::vector<D>& train_data,
                            const std::vector<D>& dev_data, unsigned num_iterations, unsigned dev_frequency, unsigned report_frequency, unsigned batch_size) {
      std::vector<unsigned> train_indices(train_data.size());
      std::iota(train_indices.begin(), train_indices.end(), 0);

      std::vector<unsigned> dev_indices(dev_data.size());
      std::iota(dev_indices.begin(), dev_indices.end(), 0);

      S best_dev_loss = S();
      bool first_dev_run = true;
      unsigned batch_counter = 0;
      for (unsigned iter = 0; iter < num_iterations && !stop_requested; ++iter) {
        // Shuffle the training data indices
        std::shuffle(train_indices.begin(), train_indices.end(), *rndeng);

        S train_loss = S();

        unsigned data_processed = 0;
        unsigned data_until_report = report_frequency;
        std::vector<unsigned>::iterator begin = train_indices.begin();
        while (begin != train_indices.end()) {
          std::vector<unsigned>::iterator end = (dev_frequency > 0) ? begin + dev_frequency : train_indices.end();
          if (end > train_indices.end()) {
            end = train_indices.end();
          }
          S batch_loss = S();
          for (auto it = begin; it != end; ++it) {
            unsigned i = *it;
            DYNET_ASSERT(i < train_data.size(), "Out-of-bounds ID in train set for multiprocessing");
            const D& datum = train_data[i];
            S datum_loss = learner->LearnFromDatum(datum, true);
            batch_loss += datum_loss;
            train_loss += datum_loss;
            if (++batch_counter == batch_size) {
              // TODO: The scaling was originally this
              // trainer->update(1.0 / batch_size);
              trainer->update();
              batch_counter = 0;
            }
            data_processed++;

            if (--data_until_report == 0) {
              data_until_report = report_frequency;
              double fractional_iter = iter + 1.0 * data_processed / train_indices.size();
              std::cerr << fractional_iter << "\t" << "loss = " << batch_loss << std::endl;
              batch_loss = S();
            }
          }

          if (stop_requested) {
            break;
          }
//          if(true) {
          if (dev_data.size() > 0) {
            S dev_loss = S();
            for (auto it = dev_indices.begin(); it != dev_indices.end(); ++it) {
              unsigned i = *it;
              DYNET_ASSERT(i < dev_data.size(), "Out-of-bounds ID in dev set for multiprocessing");
              const D& datum = dev_data[i];
              S datum_loss = learner->LearnFromDatum(datum, false);
              dev_loss += datum_loss;
            }
            bool new_best = (first_dev_run || dev_loss < best_dev_loss);
            first_dev_run = false;
            double fractional_iter = iter + 1.0 * data_processed / train_indices.size();
            std::cerr << fractional_iter << "\t" << "dev loss = " << dev_loss << (new_best ? " (New best!)" : "") << std::endl;
            if (stop_requested) {
              break;
            }
            if (new_best) {
              learner->SaveModel();
              best_dev_loss = dev_loss;
            }
          }
          else {
            learner->SaveModel();
          }

          begin = end;
        }
      }
    }
  }
}

namespace tg {
  namespace dy {

    /**
     * internal helper class for function mp_train
     * \tparam DATUM represents a single data item
     */
    template<typename DATUM>
    class _mp_train_learner : private dynet::mp::ILearner<DATUM, float> {
    public:
      _mp_train_learner(unsigned num_workers, unsigned num_epochs, const std::vector<DATUM> &training_set,
                        const std::vector<DATUM> &dev_set, std::function<dy::tensor(const DATUM &)> compute_loss,
                        std::function<void()> save)
        :
        compute_loss(compute_loss), save(save) {
        if (training_set.empty()) return;
        compute_loss(
          training_set[0]); // for its side-effect only. to ensure that all lazy-initialized layers has been initialized before going parallel
        if (num_workers <= 1) {
          dynet::mp::run_single_process_patched(this, dy::_trainer(), training_set, dev_set, num_epochs, dev_set.size(),
                                        dev_set.empty()?training_set.size():dev_set.size(), 1);
        } else {
          dynet::mp::run_multi_process(num_workers, this, dy::_trainer(), training_set, dev_set, num_epochs,
                                       dev_set.size(), dev_set.size());
        }

      }

      virtual ~_mp_train_learner() {}

    private:
      virtual float LearnFromDatum(const DATUM &datum, bool learn) {
        if (dy::tensor::get_exprs_counter() != 0) {
          throw std::runtime_error(
            "NO GLOBAL TENSOR. All dy::Tensor instances must be cleaned up before training on a new Datum. Otherwise severe memory leak will occur while training.");
        }
        dy::tensor loss = compute_loss(datum);
        float ret = loss.as_scalar();
        if (learn) dy::_cg().backward(loss);
        return ret;
        return 0;
      }

      virtual void SaveModel() { save(); }

      std::function<dy::tensor(const DATUM &)> compute_loss;
      std::function<void()> save;
    };

    /**
     * data-parallel training.
     * this function returns after all the data have finished training
     * \tparam DATUM type of a single datum
     * \param num_epochs number of epochs
     * \param training_set all training data
     * \param dev_set all dev data
     * \param compute_loss a function that accepts a datum and returns the loss
     * \param on_save how to save your model
     */
    template<typename DATUM>
    void fit(unsigned num_epochs, const std::vector<DATUM> &training_set,
             const std::vector<DATUM> &dev_set, std::function<dy::tensor(const DATUM &)> compute_loss,
             std::function<void()> save) {
      _mp_train_learner<DATUM>(_num_workers(), num_epochs, training_set, dev_set, compute_loss, save);
    }

    /**
     * data-parallel training.
     * this function returns after all the data have finished training
     * \tparam DATUM type of a single datum
     * \param num_epochs number of epochs
     * \param training_set all training data
     * \param dev_set all dev data
     * \param compute_loss a function that accepts a datum and returns the loss
     */
    template<typename DATUM>
    void fit(unsigned num_epochs, const std::vector<DATUM> &training_set,
             const std::vector<DATUM> &dev_set, std::function<dy::tensor(const DATUM &)> compute_loss) {
      _mp_train_learner<DATUM>(_num_workers(), num_epochs, training_set, dev_set, compute_loss, []() {});
    }

    /**
     * data-parallel training.
     * this function returns after all the data have finished training
     * \tparam DATUM type of a single datum
     * \param num_epochs number of epochs
     * \param training_set all training data
     * \param compute_loss a function that accepts a datum and returns the loss
     * \param on_save how to save your model
     */
    template<typename DATUM>
    void fit(unsigned num_epochs, const std::vector<DATUM> &training_set,
             std::function<dy::tensor(const DATUM &)> compute_loss,
             std::function<void()> save) {
      _mp_train_learner<DATUM>(_num_workers(), num_epochs, training_set, std::vector<DATUM>(), compute_loss,
                               save);
    }

    /**
     * data-parallel training.
     * this function returns after all the data have finished training
     * \tparam DATUM type of a single datum
     * \param num_epochs number of epochs
     * \param training_set all training data
     * \param compute_loss a function that accepts a datum and returns the loss
     */
    template<typename DATUM>
    void fit(unsigned num_epochs, const std::vector<DATUM> &training_set,
             std::function<dy::tensor(const DATUM &)> compute_loss) {
      _mp_train_learner<DATUM>(_num_workers(), num_epochs, training_set, std::vector<DATUM>(), compute_loss,
                               []() {});
    }
  }
}

#endif //DYNET_WRAPPER_DY_MP_TRAIN_HPP
