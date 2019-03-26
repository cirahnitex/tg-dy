//
// Created by YAN Yuchen on 5/5/2018.
//

#ifndef DYANA_MP_TRAIN_HPP
#define DYANA_MP_TRAIN_HPP

#include <dynet/dynet.h>
#include <functional>
#include <dynet/mp.h>
#include "dyana_common.hpp"

namespace dynet {
  namespace mp {
    template<class D, class S>
    void run_single_process_patched(ILearner<D, S> *learner, Trainer *trainer, const std::vector<D> &train_data,
                                    const std::vector<D> &dev_data, unsigned num_iterations, unsigned dev_frequency,
                                    unsigned report_frequency, unsigned batch_size) {
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
            const D &datum = train_data[i];
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
              const D &datum = dev_data[i];
              S datum_loss = learner->LearnFromDatum(datum, false);
              dev_loss += datum_loss;
            }
            bool new_best = (first_dev_run || dev_loss < best_dev_loss);
            first_dev_run = false;
            double fractional_iter = iter + 1.0 * data_processed / train_indices.size();
            std::cerr << fractional_iter << "\t" << "dev loss = " << dev_loss << (new_best ? " (New best!)" : "")
                      << std::endl;
            if (stop_requested) {
              break;
            }
            if (new_best) {
              learner->SaveModel();
              best_dev_loss = dev_loss;
            }
          } else {
            learner->SaveModel();
          }

          begin = end;
        }
      }
    }
  }
}

namespace tg {
  namespace dyana {

    /**
     * internal helper class for function mp_train
     * \tparam DATUM represents a single data item
     */
    template<typename DATUM>
    class _mp_train_learner : private dynet::mp::ILearner<DATUM, float> {
    public:
      _mp_train_learner(unsigned num_workers, unsigned num_epochs, const std::vector<DATUM> &training_set,
                        const std::vector<DATUM> &dev_set, std::function<dyana::tensor(const DATUM &)> compute_loss,
                        std::function<void()> save)
        :
        compute_loss(compute_loss), save(save) {
        if (training_set.empty()) return;
        parameter::_force_garbage_collection();
        lookup_parameter::_force_garbage_collection();
        compute_loss(
          training_set[0]); // for its side-effect only. to ensure that all lazy-initialized layers has been initialized before going parallel
        if (num_workers <= 1) {
          dynet::mp::run_single_process_patched(this, dyana::_trainer(), training_set, dev_set, num_epochs,
                                                dev_set.size(),
                                                dev_set.empty() ? training_set.size() : dev_set.size(), 1);
        } else {
          dynet::mp::run_multi_process(num_workers, this, dyana::_trainer(), training_set, dev_set, num_epochs,
                                       dev_set.size(), dev_set.size());
        }

      }

      virtual ~_mp_train_learner() {}

    private:
      virtual float LearnFromDatum(const DATUM &datum, bool learn) {
        if (dyana::tensor::get_exprs_counter() != 0) {
          throw std::runtime_error(
            "NO GLOBAL TENSOR. All dyana::Tensor instances must be cleaned up before training on a new Datum. Otherwise severe memory leak will occur while training.");
        }
        dyana::tensor loss = compute_loss(datum);
        float ret = loss.as_scalar();
        if (learn) dyana::_cg().backward(loss);
        return ret;
        return 0;
      }

      virtual void SaveModel() { save(); }

      std::function<dyana::tensor(const DATUM &)> compute_loss;
      std::function<void()> save;
    };

    template<typename T0, typename T1>
    std::vector<std::pair<T0, T1>> zip(const std::vector<T0> &list0, const std::vector<T1> &list1) {
      if (list0.size() != list1.size()) throw std::runtime_error("zip: two lists must have the same length");
      std::vector<std::pair<T0, T1>> ret;
      ret.reserve(list0.size());
      for (size_t i = 0; i < list0.size(); ++i) {
        ret.push_back(std::make_pair(list0[i], list1[i]));
      }
      return ret;
    }

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
             const std::vector<DATUM> &dev_set, std::function<dyana::tensor(const DATUM &)> compute_loss,
             std::function<void()> save) {
      _mp_train_learner<DATUM>(_num_workers(), num_epochs, training_set, dev_set, compute_loss, save);
    }

    /**
     * data-parallel training
     * this function returns after all the data have finished training
     * \tparam EXAMPLE type of a single example. For example, an image
     * \tparam ORACLE type of a sincle oracle. For example, string that may contain "cat" or "dog"
     * \param num_epochs number of epochs
     * \param training_set the list of all training examples
     * \param training_oracles the list of all training oracles
     * \param dev_set the list of all dev examples
     * \param dev_oracles the list of all dev oracles
     * \param compute_loss a function that accepts an example and an oracle, returns the loss
     * \param save how to save your model
     */
    template<typename EXAMPLE, typename ORACLE>
    void fit(unsigned num_epochs, const std::vector<EXAMPLE> &training_set, const std::vector<ORACLE> &training_oracles,
             const std::vector<EXAMPLE> &dev_set, const std::vector<ORACLE> &dev_oracles,
             std::function<dyana::tensor(const EXAMPLE &, const ORACLE &)> compute_loss,
             std::function<void()> save) {
      _mp_train_learner<std::pair<EXAMPLE, ORACLE>>(_num_workers(), num_epochs,
                                                                 zip(training_set, training_oracles),
                                                                 zip(dev_set, dev_oracles), [&](
          const std::pair<EXAMPLE, ORACLE> datum) {
          return compute_loss(datum.first, datum.second);
        }, save);
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
             const std::vector<DATUM> &dev_set, std::function<dyana::tensor(const DATUM &)> compute_loss) {
      _mp_train_learner<DATUM>(_num_workers(), num_epochs, training_set, dev_set, compute_loss, []() {});
    }

    /**
     * data-parallel training
     * this function returns after all the data have finished training
     * \tparam EXAMPLE type of a single example. For example, an image
     * \tparam ORACLE type of a sincle oracle. For example, string that may contain "cat" or "dog"
     * \param num_epochs number of epochs
     * \param training_set the list of all training examples
     * \param training_oracles the list of all training oracles
     * \param dev_set the list of all dev examples
     * \param dev_oracles the list of all dev oracles
     * \param compute_loss a function that accepts an example and an oracle, returns the loss
     */
    template<typename EXAMPLE, typename ORACLE>
    void fit(unsigned num_epochs, const std::vector<EXAMPLE> &training_set, const std::vector<ORACLE> &training_oracles,
             const std::vector<EXAMPLE> &dev_set, const std::vector<ORACLE> &dev_oracles,
             std::function<dyana::tensor(const EXAMPLE &, const ORACLE &)> compute_loss) {
      _mp_train_learner<std::pair<EXAMPLE, ORACLE>>(_num_workers(), num_epochs,
                                                    zip(training_set, training_oracles),
                                                    zip(dev_set, dev_oracles), [&](
          const std::pair<EXAMPLE, ORACLE> datum) {
          return compute_loss(datum.first, datum.second);
        }, []() {});
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
             std::function<dyana::tensor(const DATUM &)> compute_loss,
             std::function<void()> save) {
      _mp_train_learner<DATUM>(_num_workers(), num_epochs, training_set, std::vector<DATUM>(), compute_loss,
                               save);
    }

    /**
     * data-parallel training
     * this function returns after all the data have finished training
     * \tparam EXAMPLE type of a single example. For example, an image
     * \tparam ORACLE type of a sincle oracle. For example, string that may contain "cat" or "dog"
     * \param num_epochs number of epochs
     * \param training_set the list of all training examples
     * \param training_oracles the list of all training oracles
     * \param compute_loss a function that accepts an example and an oracle, returns the loss
     * \param save how to save your model
     */
    template<typename EXAMPLE, typename ORACLE>
    void fit(unsigned num_epochs, const std::vector<EXAMPLE> &training_set, const std::vector<ORACLE> &training_oracles,
             std::function<dyana::tensor(const EXAMPLE &, const ORACLE &)> compute_loss, std::function<void()> save) {
      _mp_train_learner<std::pair<EXAMPLE, ORACLE>>(_num_workers(), num_epochs,
                                                    zip(training_set, training_oracles),
                                                    std::vector<std::pair<EXAMPLE, ORACLE>>(), [&](
          const std::pair<EXAMPLE, ORACLE> datum) {
          return compute_loss(datum.first, datum.second);
        }, save);
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
             std::function<dyana::tensor(const DATUM &)> compute_loss) {
      _mp_train_learner<DATUM>(_num_workers(), num_epochs, training_set, std::vector<DATUM>(), compute_loss,
                               []() {});
    }

    /**
     * data-parallel training
     * this function returns after all the data have finished training
     * \tparam EXAMPLE type of a single example. For example, an image
     * \tparam ORACLE type of a sincle oracle. For example, string that may contain "cat" or "dog"
     * \param num_epochs number of epochs
     * \param training_set the list of all training examples
     * \param training_oracles the list of all training oracles
     * \param compute_loss a function that accepts an example and an oracle, returns the loss
     */
    template<typename EXAMPLE, typename ORACLE>
    void fit(unsigned num_epochs, const std::vector<EXAMPLE> &training_set, const std::vector<ORACLE> &training_oracles,
             std::function<dyana::tensor(const EXAMPLE &, const ORACLE &)> compute_loss) {
      _mp_train_learner<std::pair<EXAMPLE, ORACLE>>(_num_workers(), num_epochs,
                                                    zip(training_set, training_oracles),
                                                    std::vector<std::pair<EXAMPLE, ORACLE>>(), [&](
          const std::pair<EXAMPLE, ORACLE> datum) {
          return compute_loss(datum.first, datum.second);
        }, []() {});
    }
  }
}

#endif //DYANA_MP_TRAIN_HPP
