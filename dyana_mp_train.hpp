//
// Created by YAN Yuchen on 5/5/2018.
//

#ifndef DYANA_MP_TRAIN_HPP
#define DYANA_MP_TRAIN_HPP

#include <dynet/dynet.h>
#include <functional>
#include <dynet/mp.h>
#include "dyana_common.hpp"
#include "dyana_serialization_helper.hpp"

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


namespace dyana {

  template<typename DATUM>
  class _mp_train_learner : private dynet::mp::ILearner<DATUM, float> {
  public:
    _mp_train_learner(unsigned num_workers, unsigned num_epochs, dynet::Trainer* trainer, const std::vector<DATUM> &training_set,
                      const std::vector<DATUM> &dev_set, std::function<dyana::tensor(const DATUM &)> compute_loss,
                      std::function<void()> save)
      :
      compute_loss(compute_loss), save(save) {
      if (training_set.empty()) return;
      _force_garbage_collection();
      compute_loss(
        training_set[0]); // for its side-effect only. to ensure that all lazy-initialized layers has been initialized before going parallel
      if (num_workers <= 1) {
        dynet::mp::run_single_process_patched(this, trainer, training_set, dev_set, num_epochs,
                                              dev_set.size(),
                                              dev_set.empty() ? training_set.size() : dev_set.size(), 1);
      } else {
        dynet::mp::run_multi_process(num_workers, this, trainer, training_set, dev_set, num_epochs,
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

  std::vector<std::tuple<>> zip() {
    return {};
  }

  template <typename ...T>
  std::vector<std::tuple<T...>> zip(std::vector<T>... lst)
  {
    const auto size = std::min({lst.size()...});
    std::vector<std::tuple<T...>>  result;
    result.reserve(size);
    for (unsigned i = 0; i < size; i++) {
      result.emplace_back(std::move(lst[i])...);
    }
    return result;
  }

  class trainer_base {
  public:

    /**
     * number of threads to spawn to train in parallel
     */
    unsigned num_workers{1};

    /**
     * number of epochs to train
     */
    unsigned num_epochs{1};
  protected:
    virtual dynet::Trainer* get_dynet_trainer_p() = 0;
  private:
    template<typename MODEL, typename D0>
    void _fit_with_dev_set_helper(MODEL &model, const std::function<void()> &save_behavior, D0&& trainingset_p0, D0&& devset_p0) {
      using V0 = typename std::decay_t<D0>::value_type;
      using datum_type = std::tuple<V0>;
      auto training_set_tuple = zip(trainingset_p0);
      auto dev_set_tuple = zip(devset_p0);

      auto compute_loss = [&](auto&&... args) {
        return model.compute_loss(args...);
      };
      auto compute_loss_tuple = [&](auto&& args_pack) {
        return std::apply(compute_loss, args_pack);
      };
      _mp_train_learner<datum_type>(num_workers, num_epochs, get_dynet_trainer_p(), training_set_tuple, dev_set_tuple, compute_loss_tuple, save_behavior);
    }
    template<typename MODEL, typename D0, typename D1>
    void _fit_with_dev_set_helper(MODEL &model,const std::function<void()> &save_behavior, D0&& trainingset_p0, D1&& trainingset_p1, D0&& devset_p0, D1&& devset_p1) {
      using V0 = typename std::decay_t<D0>::value_type;
      using V1 = typename std::decay_t<D1>::value_type;
      using datum_type = std::tuple<V0, V1>;
      auto training_set_tuple = zip(trainingset_p0, trainingset_p1);
      auto dev_set_tuple = zip(devset_p0, devset_p1);

      auto compute_loss = [&](auto&&... args) {
        return model.compute_loss(args...);
      };
      auto compute_loss_tuple = [&](auto&& args_pack) {
        return std::apply(compute_loss, args_pack);
      };

      _mp_train_learner<datum_type>(num_workers, num_epochs, get_dynet_trainer_p(), training_set_tuple, dev_set_tuple, compute_loss_tuple, save_behavior);
    }
    template<typename MODEL, typename D0, typename D1, typename D2>
    void _fit_with_dev_set_helper(MODEL &model, const std::function<void()> &save_behavior,
                                  D0&& trainingset_p0, D1&& trainingset_p1, D2&& trainingset_p2,
                                  D0&& devset_p0, D1&& devset_p1, D2&& devset_p2) {
      using V0 = typename std::decay_t<D0>::value_type;
      using V1 = typename std::decay_t<D1>::value_type;
      using V2 = typename std::decay_t<D2>::value_type;
      using datum_type = std::tuple<V0, V1, V2>;
      auto training_set_tuple = zip(trainingset_p0, trainingset_p1, trainingset_p2);
      auto dev_set_tuple = zip(devset_p0, devset_p1, devset_p2);

      auto compute_loss = [&](auto&&... args) {
        return model.compute_loss(args...);
      };
      auto compute_loss_tuple = [&](auto&& args_pack) {
        return std::apply(compute_loss, args_pack);
      };

      _mp_train_learner<datum_type>(num_workers, num_epochs, get_dynet_trainer_p(), training_set_tuple, dev_set_tuple, compute_loss_tuple, save_behavior);
    }
    template<typename MODEL, typename D0, typename D1, typename D2, typename D3>
    void _fit_with_dev_set_helper(MODEL &model, const std::function<void()> &save_behavior,
                                  D0&& trainingset_p0, D1&& trainingset_p1, D2&& trainingset_p2, D3&& trainingset_p3,
                                  D0&& devset_p0, D1&& devset_p1, D2&& devset_p2, D3&& devset_p3) {
      using V0 = typename std::decay_t<D0>::value_type;
      using V1 = typename std::decay_t<D1>::value_type;
      using V2 = typename std::decay_t<D2>::value_type;
      using V3 = typename std::decay_t<D3>::value_type;
      using datum_type = std::tuple<V0, V1, V2, V3>;
      auto training_set_tuple = zip(trainingset_p0, trainingset_p1, trainingset_p2, trainingset_p3);
      auto dev_set_tuple = zip(devset_p0, devset_p1, devset_p2, devset_p3);

      auto compute_loss = [&](auto&&... args) {
        return model.compute_loss(args...);
      };
      auto compute_loss_tuple = [&](auto&& args_pack) {
        return std::apply(compute_loss, args_pack);
      };

      _mp_train_learner<datum_type>(num_workers, num_epochs, get_dynet_trainer_p(), training_set_tuple, dev_set_tuple, compute_loss_tuple, save_behavior);
    }
    template<typename MODEL, typename D0, typename D1, typename D2, typename D3, typename D4>
    void _fit_with_dev_set_helper(MODEL &model, const std::function<void()> &save_behavior,
                                  D0&& trainingset_p0, D1&& trainingset_p1, D2&& trainingset_p2, D3&& trainingset_p3,
                                  D4&& trainingset_p4,
                                  D0&& devset_p0, D1&& devset_p1, D2&& devset_p2, D3&& devset_p3,
                                  D4&& devset_p4) {
      using V0 = typename std::decay_t<D0>::value_type;
      using V1 = typename std::decay_t<D1>::value_type;
      using V2 = typename std::decay_t<D2>::value_type;
      using V3 = typename std::decay_t<D3>::value_type;
      using V4 = typename std::decay_t<D4>::value_type;
      using datum_type = std::tuple<V0, V1, V2, V3, V4>;
      auto training_set_tuple = zip(trainingset_p0, trainingset_p1, trainingset_p2, trainingset_p3, trainingset_p4);
      auto dev_set_tuple = zip(devset_p0, devset_p1, devset_p2, devset_p3, devset_p4);

      auto compute_loss = [&](auto&&... args) {
        return model.compute_loss(args...);
      };
      auto compute_loss_tuple = [&](auto&& args_pack) {
        return std::apply(compute_loss, args_pack);
      };

      _mp_train_learner<datum_type>(num_workers, num_epochs, get_dynet_trainer_p(), training_set_tuple, dev_set_tuple, compute_loss_tuple, save_behavior);
    }
    template<typename MODEL, typename D0, typename D1, typename D2, typename D3, typename D4, typename D5>
    void _fit_with_dev_set_helper(MODEL &model, const std::function<void()> &save_behavior,
                                  D0&& trainingset_p0, D1&& trainingset_p1, D2&& trainingset_p2, D3&& trainingset_p3,
                                  D4&& trainingset_p4, D5&& trainingset_p5,
                                  D0&& devset_p0, D1&& devset_p1, D2&& devset_p2, D3&& devset_p3,
                                  D4&& devset_p4, D5&& devset_p5) {
      using V0 = typename std::decay_t<D0>::value_type;
      using V1 = typename std::decay_t<D1>::value_type;
      using V2 = typename std::decay_t<D2>::value_type;
      using V3 = typename std::decay_t<D3>::value_type;
      using V4 = typename std::decay_t<D4>::value_type;
      using V5 = typename std::decay_t<D5>::value_type;
      using datum_type = std::tuple<V0, V1, V2, V3, V4, V5>;
      auto training_set_tuple = zip(trainingset_p0, trainingset_p1, trainingset_p2, trainingset_p3, trainingset_p4, trainingset_p5);
      auto dev_set_tuple = zip(devset_p0, devset_p1, devset_p2, devset_p3, devset_p4, devset_p5);

      auto compute_loss = [&](auto&&... args) {
        return model.compute_loss(args...);
      };
      auto compute_loss_tuple = [&](auto&& args_pack) {
        return std::apply(compute_loss, args_pack);
      };

      _mp_train_learner<datum_type>(num_workers, num_epochs, get_dynet_trainer_p(), training_set_tuple, dev_set_tuple, compute_loss_tuple, save_behavior);
    }
    template<typename MODEL, typename D0, typename D1, typename D2, typename D3, typename D4, typename D5, typename D6>
    void _fit_with_dev_set_helper(MODEL &model, const std::function<void()> &save_behavior,
                                  D0&& trainingset_p0, D1&& trainingset_p1, D2&& trainingset_p2, D3&& trainingset_p3,
                                  D4&& trainingset_p4, D5&& trainingset_p5, D6&& trainingset_p6,
                                  D0&& devset_p0, D1&& devset_p1, D2&& devset_p2, D3&& devset_p3,
                                  D4&& devset_p4, D5&& devset_p5, D6&& devset_p6) {
      using V0 = typename std::decay_t<D0>::value_type;
      using V1 = typename std::decay_t<D1>::value_type;
      using V2 = typename std::decay_t<D2>::value_type;
      using V3 = typename std::decay_t<D3>::value_type;
      using V4 = typename std::decay_t<D4>::value_type;
      using V5 = typename std::decay_t<D5>::value_type;
      using V6 = typename std::decay_t<D6>::value_type;
      using datum_type = std::tuple<V0, V1, V2, V3, V4, V5, V6>;
      auto training_set_tuple = zip(
        trainingset_p0, trainingset_p1, trainingset_p2, trainingset_p3,
        trainingset_p4, trainingset_p5, trainingset_p6);
      auto dev_set_tuple = zip(
        devset_p0, devset_p1, devset_p2, devset_p3,
        devset_p4, devset_p5, devset_p6);

      auto compute_loss = [&](auto&&... args) {
        return model.compute_loss(args...);
      };
      auto compute_loss_tuple = [&](auto&& args_pack) {
        return std::apply(compute_loss, args_pack);
      };

      _mp_train_learner<datum_type>(num_workers, num_epochs, get_dynet_trainer_p(), training_set_tuple, dev_set_tuple, compute_loss_tuple, save_behavior);
    }
    template<typename MODEL, typename D0, typename D1, typename D2, typename D3, typename D4, typename D5, typename D6, typename D7>
    void _fit_with_dev_set_helper(MODEL &model, const std::function<void()> &save_behavior,
                                  D0&& trainingset_p0, D1&& trainingset_p1, D2&& trainingset_p2, D3&& trainingset_p3,
                                  D4&& trainingset_p4, D5&& trainingset_p5, D6&& trainingset_p6, D7&& trainingset_p7,
                                  D0&& devset_p0, D1&& devset_p1, D2&& devset_p2, D3&& devset_p3,
                                  D4&& devset_p4, D5&& devset_p5, D6&& devset_p6, D7&& devset_p7) {
      using V0 = typename std::decay_t<D0>::value_type;
      using V1 = typename std::decay_t<D1>::value_type;
      using V2 = typename std::decay_t<D2>::value_type;
      using V3 = typename std::decay_t<D3>::value_type;
      using V4 = typename std::decay_t<D4>::value_type;
      using V5 = typename std::decay_t<D5>::value_type;
      using V6 = typename std::decay_t<D6>::value_type;
      using V7 = typename std::decay_t<D7>::value_type;
      using datum_type = std::tuple<V0, V1, V2, V3, V4, V5, V6, V7>;
      auto training_set_tuple = zip(
        trainingset_p0, trainingset_p1, trainingset_p2, trainingset_p3,
        trainingset_p4, trainingset_p5, trainingset_p6, trainingset_p7);
      auto dev_set_tuple = zip(
        devset_p0, devset_p1, devset_p2, devset_p3,
        devset_p4, devset_p5, devset_p6, devset_p7);

      auto compute_loss = [&](auto&&... args) {
        return model.compute_loss(args...);
      };
      auto compute_loss_tuple = [&](auto&& args_pack) {
        return std::apply(compute_loss, args_pack);
      };

      _mp_train_learner<datum_type>(num_workers, num_epochs, get_dynet_trainer_p(), training_set_tuple, dev_set_tuple, compute_loss_tuple, save_behavior);
    }
  public:

    /**
     * train your model on a training set. For each epochs, it reports the score on the training set (to standard error)
     * \param model your model. must have a compute_loss function that returns a loss value as tensor
     * \param training_set the entire training set as N vector of values.
     * where N is the number of parameters the compuse_loss function takes.
     * the value_type of vector #0 should match compute_loss function's expected parameter type #0
     * the value_type of vector #1 should match compute_loss function's expected parameter type #1
     * ...
     * the value_type of vector #(N-1) should match compute_loss function's expected parameter type #(N-1)
     */
    template<typename MODEL, typename ...DATASET>
    void train(MODEL &model, DATASET ...training_set) {
      auto training_set_tuple = zip(training_set...);

      auto compute_loss = [&](auto&&... args) {
        return model.compute_loss(args...);
      };
      auto compute_loss_tuple = [&](auto&& args_pack) {
        return std::apply(compute_loss, args_pack);
      };

      using DATASET_TUPLE = std::tuple<typename DATASET::value_type...>;
      _mp_train_learner<DATASET_TUPLE>(num_workers, num_epochs, get_dynet_trainer_p(), training_set_tuple, std::vector<DATASET_TUPLE>{}, compute_loss_tuple, [](){});
    }

    /**
     * train your model on a training set. while training, also report scores on a dev set (to standard error).
     * \param model your model. must have a compute_loss function that returns a loss value as tensor
     * \param training_set_and_dev_set the entire training set as N vector of values,
     * followed by the entire dev set as N vector of values.
     * where N is the number of parameters the compuse_loss function takes.
     * the value_type of vector #0 should match compute_loss function's expected parameter type #0
     * the value_type of vector #1 should match compute_loss function's expected parameter type #1
     * ...
     * the value_type of vector #(N-1) should match compute_loss function's expected parameter type #(N-1)
     * (end of training set)
     * the value_type of vector #(N) should match compute_loss function's expected parameter type #0
     * the value_type of vector #(N+1) should match compute_loss function's expected parameter type #1
     * ...
     * the value_type of vector #(2N-1) should match compute_loss function's expected parameter type #(N-1)
     * (end of dev set)
     */
    template<typename MODEL, typename ...Args>
    void train_reporting_dev_score(MODEL &model, Args... training_set_and_dev_set) {
      _fit_with_dev_set_helper(model, [](){}, training_set_and_dev_set...);
    }

    /**
     * train your model on a training set. while training, also report scores on a dev set.
     * Whenever the model reaches a new best score, the model will be saved to a file.
     * when saving the model, cereal::BinaryOutputArchive is used under the hood.
     * \param model your model. must have a compute_loss function that returns a loss value as tensor
     * \param file_path_to_save_to path to the file where your model is saved to
     * \param training_set_and_dev_set the entire training set as N vector of values,
     * followed by the entire dev set as N vector of values.
     * where N is the number of parameters the compuse_loss function takes.
     * the value_type of vector #0 should match compute_loss function's expected parameter type #0
     * the value_type of vector #1 should match compute_loss function's expected parameter type #1
     * ...
     * the value_type of vector #(N-1) should match compute_loss function's expected parameter type #(N-1)
     * (end of training set)
     * the value_type of vector #(N) should match compute_loss function's expected parameter type #0
     * the value_type of vector #(N+1) should match compute_loss function's expected parameter type #1
     * ...
     * the value_type of vector #(2N-1) should match compute_loss function's expected parameter type #(N-1)
     * (end of dev set)
     */
    template<typename MODEL, typename ...Args>
    void train_reporting_dev_score_save_best(MODEL &model, const std::string &file_path_to_save_to, Args... training_set_and_dev_set) {

      auto save_behavior = [&]() {
        std::ofstream ofs(file_path_to_save_to);
        if(!ofs.is_open()) throw std::runtime_error("could not open file for output: "+file_path_to_save_to);
        cereal::BinaryOutputArchive oa(ofs);
        oa << model;
      };
      _fit_with_dev_set_helper(model, save_behavior, training_set_and_dev_set...);
    }
  };

  /**
   * \ingroup optimizers
   *
   * \brief Stochastic gradient descent trainer
   * \details This trainer performs stochastic gradient descent, the goto optimization procedure for neural networks.
   * In the standard setting, the learning rate at epoch \f$t\f$ is \f$\eta_t=\frac{\eta_0}{1+\eta_{\mathrm{decay}}t}\f$
   *
   * Reference : [reference needed](ref.need.ed)
   *
   */
  class simple_sgd_trainer :public trainer_base {
    dynet::SimpleSGDTrainer trainer_m;
  protected:
    dynet::Trainer* get_dynet_trainer_p() override {
      return &trainer_m;
    };
  public:

    /**
     * \brief Constructor
     *
     * \param m ParameterCollection to be trained
     * \param learning_rate Initial learning rate
     */
    explicit simple_sgd_trainer(float learning_rate = 0.1):trainer_m(*_pc(), learning_rate) {}
  };

  /**
   * \ingroup optimizers
   *
   * \brief Adam optimizer
   * \details The Adam optimizer is similar to RMSProp but uses unbiased estimates
   * of the first and second moments of the gradient
   *
   * Reference : [Adam: A Method for Stochastic Optimization](https://arxiv.org/pdf/1412.6980v8)
   *
   */
  class adam_trainer :public trainer_base {
    dynet::AdamTrainer trainer_m;
  protected:
    dynet::Trainer* get_dynet_trainer_p() override {
      return &trainer_m;
    };
  public:
    /**
     * \brief Constructor
     *
     * \param learning_rate Initial learning rate
     * \param beta_1 Moving average parameter for the mean
     * \param beta_2 Moving average parameter for the variance
     * \param eps Bias parameter \f$\epsilon\f$
     */
    explicit adam_trainer(float learning_rate = 0.001, float beta_1 = 0.9, float beta_2 = 0.999, float eps = 1e-8):trainer_m(*_pc(), learning_rate, beta_1, beta_2, eps){}
  };

}


#endif //DYANA_MP_TRAIN_HPP
