//
// Created by YAN Yuchen on 5/5/2018.
//

#ifndef DYANA_MP_TRAIN_HPP
#define DYANA_MP_TRAIN_HPP

#include <dynet/dynet.h>
#include <functional>
#include <utility>
#include <dynet/mp.h>
#include "dyana_common.hpp"
#include "dyana_serialization_helper.hpp"
#include "dyana_event_emitter.hpp"
#include "dyana_timer.hpp"
#include "dyana_guard.macro.hpp"

namespace dynet {
  namespace mp {
    template<class D, class S>
    void run_single_process_hacked(ILearner<D, S>* learner, Trainer* trainer, const std::vector<D>& train_data,
                            const std::vector<D>& dev_data, unsigned num_iterations, float num_reports_per_epoch) {
      unsigned report_frequency = train_data.size() / num_reports_per_epoch;
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
          std::vector<unsigned>::iterator end = begin + report_frequency;
          if (end > train_indices.end()) {
            end = train_indices.end();
          }
          S batch_loss = S();
          std::chrono::steady_clock::time_point start_time = std::chrono::steady_clock::now();
          for (auto it = begin; it != end; ++it) {
            unsigned i = *it;
            DYNET_ASSERT(i < train_data.size(), "Out-of-bounds ID in train set for multiprocessing");
            const D& datum = train_data[i];
            S datum_loss = learner->LearnFromDatum(datum, true);
            batch_loss += datum_loss;
            train_loss += datum_loss;
            if (++batch_counter == /*batch_size*/1) {
              // TODO: The scaling was originally this
              // trainer->update(1.0 / batch_size);
              trainer->update();
              batch_counter = 0;
            }
            data_processed++;

            if (--data_until_report == 0) {
              std::chrono::steady_clock::time_point end_time = std::chrono::steady_clock::now();
              double seconds_elapsed = std::chrono::duration_cast<std::chrono::microseconds>(end_time - start_time).count() / 1000000.0;
              start_time = end_time;

              data_until_report = report_frequency;
              double fractional_iter = iter + 1.0 * data_processed / train_indices.size();
              std::cerr << fractional_iter << "\t" << "loss = " << batch_loss << " (" << seconds_elapsed << "s)" << std::endl;
              batch_loss = S();
            }
          }

          if (stop_requested) {
            break;
          }

          begin = end;
        }

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
          std::cerr << iter + 1.0 << "\t" << "dev loss = " << dev_loss << (new_best ? " (New best!)" : "") << std::endl;
          if (new_best) {
            learner->SaveModel();
            best_dev_loss = dev_loss;
          }
        }
      }
    }

    template<class D, class S>
    void run_parent_hacked(const std::vector<D>& train_data, const std::vector<D>& dev_data, ILearner<D, S>* learner,
                    std::vector<Workload>& workloads, unsigned num_iterations, float num_reports_per_epoch) {
      unsigned report_frequency = train_data.size() / num_reports_per_epoch;
      const unsigned num_children = workloads.size();
      boost::interprocess::message_queue mq(boost::interprocess::create_only, queue_name.c_str(), 10000, sizeof(unsigned));
      std::vector<unsigned> train_indices(train_data.size());
      std::iota(train_indices.begin(), train_indices.end(), 0);

      std::vector<unsigned> dev_indices(dev_data.size());
      std::iota(dev_indices.begin(), dev_indices.end(), 0);

      S best_dev_loss = S();
      bool first_dev_run = true;
      for (unsigned iter = 0; iter < num_iterations && !stop_requested; ++iter) {
        // Shuffle the training data indices
        std::shuffle(train_indices.begin(), train_indices.end(), *rndeng);

        S train_loss = S();

        std::vector<unsigned>::iterator begin = train_indices.begin();
        while (begin != train_indices.end()) {
          std::vector<unsigned>::iterator end = begin + report_frequency;
          if (end > train_indices.end()) {
            end = train_indices.end();
          }

          std::chrono::steady_clock::time_point start_time = std::chrono::steady_clock::now();
          double fractional_iter = iter + 1.0 * distance(train_indices.begin(), end) / train_indices.size();
          S batch_loss = run_data_set<S>(begin, end, workloads, mq, {false, end == train_indices.end(), report_frequency});
          std::chrono::steady_clock::time_point end_time = std::chrono::steady_clock::now();
          double seconds_elapsed = std::chrono::duration_cast<std::chrono::microseconds>(end_time - start_time).count() / 1000000.0;
          train_loss += batch_loss;
          std::cerr << fractional_iter << "\t" << "loss = " << batch_loss << " (" << seconds_elapsed << "s)" << std::endl;

          if (stop_requested) {
            break;
          }

          begin = end;
        }

        if (dev_data.size() > 0) {
          S dev_loss = run_data_set<S>(dev_indices.begin(), dev_indices.end(), workloads, mq, {true, false, report_frequency});
          bool new_best = (first_dev_run || dev_loss < best_dev_loss);
          first_dev_run = false;
          std::cerr << iter + 1 << "\t" << "dev loss = " << dev_loss << (new_best ? " (New best!)" : "") << std::endl;
          if (new_best) {
            learner->SaveModel();
            best_dev_loss = dev_loss;
          }
        } else {
          learner->SaveModel();
        }
      }

      // Kill all children one by one and wait for them to exit
      for (unsigned cid = 0; cid < num_children; ++cid) {
        bool cont = false;
        write_data(workloads[cid].p2c[1], cont);
        wait(NULL);
      }
      boost::interprocess::message_queue::remove(queue_name.c_str());
    }

    template<class D, class S>
    void run_multi_process_hacked(unsigned num_children, ILearner<D, S>* learner, Trainer* trainer, const std::vector<D>& train_data,
                           const std::vector<D>& dev_data, unsigned num_iterations, float num_reports_per_epoch) {
      queue_name = generate_queue_name();
      boost::interprocess::message_queue::remove(queue_name.c_str());
      boost::interprocess::message_queue::remove(queue_name.c_str());
      shared_memory_name = generate_shared_memory_name();
      shared_object = get_shared_memory<SharedObject>();
      std::vector<Workload> workloads = create_workloads(num_children);
      unsigned cid = spawn_children(workloads);
      if (cid < num_children) {
        run_child(cid, learner, trainer, workloads, train_data, dev_data);
        exit(0);
      }
      else {
        run_parent_hacked(train_data, dev_data, learner, workloads, num_iterations, num_reports_per_epoch);
        cleanup(workloads);
      }
    }

  }
}


namespace dyana {

  /**
   * when guarded, the model is training
   * useful when determining whether to add training noise or not, like dropout.
   */
  DEFINE_THREAD_LOCAL_GUARAD(training_guard)

  template<typename DATUM>
  class _mp_train_learner : private dynet::mp::ILearner<DATUM, float> {
  public:
    _mp_train_learner(unsigned num_workers, unsigned num_epochs, dynet::Trainer* trainer, const std::vector<DATUM> &training_set,
                      const std::vector<DATUM> &dev_set, std::function<dyana::tensor(const DATUM &)> compute_loss,
                      std::function<void()> save, float num_reports_per_epoch)
      :
      compute_loss(std::move(compute_loss)), save(std::move(save)) {

      if (num_workers <= 1) {
        dynet::mp::run_single_process_hacked(this, trainer, training_set, dev_set, num_epochs,
                                             num_reports_per_epoch);
      } else {
        multiprocessing_guard _;
        dynet::mp::run_multi_process_hacked(num_workers, this, trainer, training_set, dev_set, num_epochs,
                                            num_reports_per_epoch);
      }

    }

  private:
    virtual float LearnFromDatum(const DATUM &datum, bool learn) {
      if (dyana::tensor::get_exprs_counter() != 0) {
        throw std::runtime_error(
          "NO GLOBAL TENSOR. All dyana::Tensor instances must be cleaned up before training on a new Datum. Otherwise severe memory leak will occur while training.");
      }
      float ret = 0;
      if(learn) {
        training_guard _;
        dyana::tensor loss = compute_loss(datum);
        ret = loss.as_scalar();
        dyana::_cg().backward(loss);
      }
      else {
        dyana::tensor loss = compute_loss(datum);
        ret = loss.as_scalar();
      }
      return ret;
    }

    virtual void SaveModel() { save(); }

    std::function<dyana::tensor(const DATUM &)> compute_loss;
    std::function<void()> save;

  };

  /**
   * ensuring that all lazy-initialized paramers are initialized
   * by randomly running 8 training datum
   * \tparam DATUM 
   * \param training_set 
   * \param compute_loss 
   */
  template<typename DATUM>
  inline void _ensure_lazy_initialize(const std::vector<DATUM>& training_set, const std::function<dyana::tensor(const DATUM&)>& compute_loss) {
    std::vector<unsigned> train_indices(training_set.size());
    std::iota(train_indices.begin(), train_indices.end(), 0);
    std::shuffle(train_indices.begin(), train_indices.end(), *dynet::rndeng);

    for(unsigned i=0; i<8 && i<train_indices.size(); i++) {
      compute_loss(training_set[train_indices[i]]);
    }
  }

  template<typename DATUM>
  std::vector<std::vector<DATUM>> _group_in_batch(const std::vector<DATUM>& dataset, unsigned batch_size) {
    std::vector<std::vector<DATUM>> ret;
    for(unsigned i=0; i<dataset.size(); i+=batch_size) {
      auto begin = dataset.begin() + i;
      auto end = (i+batch_size>=dataset.size())?dataset.end():(begin+batch_size);
      ret.emplace_back(begin, end);
    }
    return ret;
  }

  template<typename DATUM>
  void _fit_impl(unsigned num_workers, unsigned batch_size, unsigned num_epochs, dynet::Trainer* trainer, const std::vector<DATUM> &training_set,
                 const std::vector<DATUM> &dev_set, std::function<dyana::tensor(const DATUM &)> compute_loss,
                 const std::function<void()>& save, float num_reports_per_epoch) {
    if (training_set.empty()) return;
    {
      training_guard _;
      _ensure_lazy_initialize(training_set, compute_loss);
    }

    if(batch_size == 1) {
      _mp_train_learner<DATUM>(num_workers, num_epochs, trainer, training_set, dev_set, compute_loss, save, num_reports_per_epoch);
      return;
    }

    auto batched_compute_loss = [&compute_loss](const std::vector<DATUM>& batched_datum){
      std::vector<dyana::tensor> losses;
      for(auto&& datum:batched_datum) {
        losses.push_back(compute_loss(datum));
      }
      return dyana::sum(losses);
    };

    _mp_train_learner<std::vector<DATUM>>(num_workers, num_epochs, trainer, _group_in_batch(training_set, batch_size), _group_in_batch(dev_set, batch_size), batched_compute_loss, save, num_reports_per_epoch);
  }

  inline std::vector<std::tuple<>> zip() {
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

    /**
     * how many times to report per epoch
     */
    float num_reports_per_epoch{1};
    
    /**
     * size of a batch
     * note that if the batch size is big, you may need to allocate more memory during dyana::initialize
     */
    unsigned batch_size{1};
  protected:
    virtual dynet::Trainer* get_dynet_trainer_p() = 0;
  private:
    event_emitter<> new_best_evt;
    template<typename MODEL, typename D0>
    void _fit_with_dev_set_helper(MODEL &model, const std::function<void()> &save_behavior, D0&& trainingset_p0, D0&& devset_p0) {
      using namespace std;
      using V0 = typename std::decay<D0>::type::value_type;
      using datum_type = std::tuple<V0>;
      auto training_set_tuple = zip(trainingset_p0);
      auto dev_set_tuple = zip(devset_p0);

      auto compute_loss_tuple = [&](const datum_type &args_pack) {
        return model.compute_loss(get<0>(args_pack));
      };
      _fit_impl<datum_type>(num_workers, batch_size, num_epochs, get_dynet_trainer_p(), training_set_tuple, dev_set_tuple,
                            compute_loss_tuple, save_behavior, num_reports_per_epoch);
    }
    template<typename MODEL, typename D0, typename D1>
    void _fit_with_dev_set_helper(MODEL &model,const std::function<void()> &save_behavior, D0&& trainingset_p0, D1&& trainingset_p1, D0&& devset_p0, D1&& devset_p1) {
      using namespace std;
      using V0 = typename std::decay<D0>::type::value_type;
      using V1 = typename std::decay<D1>::type::value_type;
      using datum_type = std::tuple<V0, V1>;
      auto training_set_tuple = zip(trainingset_p0, trainingset_p1);
      auto dev_set_tuple = zip(devset_p0, devset_p1);

      auto compute_loss_tuple = [&](const datum_type &args_pack) {
        return model.compute_loss(get<0>(args_pack), get<1>(args_pack));
      };

      _fit_impl<datum_type>(num_workers, batch_size, num_epochs, get_dynet_trainer_p(), training_set_tuple, dev_set_tuple,
                            compute_loss_tuple, save_behavior, num_reports_per_epoch);
    }
    template<typename MODEL, typename D0, typename D1, typename D2>
    void _fit_with_dev_set_helper(MODEL &model, const std::function<void()> &save_behavior,
                                  D0&& trainingset_p0, D1&& trainingset_p1, D2&& trainingset_p2,
                                  D0&& devset_p0, D1&& devset_p1, D2&& devset_p2) {
      using namespace std;
      using V0 = typename std::decay<D0>::type::value_type;
      using V1 = typename std::decay<D1>::type::value_type;
      using V2 = typename std::decay<D2>::type::value_type;
      using datum_type = std::tuple<V0, V1, V2>;
      auto training_set_tuple = zip(trainingset_p0, trainingset_p1, trainingset_p2);
      auto dev_set_tuple = zip(devset_p0, devset_p1, devset_p2);

      auto compute_loss_tuple = [&](const datum_type &args_pack) {
        return model.compute_loss(get<0>(args_pack), get<1>(args_pack), get<2>(args_pack));
      };

      _fit_impl<datum_type>(num_workers, batch_size, num_epochs, get_dynet_trainer_p(), training_set_tuple, dev_set_tuple,
                            compute_loss_tuple, save_behavior, num_reports_per_epoch);
    }
    template<typename MODEL, typename D0, typename D1, typename D2, typename D3>
    void _fit_with_dev_set_helper(MODEL &model, const std::function<void()> &save_behavior,
                                  D0&& trainingset_p0, D1&& trainingset_p1, D2&& trainingset_p2, D3&& trainingset_p3,
                                  D0&& devset_p0, D1&& devset_p1, D2&& devset_p2, D3&& devset_p3) {
      using namespace std;
      using V0 = typename std::decay<D0>::type::value_type;
      using V1 = typename std::decay<D1>::type::value_type;
      using V2 = typename std::decay<D2>::type::value_type;
      using V3 = typename std::decay<D3>::type::value_type;
      using datum_type = std::tuple<V0, V1, V2, V3>;
      auto training_set_tuple = zip(trainingset_p0, trainingset_p1, trainingset_p2, trainingset_p3);
      auto dev_set_tuple = zip(devset_p0, devset_p1, devset_p2, devset_p3);

      auto compute_loss_tuple = [&](const datum_type &args_pack) {
        return model.compute_loss(get<0>(args_pack), get<1>(args_pack), get<2>(args_pack), get<3>(args_pack));
      };

      _fit_impl<datum_type>(num_workers, batch_size, num_epochs, get_dynet_trainer_p(), training_set_tuple, dev_set_tuple,
                            compute_loss_tuple, save_behavior, num_reports_per_epoch);
    }
    template<typename MODEL, typename D0, typename D1, typename D2, typename D3, typename D4>
    void _fit_with_dev_set_helper(MODEL &model, const std::function<void()> &save_behavior,
                                  D0&& trainingset_p0, D1&& trainingset_p1, D2&& trainingset_p2, D3&& trainingset_p3,
                                  D4&& trainingset_p4,
                                  D0&& devset_p0, D1&& devset_p1, D2&& devset_p2, D3&& devset_p3,
                                  D4&& devset_p4) {
      using namespace std;
      using V0 = typename std::decay<D0>::type::value_type;
      using V1 = typename std::decay<D1>::type::value_type;
      using V2 = typename std::decay<D2>::type::value_type;
      using V3 = typename std::decay<D3>::type::value_type;
      using V4 = typename std::decay<D4>::type::value_type;
      using datum_type = std::tuple<V0, V1, V2, V3, V4>;
      auto training_set_tuple = zip(trainingset_p0, trainingset_p1, trainingset_p2, trainingset_p3, trainingset_p4);
      auto dev_set_tuple = zip(devset_p0, devset_p1, devset_p2, devset_p3, devset_p4);

      auto compute_loss_tuple = [&](const datum_type &args_pack) {
        return model.compute_loss(
          get<0>(args_pack), get<1>(args_pack), get<2>(args_pack), get<3>(args_pack),
          get<4>(args_pack));
      };

      _fit_impl<datum_type>(num_workers, batch_size, num_epochs, get_dynet_trainer_p(), training_set_tuple, dev_set_tuple,
                            compute_loss_tuple, save_behavior, num_reports_per_epoch);
    }
    template<typename MODEL, typename D0, typename D1, typename D2, typename D3, typename D4, typename D5>
    void _fit_with_dev_set_helper(MODEL &model, const std::function<void()> &save_behavior,
                                  D0&& trainingset_p0, D1&& trainingset_p1, D2&& trainingset_p2, D3&& trainingset_p3,
                                  D4&& trainingset_p4, D5&& trainingset_p5,
                                  D0&& devset_p0, D1&& devset_p1, D2&& devset_p2, D3&& devset_p3,
                                  D4&& devset_p4, D5&& devset_p5) {
      using namespace std;
      using V0 = typename std::decay<D0>::type::value_type;
      using V1 = typename std::decay<D1>::type::value_type;
      using V2 = typename std::decay<D2>::type::value_type;
      using V3 = typename std::decay<D3>::type::value_type;
      using V4 = typename std::decay<D4>::type::value_type;
      using V5 = typename std::decay<D5>::type::value_type;
      using datum_type = std::tuple<V0, V1, V2, V3, V4, V5>;
      auto training_set_tuple = zip(trainingset_p0, trainingset_p1, trainingset_p2, trainingset_p3, trainingset_p4, trainingset_p5);
      auto dev_set_tuple = zip(devset_p0, devset_p1, devset_p2, devset_p3, devset_p4, devset_p5);

      auto compute_loss_tuple = [&](const datum_type &args_pack) {
        return model.compute_loss(
          get<0>(args_pack), get<1>(args_pack), get<2>(args_pack), get<3>(args_pack),
          get<4>(args_pack), get<5>(args_pack));
      };

      _fit_impl<datum_type>(num_workers, batch_size, num_epochs, get_dynet_trainer_p(), training_set_tuple, dev_set_tuple,
                            compute_loss_tuple, save_behavior, num_reports_per_epoch);
    }
    template<typename MODEL, typename D0, typename D1, typename D2, typename D3, typename D4, typename D5, typename D6>
    void _fit_with_dev_set_helper(MODEL &model, const std::function<void()> &save_behavior,
                                  D0&& trainingset_p0, D1&& trainingset_p1, D2&& trainingset_p2, D3&& trainingset_p3,
                                  D4&& trainingset_p4, D5&& trainingset_p5, D6&& trainingset_p6,
                                  D0&& devset_p0, D1&& devset_p1, D2&& devset_p2, D3&& devset_p3,
                                  D4&& devset_p4, D5&& devset_p5, D6&& devset_p6) {
      using namespace std;
      using V0 = typename std::decay<D0>::type::value_type;
      using V1 = typename std::decay<D1>::type::value_type;
      using V2 = typename std::decay<D2>::type::value_type;
      using V3 = typename std::decay<D3>::type::value_type;
      using V4 = typename std::decay<D4>::type::value_type;
      using V5 = typename std::decay<D5>::type::value_type;
      using V6 = typename std::decay<D6>::type::value_type;
      using datum_type = std::tuple<V0, V1, V2, V3, V4, V5, V6>;
      auto training_set_tuple = zip(
        trainingset_p0, trainingset_p1, trainingset_p2, trainingset_p3,
        trainingset_p4, trainingset_p5, trainingset_p6);
      auto dev_set_tuple = zip(
        devset_p0, devset_p1, devset_p2, devset_p3,
        devset_p4, devset_p5, devset_p6);

      auto compute_loss_tuple = [&](const datum_type &args_pack) {
        return model.compute_loss(
          get<0>(args_pack), get<1>(args_pack), get<2>(args_pack), get<3>(args_pack),
          get<4>(args_pack), get<5>(args_pack), get<6>(args_pack));
      };

      _fit_impl<datum_type>(num_workers, batch_size, num_epochs, get_dynet_trainer_p(), training_set_tuple, dev_set_tuple,
                            compute_loss_tuple, save_behavior, num_reports_per_epoch);
    }
    template<typename MODEL, typename D0, typename D1, typename D2, typename D3, typename D4, typename D5, typename D6, typename D7>
    void _fit_with_dev_set_helper(MODEL &model, const std::function<void()> &save_behavior,
                                  D0&& trainingset_p0, D1&& trainingset_p1, D2&& trainingset_p2, D3&& trainingset_p3,
                                  D4&& trainingset_p4, D5&& trainingset_p5, D6&& trainingset_p6, D7&& trainingset_p7,
                                  D0&& devset_p0, D1&& devset_p1, D2&& devset_p2, D3&& devset_p3,
                                  D4&& devset_p4, D5&& devset_p5, D6&& devset_p6, D7&& devset_p7) {
      using namespace std;
      using V0 = typename std::decay<D0>::type::value_type;
      using V1 = typename std::decay<D1>::type::value_type;
      using V2 = typename std::decay<D2>::type::value_type;
      using V3 = typename std::decay<D3>::type::value_type;
      using V4 = typename std::decay<D4>::type::value_type;
      using V5 = typename std::decay<D5>::type::value_type;
      using V6 = typename std::decay<D6>::type::value_type;
      using V7 = typename std::decay<D7>::type::value_type;
      using datum_type = std::tuple<V0, V1, V2, V3, V4, V5, V6, V7>;
      auto training_set_tuple = zip(
        trainingset_p0, trainingset_p1, trainingset_p2, trainingset_p3,
        trainingset_p4, trainingset_p5, trainingset_p6, trainingset_p7);
      auto dev_set_tuple = zip(
        devset_p0, devset_p1, devset_p2, devset_p3,
        devset_p4, devset_p5, devset_p6, devset_p7);

      auto compute_loss_tuple = [&](const datum_type &args_pack) {
        return model.compute_loss(
          get<0>(args_pack), get<1>(args_pack), get<2>(args_pack), get<3>(args_pack),
          get<4>(args_pack), get<5>(args_pack), get<6>(args_pack), get<7>(args_pack));
      };

      _fit_impl<datum_type>(num_workers, batch_size, num_epochs, get_dynet_trainer_p(), training_set_tuple, dev_set_tuple,
                            compute_loss_tuple, save_behavior, num_reports_per_epoch);
    }



    template<typename MODEL, typename D0>
    void _fit_helper(MODEL &model, const std::function<void()> &save_behavior, D0&& trainingset_p0) {
      using namespace std;
      using V0 = typename std::decay<D0>::type::value_type;
      using datum_type = std::tuple<V0>;
      auto training_set_tuple = zip(trainingset_p0);

      auto compute_loss_tuple = [&](const datum_type &args_pack) {
        return model.compute_loss(get<0>(args_pack));
      };
      _fit_impl<datum_type>(num_workers, batch_size, num_epochs, get_dynet_trainer_p(), training_set_tuple, {}, compute_loss_tuple,
                            save_behavior, num_reports_per_epoch);
    }
    template<typename MODEL, typename D0, typename D1>
    void _fit_helper(MODEL &model,const std::function<void()> &save_behavior, D0&& trainingset_p0, D1&& trainingset_p1) {
      using namespace std;
      using V0 = typename std::decay<D0>::type::value_type;
      using V1 = typename std::decay<D1>::type::value_type;
      using datum_type = std::tuple<V0, V1>;
      auto training_set_tuple = zip(trainingset_p0, trainingset_p1);

      auto compute_loss_tuple = [&](const datum_type &args_pack) {
        return model.compute_loss(get<0>(args_pack), get<1>(args_pack));
      };

      _fit_impl<datum_type>(num_workers, batch_size, num_epochs, get_dynet_trainer_p(), training_set_tuple, {}, compute_loss_tuple,
                            save_behavior, num_reports_per_epoch);
    }
    template<typename MODEL, typename D0, typename D1, typename D2>
    void _fit_helper(MODEL &model, const std::function<void()> &save_behavior,
                                  D0&& trainingset_p0, D1&& trainingset_p1, D2&& trainingset_p2) {
      using namespace std;
      using V0 = typename std::decay<D0>::type::value_type;
      using V1 = typename std::decay<D1>::type::value_type;
      using V2 = typename std::decay<D2>::type::value_type;
      using datum_type = std::tuple<V0, V1, V2>;
      auto training_set_tuple = zip(trainingset_p0, trainingset_p1, trainingset_p2);

      auto compute_loss_tuple = [&](const datum_type &args_pack) {
        return model.compute_loss(get<0>(args_pack), get<1>(args_pack), get<2>(args_pack));
      };

      _fit_impl<datum_type>(num_workers, batch_size, num_epochs, get_dynet_trainer_p(), training_set_tuple, {}, compute_loss_tuple,
                            save_behavior, num_reports_per_epoch);
    }
    template<typename MODEL, typename D0, typename D1, typename D2, typename D3>
    void _fit_helper(MODEL &model, const std::function<void()> &save_behavior,
                                  D0&& trainingset_p0, D1&& trainingset_p1, D2&& trainingset_p2, D3&& trainingset_p3) {
      using namespace std;
      using V0 = typename std::decay<D0>::type::value_type;
      using V1 = typename std::decay<D1>::type::value_type;
      using V2 = typename std::decay<D2>::type::value_type;
      using V3 = typename std::decay<D3>::type::value_type;
      using datum_type = std::tuple<V0, V1, V2, V3>;
      auto training_set_tuple = zip(trainingset_p0, trainingset_p1, trainingset_p2, trainingset_p3);

      auto compute_loss_tuple = [&](const datum_type &args_pack) {
        return model.compute_loss(get<0>(args_pack), get<1>(args_pack), get<2>(args_pack), get<3>(args_pack));
      };

      _fit_impl<datum_type>(num_workers, batch_size, num_epochs, get_dynet_trainer_p(), training_set_tuple, {}, compute_loss_tuple,
                            save_behavior, num_reports_per_epoch);
    }
    template<typename MODEL, typename D0, typename D1, typename D2, typename D3, typename D4>
    void _fit_helper(MODEL &model, const std::function<void()> &save_behavior,
                                  D0&& trainingset_p0, D1&& trainingset_p1, D2&& trainingset_p2, D3&& trainingset_p3,
                                  D4&& trainingset_p4) {
      using namespace std;
      using V0 = typename std::decay<D0>::type::value_type;
      using V1 = typename std::decay<D1>::type::value_type;
      using V2 = typename std::decay<D2>::type::value_type;
      using V3 = typename std::decay<D3>::type::value_type;
      using V4 = typename std::decay<D4>::type::value_type;
      using datum_type = std::tuple<V0, V1, V2, V3, V4>;
      auto training_set_tuple = zip(trainingset_p0, trainingset_p1, trainingset_p2, trainingset_p3, trainingset_p4);

      auto compute_loss_tuple = [&](const datum_type &args_pack) {
        return model.compute_loss(
          get<0>(args_pack), get<1>(args_pack), get<2>(args_pack), get<3>(args_pack),
          get<4>(args_pack));
      };

      _fit_impl<datum_type>(num_workers, batch_size, num_epochs, get_dynet_trainer_p(), training_set_tuple, {}, compute_loss_tuple,
                            save_behavior, num_reports_per_epoch);
    }
    template<typename MODEL, typename D0, typename D1, typename D2, typename D3, typename D4, typename D5>
    void _fit_helper(MODEL &model, const std::function<void()> &save_behavior,
                                  D0&& trainingset_p0, D1&& trainingset_p1, D2&& trainingset_p2, D3&& trainingset_p3,
                                  D4&& trainingset_p4, D5&& trainingset_p5) {
      using namespace std;
      using V0 = typename std::decay<D0>::type::value_type;
      using V1 = typename std::decay<D1>::type::value_type;
      using V2 = typename std::decay<D2>::type::value_type;
      using V3 = typename std::decay<D3>::type::value_type;
      using V4 = typename std::decay<D4>::type::value_type;
      using V5 = typename std::decay<D5>::type::value_type;
      using datum_type = std::tuple<V0, V1, V2, V3, V4, V5>;
      auto training_set_tuple = zip(trainingset_p0, trainingset_p1, trainingset_p2, trainingset_p3, trainingset_p4, trainingset_p5);

      auto compute_loss_tuple = [&](const datum_type &args_pack) {
        return model.compute_loss(
          get<0>(args_pack), get<1>(args_pack), get<2>(args_pack), get<3>(args_pack),
          get<4>(args_pack), get<5>(args_pack));
      };

      _fit_impl<datum_type>(num_workers, batch_size, num_epochs, get_dynet_trainer_p(), training_set_tuple, {}, compute_loss_tuple,
                            save_behavior, num_reports_per_epoch);
    }
    template<typename MODEL, typename D0, typename D1, typename D2, typename D3, typename D4, typename D5, typename D6>
    void _fit_helper(MODEL &model, const std::function<void()> &save_behavior,
                                  D0&& trainingset_p0, D1&& trainingset_p1, D2&& trainingset_p2, D3&& trainingset_p3,
                                  D4&& trainingset_p4, D5&& trainingset_p5, D6&& trainingset_p6) {
      using namespace std;
      using V0 = typename std::decay<D0>::type::value_type;
      using V1 = typename std::decay<D1>::type::value_type;
      using V2 = typename std::decay<D2>::type::value_type;
      using V3 = typename std::decay<D3>::type::value_type;
      using V4 = typename std::decay<D4>::type::value_type;
      using V5 = typename std::decay<D5>::type::value_type;
      using V6 = typename std::decay<D6>::type::value_type;
      using datum_type = std::tuple<V0, V1, V2, V3, V4, V5, V6>;
      auto training_set_tuple = zip(
        trainingset_p0, trainingset_p1, trainingset_p2, trainingset_p3,
        trainingset_p4, trainingset_p5, trainingset_p6);

      auto compute_loss_tuple = [&](const datum_type &args_pack) {
        return model.compute_loss(
          get<0>(args_pack), get<1>(args_pack), get<2>(args_pack), get<3>(args_pack),
          get<4>(args_pack), get<5>(args_pack), get<6>(args_pack));
      };

      _fit_impl<datum_type>(num_workers, batch_size, num_epochs, get_dynet_trainer_p(), training_set_tuple, {}, compute_loss_tuple,
                            save_behavior, num_reports_per_epoch);
    }
    template<typename MODEL, typename D0, typename D1, typename D2, typename D3, typename D4, typename D5, typename D6, typename D7>
    void _fit_helper(MODEL &model, const std::function<void()> &save_behavior,
                                  D0&& trainingset_p0, D1&& trainingset_p1, D2&& trainingset_p2, D3&& trainingset_p3,
                                  D4&& trainingset_p4, D5&& trainingset_p5, D6&& trainingset_p6, D7&& trainingset_p7) {
      using namespace std;
      using V0 = typename std::decay<D0>::type::value_type;
      using V1 = typename std::decay<D1>::type::value_type;
      using V2 = typename std::decay<D2>::type::value_type;
      using V3 = typename std::decay<D3>::type::value_type;
      using V4 = typename std::decay<D4>::type::value_type;
      using V5 = typename std::decay<D5>::type::value_type;
      using V6 = typename std::decay<D6>::type::value_type;
      using V7 = typename std::decay<D7>::type::value_type;
      using datum_type = std::tuple<V0, V1, V2, V3, V4, V5, V6, V7>;
      auto training_set_tuple = zip(
        trainingset_p0, trainingset_p1, trainingset_p2, trainingset_p3,
        trainingset_p4, trainingset_p5, trainingset_p6, trainingset_p7);

      auto compute_loss_tuple = [&](const datum_type &args_pack) {
        return model.compute_loss(
          get<0>(args_pack), get<1>(args_pack), get<2>(args_pack), get<3>(args_pack),
          get<4>(args_pack), get<5>(args_pack), get<6>(args_pack), get<7>(args_pack));
      };

      _fit_impl<datum_type>(num_workers, batch_size, num_epochs, get_dynet_trainer_p(), training_set_tuple, {}, compute_loss_tuple,
                            save_behavior, num_reports_per_epoch);
    }
  public:
    virtual ~trainer_base() {
      // perform parameter collection GC only when destroying a trainer
      // because parameter collection GC will invalidate existing trainers
      // as a result, performing GC here is techically not 100% safe, but enough for daily use
      _force_garbage_collection();
    }
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
      _fit_helper(model, [](){}, training_set...);
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
      _fit_with_dev_set_helper(model, [&](){
        new_best_evt.fire();
      }, training_set_and_dev_set...);
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
        save_model(model, file_path_to_save_to);
        new_best_evt.fire();
      };
      _fit_with_dev_set_helper(model, save_behavior, training_set_and_dev_set...);
    }

    void add_new_best_listener(const event_emitter<>::listener_ptr& listener) {
      new_best_evt.add_listener(listener);
    }

    void remove_new_best_listener(const event_emitter<>::listener_ptr& listener) {
      new_best_evt.remove_listener(listener);
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
