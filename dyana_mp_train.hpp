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
    template<class LEARNER, class D, class S>
    void run_single_process_hacked(LEARNER* learner, Trainer* trainer, const std::vector<D>& train_data,
                            const std::vector<D>& dev_data, unsigned num_iterations, unsigned num_reports_per_epoch) {
      unsigned report_frequency = (train_data.size() + num_reports_per_epoch - 1) / num_reports_per_epoch;
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

        learner->handle_epoch_completion();

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

    template<class LEARNER, class D, class S>
    void run_parent_hacked(const std::vector<D>& train_data, const std::vector<D>& dev_data, LEARNER* learner, std::vector<Workload>& workloads, unsigned num_iterations, unsigned num_reports_per_epoch) {
      unsigned report_frequency = (train_data.size() + num_reports_per_epoch - 1) / num_reports_per_epoch;
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

        learner->handle_epoch_completion();

        if (dev_data.size() > 0) {
          S dev_loss = run_data_set<S>(dev_indices.begin(), dev_indices.end(), workloads, mq, {true, false, report_frequency});
          bool new_best = (first_dev_run || dev_loss < best_dev_loss);
          first_dev_run = false;
          std::cerr << iter + 1 << "\t" << "dev loss = " << dev_loss << (new_best ? " (New best!)" : "") << std::endl;
          if (new_best) {
            learner->SaveModel();
            best_dev_loss = dev_loss;
          }
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

    template<class LEARNER, class D, class S>
    void run_multi_process_hacked(unsigned num_children, LEARNER* learner, Trainer* trainer, const std::vector<D>& train_data,
                           const std::vector<D>& dev_data, unsigned num_iterations, float num_reports_per_epoch) {
      queue_name = generate_queue_name();
      boost::interprocess::message_queue::remove(queue_name.c_str());
      boost::interprocess::message_queue::remove(queue_name.c_str());
      shared_memory_name = generate_shared_memory_name();
      shared_object = get_shared_memory<SharedObject>();
      std::vector<Workload> workloads = create_workloads(num_children);
      unsigned cid = spawn_children(workloads);
      if (cid < num_children) {
        run_child<D, S>(cid, learner, trainer, workloads, train_data, dev_data);
        exit(0);
      }
      else {
        run_parent_hacked<LEARNER, D, S>(train_data, dev_data, learner, workloads, num_iterations, num_reports_per_epoch);
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

  using learning_rate_scheduler_t = std::function<float(unsigned datum_cnt, unsigned epoch_cnt)>;

  template<typename DATUM>
  class _mp_train_learner : public dynet::mp::ILearner<DATUM, float> {
  public:
    _mp_train_learner(unsigned num_workers, unsigned num_epochs, dynet::Trainer* trainer, learning_rate_scheduler_t learning_rate_scheduler, const std::vector<DATUM> &training_set,
                      const std::vector<DATUM> &dev_set, std::function<dyana::tensor(const DATUM &)> compute_loss, std::function<void()> new_best_callback, std::function<void()> epoch_completion_callback, float num_reports_per_epoch)
      :
      compute_loss(std::move(compute_loss)), new_best_callback(std::move(new_best_callback)), epoch_completion_callback(std::move(epoch_completion_callback)), trained_datum_counter_holder(), trained_datum_counter(trained_datum_counter_holder.add_parameters({1})), learning_rate_scheduler(std::move(learning_rate_scheduler)), trainer(trainer), curr_epoch_index() {

      trained_datum_counter.set_value({0});

      if (num_workers <= 1) {
        dynet::mp::run_single_process_hacked<_mp_train_learner<DATUM>, DATUM, float>(this, trainer, training_set, dev_set, num_epochs,
                                             num_reports_per_epoch);
      } else {
        multiprocessing_guard _;
        dynet::mp::run_multi_process_hacked<_mp_train_learner<DATUM>, DATUM, float>(num_workers, this, trainer, training_set, dev_set, num_epochs,
                                            num_reports_per_epoch);
      }

    }

    virtual void SaveModel() { new_best_callback(); }

    virtual void handle_epoch_completion() {
      epoch_completion_callback();
      ++curr_epoch_index;
    }


    virtual float LearnFromDatum(const DATUM &datum, bool learn) {
      if (dyana::tensor::get_exprs_counter() != 0) {
        throw std::runtime_error(
          "NO GLOBAL TENSOR. All dyana::Tensor instances must be cleaned up before training on a new Datum. Otherwise severe memory leak will occur while training.");
      }
      float ret = 0;
      try {
        if(learn) {
          training_guard _;
          dyana::tensor loss = compute_loss(datum);
          ret = loss.as_scalar();
          dyana::_cg().backward(loss);

          // increment trained datum counter only when learning rate scheduler is needed
          // because learning rate scheduler is the only usage if trained datum counter
          if(learning_rate_scheduler) {
            auto counter = increment_trained_datum_counter();
            trainer->learning_rate = learning_rate_scheduler(counter, curr_epoch_index);
          }

        }
        else {
          dyana::tensor loss = compute_loss(datum);
          ret = loss.as_scalar();
        }
      }
      catch(std::exception& ex) {
        std::cerr << ex.what() << std::endl;
      }
      return ret;
    }

  private:
    /**
     * trained datum counter ++
     * \return the trained datum counter before increment
     */
    unsigned increment_trained_datum_counter() {
      using namespace std;

      auto ret = (unsigned)(dynet::as_vector(*(trained_datum_counter.values()))[0]);

      if(multiprocessing_guard::is_guarded()) {
        dynet::mp::shared_object->counter_mutex.wait();
      }

      trained_datum_counter.set_value(std::vector<float>{(float)(ret + 1)});

      if(multiprocessing_guard::is_guarded()) {
        dynet::mp::shared_object->counter_mutex.post();
      }

      return ret;
    }

    std::function<dyana::tensor(const DATUM &)> compute_loss;
    std::function<void()> new_best_callback;
    std::function<void()> epoch_completion_callback;
    dynet::ParameterCollection trained_datum_counter_holder;
    dynet::Parameter trained_datum_counter;
    learning_rate_scheduler_t learning_rate_scheduler;
    dynet::Trainer* trainer;
    unsigned curr_epoch_index;

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
  void _fit_impl(unsigned num_workers, unsigned batch_size, unsigned num_epochs, dynet::Trainer* trainer, learning_rate_scheduler_t scheduler, const std::vector<DATUM> &training_set,
                 const std::vector<DATUM> &dev_set, std::function<dyana::tensor(const std::vector<DATUM> &)> batched_loss_fn, const std::function<void()>& new_best_callback, const std::function<void()>& epoch_completion_callback, float num_reports_per_epoch) {
    if (training_set.empty()) return;

    std::function<dyana::tensor(const DATUM&)> loss_fn = [&](const DATUM& datum)->dyana::tensor {
      return batched_loss_fn(std::vector<DATUM>{datum});
    };

    {
      training_guard _;
      _ensure_lazy_initialize(training_set, loss_fn);
    }

    if(batch_size == 1) {
      _mp_train_learner<DATUM>(num_workers, num_epochs, trainer, scheduler, training_set, dev_set, loss_fn, new_best_callback, epoch_completion_callback, num_reports_per_epoch);
      return;
    }

    auto batched_scheduler = (scheduler)?[&scheduler, &batch_size](unsigned datum_idx, unsigned epoch_idx) {
      return scheduler(datum_idx * batch_size, epoch_idx);
    }:scheduler;

    _mp_train_learner<std::vector<DATUM>>(num_workers, num_epochs, trainer, batched_scheduler, _group_in_batch(training_set, batch_size), _group_in_batch(dev_set, batch_size), batched_loss_fn, new_best_callback, epoch_completion_callback, num_reports_per_epoch);
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
  private:

    unsigned num_workers{1};
    unsigned num_epochs{1};
    float num_reports_per_epoch{1};
    unsigned batch_size{1};

    /**
     * override the default learning rate schedular
     */
    learning_rate_scheduler_t learning_rate_scheduler;

    event_emitter<> new_best_evt;
    event_emitter<> epoch_completion_evt;
  protected:
    virtual dynet::Trainer* get_dynet_trainer_p() = 0;

  public:
    virtual ~trainer_base() {
      // perform parameter collection GC only when destroying a trainer
      // because parameter collection GC will invalidate existing trainers
      // as a result, performing GC here is techically not 100% safe, but enough for daily use
      _force_garbage_collection();
    }

    /**
     * number of threads to spawn to train in parallel
     * only work with CPU training
     * \param num_workers
     */
    void set_num_workers(unsigned num_workers) {
      this->num_workers = num_workers;
    }

    /**
     * number of epochs to train
     * \param num_epochs
     */
    void set_num_epochs(unsigned num_epochs) {
      this->num_epochs = num_epochs;
    }

    /**
     * how many times to report per epoch
     * \param num_reports_per_epoch don't necessarily have to be an integer
     */
    void set_num_reports_per_epoch(float num_reports_per_epoch) {
      this->num_reports_per_epoch = num_reports_per_epoch;
    }

    /**
     * size of a batch
     * note that if the batch size is big, you may need to allocate more memory during dyana::initialize
     * \param batch_size
     */
    void set_batch_size(unsigned batch_size) {
      this->batch_size = batch_size;
    }

    /**
     * Set a dynamic learning rate.
     * Overrides the static learning rate specified in the constructor.
     * \param scheduler a function that takes:
     *                  (1) the datum index (starts from 0). Not to be confused with batcch index.
     *                  (2) the epoch index (starts from 0)
     *                  and returns the learning rate
     */
    void set_learning_rate_scheduler(learning_rate_scheduler_t scheduler) {
      this->learning_rate_scheduler = std::move(scheduler);
    }

    /**
     * Train your model.
     * The default strategy to compute the loss of a batch is to sum up the losses for each individual datum in the batch
     * \tparam DATUM represents a datum
     * \param loss_fn a function that takes a datum and returns a loss to minimize
     * \param training_set a list of datum to train on
     */
    template<typename DATUM>
    void train(const std::function<dyana::tensor(const DATUM&)> &loss_fn, const std::vector<DATUM>& training_set) {
      auto batched_loss_fn = [&](const std::vector<DATUM>& datum_batch) {
        if(datum_batch.size() == 0) return loss_fn(datum_batch.front());
        std::vector<dyana::tensor> losses;
        for(auto&& datum:datum_batch) {
          losses.push_back(loss_fn(datum));
        }
        return dyana::sum(losses);
      };
      return train<DATUM>(batched_loss_fn, training_set);
    }

    /**
     * Train your model with custom batch loss computation strategy.
     * \tparam DATUM represents a datum
     * \param batched_loss_fn a function that takes a minibatch of datum and returns a loss to minimize
     * \param training_set a list of datum to train on
     */
    template<typename DATUM>
    void train(const std::function<dyana::tensor(const std::vector<DATUM>&)>& batched_loss_fn, const std::vector<DATUM>& training_set) {
      auto save_behavior = [&]() {};
      auto epoch_completion_behavior = [&]() {
        epoch_completion_evt.fire();
      };

      _fit_impl<DATUM>(num_workers, batch_size, num_epochs, get_dynet_trainer_p(), learning_rate_scheduler, training_set, std::vector<DATUM>{}, batched_loss_fn, save_behavior, epoch_completion_behavior,num_reports_per_epoch);
    }

    /**
     * Train your model and validate it against a dev set after each epoch.
     * The default strategy to compute the loss of a batch is to sum up the losses for each individual datum in the batch
     * \tparam DATUM represents a datum
     * \param loss_fn a function that takes a datum and returns a loss to minimize
     * \param training_set a list of datum to train on
     * \param dev_set a list of datum to validate against
     */
    template<typename DATUM>
    void train_reporting_dev_score(const std::function<dyana::tensor(const DATUM&)> &loss_fn,  const std::vector<DATUM>& training_set, const std::vector<DATUM>& dev_set) {

      auto batched_loss_fn = [&](const std::vector<DATUM>& datum_batch) {
        if(datum_batch.size() == 0) return loss_fn(datum_batch.front());
        std::vector<dyana::tensor> losses;
        for(auto&& datum:datum_batch) {
          losses.push_back(loss_fn(datum));
        }
        return dyana::sum(losses);
      };

      train_reporting_dev_score(batched_loss_fn, training_set, dev_set);
    }

    /**
     * Train your model and validate it against a dev set after each epoch,
     * with custom batch loss computation strategy.
     * \tparam DATUM represents a datum
     * \param batched_loss_fn a function that takes a minibatch of datum and returns a loss to minimize
     * \param training_set a list of datum to train on
     * \param dev_set a list of datum to validate against
     */
    template<typename DATUM>
    void train_reporting_dev_score(const std::function<dyana::tensor(const std::vector<DATUM>&)> &batched_loss_fn, const std::vector<DATUM>& training_set, const std::vector<DATUM>& dev_set) {

      auto save_behavior = [&]() {
        new_best_evt.fire();
      };

      auto epoch_completion_behavior = [&]() {
        epoch_completion_evt.fire();
      };

      _fit_impl<DATUM>(num_workers, batch_size, num_epochs, get_dynet_trainer_p(), learning_rate_scheduler, training_set, dev_set,
                       batched_loss_fn, save_behavior, epoch_completion_behavior, num_reports_per_epoch);
    }

    /**
     * Register a listener that will be triggered
     * when a new best validation score has been achieved.
     * A good opportunity for you to save your model.
     *
     * Never triggers without a dev set
     *
     * \param listener the callback function to invoke
     * \return A handle to your listener.
     *         Please keep this handle if you want to unregister this listener later on.
     */
    event_emitter<>::listener_handle_t add_new_best_listener(const event_emitter<>::listener_t& listener) {
      return new_best_evt.add_listener(listener);
    }

    /**
     * Unregister a new best event listener
     * \param listener the listener handle
     */
    void remove_new_best_listener(const event_emitter<>::listener_handle_t& listener) {
      new_best_evt.remove_listener(listener);
    }

    /**
     * Register a listener that will be triggered
     * when an epoch is completed.
     *
     * the event sequence during the lifetime of an epoch is as follows:
     *   (1) Training for an epoch
     *   (2) epoch completion event
     *   (3) validating against the dev set
     *   (4) (possibly) new best event
     * \param listener
     * \return A handle to your listener.
     */
    event_emitter<>::listener_handle_t add_epoch_completion_listener(const event_emitter<>::listener_t& listener) {
      return epoch_completion_evt.add_listener(listener);
    }

    /**
     * Unregister an epoch completion event listener
     * \param listener the listener handle
     */
    void remove_epoch_completion_listener(const event_emitter<>::listener_handle_t& listener) {
      epoch_completion_evt.remove_listener(listener);
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
