//
// Created by Dekai WU and YAN Yuchen on 20190322.
//

#ifndef DYNET_WRAPPER_DY_DIRTY_CORE_HPP
#define DYNET_WRAPPER_DY_DIRTY_CORE_HPP
#define _DYNET_WRAPPER_DEFAULT_UNK "&unk;"

#include <dynet/dynet.h>
#include <vector>
#include <dynet/training.h>

#define AUTO_START_GRAPH(ptr, action) \
  if(tg::dy::_those_who_have_their_graph_started().count(ptr)<=0) {\
    action;\
    tg::dy::_those_who_have_their_graph_started().insert(ptr);\
  }\

#define AUTO_START_THIS_GRAPH(action) AUTO_START_GRAPH(this, action)

namespace tg {
  namespace dy {

    typedef dynet::Dim Dim;

    inline bool &_is_initialized() { static bool _ = false;return _; }

    inline dynet::ParameterCollection* &_pc() {
      static dynet::ParameterCollection* _pc = new dynet::ParameterCollection();
      return _pc;
    }

    inline dynet::Trainer* &_trainer() {
      static dynet::Trainer* _trainer = new dynet::AdamTrainer(*_pc());
      if(_trainer->model != _pc()) {
        _trainer->model = _pc();
        _trainer->restart();
      }
      return _trainer;
    }

    enum trainer_type {SIMPLE_SGD, CYCLICAL_SGD, MOMENTUM_SGD, ADAGRAD, ADADELTA, RMS_PROP, ADAM, AMSGRAD, EXPONENTIATED_GRADIENT};
    inline unsigned &_num_workers() {static unsigned _=1; return _;}

    /**
     * call this before any other dynet related stuffs are called
     */
    inline void initialize(unsigned num_workers=1, trainer_type trainer = ADAM, float learning_rate = 0.01, unsigned memory=512) {
      if (_is_initialized()) return;
      std::vector<std::string> arguments = {"", "--dynet-mem="+std::to_string(memory)};

      std::vector<char *> argv;
      for (const auto &arg : arguments)
        argv.push_back((char *) arg.data());
      argv.push_back(nullptr);

      int argc = (int) argv.size() - 1;
      char **argv2 = argv.data();
      auto dynet_params = dynet::extract_dynet_params(argc, argv2, true);
      dynet::initialize(dynet_params);

      // dynet internally uses the mt1997 RNG, but with one exception.
      // in interprocess, when generating queue names, it uses rand() instead of mt1997 RNG
      // so we also need to randomize this
      // otherwise you cannot have multiple dynet program running on the same machine! queue name clash!
      srand(dynet_params.random_seed);
      _is_initialized() = true;
      _num_workers()=num_workers;
      if(_trainer()) delete _trainer();
      switch (trainer) {
        case SIMPLE_SGD:
          _trainer() = new dynet::SimpleSGDTrainer(*_pc(), learning_rate);
          break;
        case CYCLICAL_SGD:
          _trainer() = new dynet::CyclicalSGDTrainer(*_pc(), learning_rate, learning_rate*10);
          break;
        case MOMENTUM_SGD:
          _trainer() = new dynet::MomentumSGDTrainer(*_pc(), learning_rate);
          break;
        case ADAGRAD:
          _trainer() = new dynet::AdagradTrainer(*_pc(), learning_rate);
          break;
        case ADADELTA:
          _trainer() = new dynet::AdadeltaTrainer(*_pc());
          break;
        case RMS_PROP:
          _trainer() = new dynet::RMSPropTrainer(*_pc(), learning_rate);
          break;
        case ADAM:
          _trainer() = new dynet::AdagradTrainer(*_pc(), learning_rate);
          break;
        case AMSGRAD:
          _trainer() = new dynet::AmsgradTrainer(*_pc(), learning_rate);
          break;
        case EXPONENTIATED_GRADIENT:
          _trainer() = new dynet::EGTrainer(*_pc(), learning_rate);
          break;
        default:
          _trainer() = new dynet::AdamTrainer(*_pc());
      }
    }

    inline void _ensure_initialized() {
      if (!_is_initialized()) { throw std::runtime_error("dy::initialize must be called beforehand"); }
    }

    /**
     * get the computation graph instance
     * \return
     */
    inline dynet::ComputationGraph &_cg() {
      _ensure_initialized();
      static dynet::ComputationGraph _cg;
      return _cg;
    }

    inline bool &_should_check_nan() {
      static bool value = false;
      return value;
    }

    inline std::unordered_set<const void *> &_those_who_have_their_graph_started() {
      static std::unordered_set<const void *> _those_who_have_their_graph_started;
      return _those_who_have_their_graph_started;
    }

    /**
     * if called, the program will trace NaN values during runtime.
     * useful for debugging
     * can be called either before or after dy::initialize. doesn't matter
     */
    inline void check_nan() {
      _should_check_nan() = true;
    }

    /**
     * destroy all previous declared Expressions
     * the computation graph have garbage collection built-in so as a user you don't need to call this explicitly
     */
    inline void _renew_cg() {
      _those_who_have_their_graph_started().clear();
      _cg().clear();
      if (_should_check_nan()) {
        _cg().set_immediate_compute(true);
        _cg().set_check_validity(true);
      }
    }

  }
}
#endif //DYNET_WRAPPER_DY_DIRTY_CORE_HPP
