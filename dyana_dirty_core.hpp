//
// Created by Dekai WU and YAN Yuchen on 20190322.
//

#ifndef DYANA_DIRTY_CORE_HPP
#define DYANA_DIRTY_CORE_HPP
#define _DYNET_WRAPPER_DEFAULT_UNK "&unk;"

#include <dynet/dynet.h>
#include <vector>
#include <dynet/training.h>
#include <mutex>

#define AUTO_START_GRAPH(ptr, action) \
  if(tg::dyana::_those_who_have_their_graph_started().count(ptr)<=0) {\
    action;\
    tg::dyana::_those_who_have_their_graph_started().insert(ptr);\
  }\

#define AUTO_START_THIS_GRAPH(action) AUTO_START_GRAPH(this, action)


namespace dyana {


  using Dim = dynet::Dim;

  inline bool& _is_initialized() {
    static bool _ = false;
    return _;
  }

  inline dynet::ParameterCollection *& _pc() {
    static dynet::ParameterCollection *_pc = new dynet::ParameterCollection();
    return _pc;
  }

  enum ProfilingVerbosity {NONE=0, ROUGH, VERBOSE};

  /**
   * call this before any other dynet related stuffs are called
   */
  inline void
  initialize(unsigned memory = 2048, ProfilingVerbosity profiling=NONE) {
    if (_is_initialized()) return;
    std::string quater_mem = std::to_string(memory / 4);
    std::vector<std::string> arguments = {
      "",
      "--dynet-mem=" + quater_mem + "," + quater_mem + "," + quater_mem + "," + quater_mem,
      "--dynet-autobatch=1",
      "--dynet-profiling=" + std::to_string((unsigned)profiling)};
    std::vector<char *> argv;
    for (const auto& arg : arguments)
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
  }

  inline void _ensure_initialized() {
    if (!_is_initialized()) { throw std::runtime_error("dyana::initialize must be called beforehand"); }
  }

  inline dynet::ComputationGraph *& _pcg() {
    _ensure_initialized();
    thread_local static auto *_cg = new dynet::ComputationGraph;
    return _cg;
  }

  /**
   * get the computation graph instance
   * \return
   */
  inline dynet::ComputationGraph& _cg() {
    return *_pcg();
  }

  inline bool& _should_check_nan() {
    static bool value = false;
    return value;
  }

  inline std::unordered_set<const void *>& _those_who_have_their_graph_started() {
    thread_local static std::unordered_set<const void *> _those_who_have_their_graph_started;
    return _those_who_have_their_graph_started;
  }

  /**
   * if called, the program will trace NaN values during runtime.
   * useful for debugging
   * can be called either before or after dyana::initialize. doesn't matter
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
    delete _pcg();
    _pcg() = new dynet::ComputationGraph;
    if (_should_check_nan()) {
      _cg().set_immediate_compute(true);
      _cg().set_check_validity(true);
    }
  }

}

#endif //DYANA_DIRTY_CORE_HPP
