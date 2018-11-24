//
// Created by YAN Yuchen on 4/30/2018.
//

#ifndef DYNET_DYNET_WRAPPER_HPP
#define DYNET_DYNET_WRAPPER_HPP

#include <dynet/dynet.h>
#include <dynet/expr.h>
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
    typedef dynet::Parameter Parameter;
    typedef dynet::LookupParameter LookupParameter;

    inline bool& _is_initialized() {
      static bool _ = false;
      return _;
    }

    /**
     * call this before any other dynet related stuffs are called
     */
    void initialize() {
      if(_is_initialized()) return;
      std::vector<std::string> arguments = {"", "--dynet-mem=512"};

      std::vector<char*> argv;
      for (const auto& arg : arguments)
        argv.push_back((char*)arg.data());
      argv.push_back(nullptr);

      int argc = (int)argv.size() - 1;
      char** argv2 = argv.data();
      auto dynet_params = dynet::extract_dynet_params(argc, argv2, true);
      dynet::initialize(dynet_params);

      // dynet internally uses the mt1997 RNG, but with one exception.
      // in interprocess, when generating queue names, it uses rand() instead of mt1997 RNG
      // so we also need to randomize this
      // otherwise you cannot have multiple dynet program running on the same machine! queue name clash!
      srand(dynet_params.random_seed);
      _is_initialized() = true;
    }

    inline void ensure_initialized() {
      if(!_is_initialized()) {throw std::runtime_error("dy::initialize must be called beforehand");}
    }

    /**
     * get the computation graph instance
     * \return
     */
    inline dynet::ComputationGraph& cg() {
      ensure_initialized();
      static dynet::ComputationGraph _cg;
      return _cg;
    }

    inline dynet::ParameterCollection& pc() {
      static dynet::ParameterCollection _pc;
      return _pc;
    }

    inline dynet::Trainer& trainer() {
      static dynet::AdamTrainer _trainer(pc());
      return _trainer;
    }

    inline bool& _should_check_nan() {
      static bool value = false;
      return value;
    }
    inline std::unordered_set<const void*>& _those_who_have_their_graph_started() {
      static std::unordered_set<const void*> _those_who_have_their_graph_started;
      return _those_who_have_their_graph_started;
    }

    /**
     * if called, the program will trace NaN values during runtime.
     * useful for debugging
     * can be called either before or after dy::initialize. doesn't matter
     */
    inline void check_nan() {
      _should_check_nan()=true;
    }

    /**
     * destroy all previous declared Expressions
     * the computation graph have garbage collection built-in so as a user you don't need to call this explicitly
     */
    inline void _renew_cg() {
      _those_who_have_their_graph_started().clear();
      cg().clear();
      if(_should_check_nan()) {
        cg().set_immediate_compute(true);
        cg().set_check_validity(true);
      }
    }

    class tensor:public dynet::Expression {
    public:
      tensor():dynet::Expression(){increment_cnt();};
      tensor(const dynet::Expression& x):dynet::Expression(x) {increment_cnt();};
      tensor(const dy::tensor& x):dynet::Expression(x) {increment_cnt();};
      tensor(dynet::Expression&& x):dynet::Expression(x) {increment_cnt();};
      tensor(dy::tensor&& x):dynet::Expression(x) {increment_cnt();};
      tensor(const dynet::Parameter& x):dynet::Expression(dynet::parameter(cg(), x)) {increment_cnt();}
      tensor(float x):dynet::Expression(dynet::input(dy::cg(), x)){increment_cnt();}
      tensor(const std::vector<float> x):dynet::Expression(dynet::input(dy::cg(), {(unsigned)x.size()}, x)) {increment_cnt();}
      tensor(const std::initializer_list<float> x):dynet::Expression(dynet::input(dy::cg(), {(unsigned)x.size()}, x)) {increment_cnt();}
      tensor(const std::vector<float>& values, const dynet::Dim& dim):dynet::Expression(dynet::input(dy::cg(), dim, values)) {increment_cnt();}
      tensor(const std::initializer_list<float>& values, const dynet::Dim& dim):dynet::Expression(dynet::input(dy::cg(), dim, values)) {increment_cnt();}
      tensor &operator=(const dynet::Expression& x) {dynet::Expression::operator=(x); return *this;};
      tensor &operator=(const dy::tensor& x) {dynet::Expression::operator=(x); return *this;} ;
      tensor &operator=(dynet::Expression&& x) {dynet::Expression::operator=(x); return *this;};
      tensor &operator=(dy::tensor&& x) {dynet::Expression::operator=(x); return *this;};
      float as_scalar() {return dynet::as_scalar(dy::cg().incremental_forward(*this));}
      std::vector<float> as_vector() {return dynet::as_vector(dy::cg().incremental_forward(*this));}
      ~tensor(){
        num_exprs()--;
        if(num_exprs()==0) dy::_renew_cg();
      }
      static std::vector<dynet::Expression> vector_cast_to_base(const std::vector<tensor>& x) {
        return std::vector<dynet::Expression>(x.begin(), x.end());
      }
      static std::vector<tensor> vector_cast_to_parent(const std::vector<dynet::Expression>& x) {
        return std::vector<tensor>(x.begin(), x.end());
      }
      static unsigned get_exprs_counter(){return num_exprs();}
    private:
      static unsigned long& num_exprs() { static unsigned long _; return _;}
      void increment_cnt() {num_exprs()++;};
    };

    inline Parameter add_parameters(const Dim& dim) {
      return pc().add_parameters(dim);
    }

    inline LookupParameter add_lookup_parameters(unsigned capacity, const Dim& dim) {
      return pc().add_lookup_parameters(capacity, dim);
    }


    /**
     * get the const Expression associated with a given Parameter
     * (Expression can be used in computation graph to define neural network topology)
     * (Parameters under const Expression will not be updated by this Expression during training)
     * \param p the Parameter
     * \return the const Expression
     */
    inline tensor const_expr(const Parameter& p) {
      return dynet::const_parameter(cg(), p);
    }


    /**
     * find the index of the largest value
     * \param logits must be of dim{n}
     * \return
     */
    inline unsigned argmax_index(const tensor& logits) {
      auto logits_value = as_vector(dy::cg().incremental_forward(logits));

      float max_value = logits_value[0];
      unsigned max_index = 0;
      for(unsigned i=1; i<logits_value.size(); ++i) {
        float val = logits_value[i];
        if(val>max_value) {
          max_value = val;
          max_index = i;
        }
      }
      return max_index;
    }

  }
}


#endif //DYNET_DYNET_WRAPPER_HPP
