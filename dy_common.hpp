//
// Created by YAN Yuchen on 4/30/2018.
//

#ifndef DYNET_DYNET_WRAPPER_HPP
#define DYNET_DYNET_WRAPPER_HPP

#include <dynet/dynet.h>
#include <dynet/expr.h>
#include <vector>
#include <dynet/training.h>

#define _RETURN_CACHED_EXPR(ptr, fallback) \
  try { \
    return _params_that_have_expr_in_cg().at(ptr);\
  }\
  catch(std::out_of_range& e) {\
    _params_that_have_expr_in_cg()[ptr] = fallback;\
    return _params_that_have_expr_in_cg()[ptr];\
  }\

#define AUTO_START_GRAPH(ptr, action) \
  if(tg::dy::_those_who_have_their_graph_started().count(ptr)<=0) {\
    action;\
    tg::dy::_those_who_have_their_graph_started().insert(ptr);\
  }\

#define AUTO_START_THIS_GRAPH(action) AUTO_START_GRAPH(this, action)

#define DECLARE_DEFAULT_CONSTRUCTORS(class_name) \
  class_name() = default;\
  class_name(const class_name&) = default;\
  class_name(class_name&&) = default;\
  class_name& operator=(const class_name&) = default;\
  class_name& operator=(class_name&&) = default;\

namespace tg {
  namespace dy {


    /**
     * get the computation graph instance
     * \return
     */
    inline dynet::ComputationGraph& cg() {
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

    inline void train_on_loss(const dynet::Expression& loss) {
      cg().forward(loss);
      cg().backward(loss);
      trainer().update();
    }

    inline std::unordered_map<const void*, dynet::Expression>& _params_that_have_expr_in_cg() {
      static std::unordered_map<const void*, dynet::Expression> _exprs;
      return _exprs;
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
      _params_that_have_expr_in_cg().clear();
      _those_who_have_their_graph_started().clear();
      cg().clear();
      if(_should_check_nan()) {
        cg().set_immediate_compute(true);
        cg().set_check_validity(true);
      }
    }

    class Expression:public dynet::Expression {
    public:
      Expression():dynet::Expression(){increment_cnt();};
      Expression(const dynet::Expression& x):dynet::Expression(x) {increment_cnt();};
      Expression(dynet::Expression&& x):dynet::Expression(x) {increment_cnt();};
      Expression &operator=(const dynet::Expression& x) {dynet::Expression::operator=(x); return *this;} ;
      Expression &operator=(dynet::Expression&& x) {dynet::Expression::operator=(x); return *this;};
      ~Expression(){
        num_exprs()--;
        if(num_exprs()==0) dy::_renew_cg();
      }
      static std::vector<dynet::Expression> vector_cast_to_base(const std::vector<Expression>& x) {
        return std::vector<dynet::Expression>(x.begin(), x.end());
      }
      static std::vector<Expression> vector_cast_to_parent(const std::vector<dynet::Expression>& x) {
        return std::vector<Expression>(x.begin(), x.end());
      }
    private:
      static unsigned long& num_exprs() { static unsigned long _; return _;}
      void increment_cnt() {num_exprs()++;};
    };

    /**
     * call this before any other dynet related stuffs are called
     */
    inline void initialize() {
      std::vector<std::string> arguments = {"", "--dynet-mem=2048"};

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
    }

    inline dynet::Parameter add_parameters(const dynet::Dim& dim) {
      return pc().add_parameters(dim);
    }

    inline dynet::LookupParameter add_lookup_parameters(unsigned capacity, const dynet::Dim& dim) {
      return pc().add_lookup_parameters(capacity, dim);
    }

    /**
     * get the Expression associated with a given Parameter
     * (Expression can be used in computation graph to define neural network topology)
     * \param p the Parameter
     * \return the Expression
     */
    inline dy::Expression expr(const dynet::Parameter& p) {
      _RETURN_CACHED_EXPR(&p, dynet::parameter(cg(), p))
    }

    /**
     * get the const Expression associated with a given Parameter
     * (Expression can be used in computation graph to define neural network topology)
     * (Parameters under const Expression will not be updated by this Expression during training)
     * \param p the Parameter
     * \return the const Expression
     */
    inline dy::Expression const_expr(const dynet::Parameter& p) {
      _RETURN_CACHED_EXPR(&p, dynet::const_parameter(cg(), p))
    }

    /**
     * create a const expression whose value is a scalar
     * \param value the scalar
     * \return the const expression
     */
    inline dy::Expression const_expr(float value) {
      return dynet::input(dy::cg(), value);
    }

    /**
     * create a const expression whose value is a dim(n) tensor
     * \param values the value of the expression
     * \return the const expression
     */
    inline dy::Expression const_expr(const std::vector<float>& values) {
      return dynet::input(dy::cg(), {(unsigned)values.size()}, values);
    }

    /**
     * find the index of the largest value
     * \param logits must be of dim{n}
     * \return
     */
    inline unsigned argmax_index(const dy::Expression& logits) {
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

    inline float as_scalar(const dy::Expression& expr) {
      return dynet::as_scalar(cg().forward(expr));
    }

    inline std::vector<float> as_vector(const dy::Expression& expr) {
      return dynet::as_vector(cg().forward(expr));
    }

  }
}


#endif //DYNET_DYNET_WRAPPER_HPP
