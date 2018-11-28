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

    inline bool &_is_initialized() { static bool _ = false;return _; }

    inline dynet::ParameterCollection &_pc() {
      static dynet::ParameterCollection _pc;
      return _pc;
    }

    inline dynet::Trainer* &_trainer() {
      static dynet::Trainer* _trainer = new dynet::AdamTrainer(_pc());
      return _trainer;
    }

    enum trainer_type {SIMPLE_SGD, CYCLICAL_SGD, MOMENTUM_SGD, ADAGRAD, ADADELTA, RMS_PROP, ADAM, AMSGRAD, EXPONENTIATED_GRADIENT};
    inline unsigned &_num_workers() {static unsigned _=1; return _;}

    /**
     * call this before any other dynet related stuffs are called
     */
    void initialize(unsigned num_workers=1, trainer_type trainer = ADAM, float learning_rate = 0.01, unsigned memory=512) {
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
          _trainer() = new dynet::SimpleSGDTrainer(_pc(), learning_rate);
          break;
        case CYCLICAL_SGD:
          _trainer() = new dynet::CyclicalSGDTrainer(_pc(), learning_rate, learning_rate*10);
          break;
        case MOMENTUM_SGD:
          _trainer() = new dynet::MomentumSGDTrainer(_pc(), learning_rate);
          break;
        case ADAGRAD:
          _trainer() = new dynet::AdagradTrainer(_pc(), learning_rate);
          break;
        case ADADELTA:
          _trainer() = new dynet::AdadeltaTrainer(_pc());
          break;
        case RMS_PROP:
          _trainer() = new dynet::RMSPropTrainer(_pc(), learning_rate);
          break;
        case ADAM:
          _trainer() = new dynet::AdagradTrainer(_pc(), learning_rate);
          break;
        case AMSGRAD:
          _trainer() = new dynet::AmsgradTrainer(_pc(), learning_rate);
          break;
        case EXPONENTIATED_GRADIENT:
          _trainer() = new dynet::EGTrainer(_pc(), learning_rate);
          break;
        default:
          _trainer() = new dynet::AdamTrainer(_pc());
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

    class tensor : public dynet::Expression {
    public:
      tensor() : dynet::Expression() { increment_cnt(); };

      tensor(const dynet::Expression &x) : dynet::Expression(x) { increment_cnt(); };

      tensor(const dy::tensor &x) : dynet::Expression(x) { increment_cnt(); };

      tensor(dynet::Expression &&x) : dynet::Expression(x) { increment_cnt(); };

      tensor(dy::tensor &&x) : dynet::Expression(x) { increment_cnt(); };

      tensor(const dynet::Parameter &x) : dynet::Expression(dynet::parameter(_cg(), x)) { increment_cnt(); }

      tensor(float x) : dynet::Expression(dynet::input(dy::_cg(), x)) { increment_cnt(); }

      tensor(const std::vector<float> x) : dynet::Expression(
        dynet::input(dy::_cg(), {(unsigned) x.size()}, x)) { increment_cnt(); }

      tensor(const std::initializer_list<float> x) : dynet::Expression(
        dynet::input(dy::_cg(), {(unsigned) x.size()}, x)) { increment_cnt(); }

      tensor(const std::vector<float> &values, const dynet::Dim &dim) : dynet::Expression(
        dynet::input(dy::_cg(), dim, values)) { increment_cnt(); }

      tensor(const std::initializer_list<float> &values, const dynet::Dim &dim) : dynet::Expression(
        dynet::input(dy::_cg(), dim, values)) { increment_cnt(); }

      tensor &operator=(const dynet::Expression &x) {
        dynet::Expression::operator=(x);
        return *this;
      };

      tensor &operator=(const dy::tensor &x) {
        dynet::Expression::operator=(x);
        return *this;
      };

      tensor &operator=(dynet::Expression &&x) {
        dynet::Expression::operator=(x);
        return *this;
      };

      tensor &operator=(dy::tensor &&x) {
        dynet::Expression::operator=(x);
        return *this;
      };

      /**
       * \brief Pick element
       * \details Pick a single element/row/column/sub-tensor.
       *          This will result in the dimension of the tensor being reduced
       *          by 1.
       *
       * \param x The input expression
       * \param v The index of the element to select
       * \param d The dimension along which to choose the element
       *
       * \return The value of x[v] along dimension d
       */
      tensor at(unsigned v, unsigned d=0) {
        return dynet::pick(*this, v, d);
      }

      /**
       * \brief Pick range of elements
       * \details Pick a range of elements from a tensor.
       *
       * \param x The input expression
       * \param s The start index
       * \param e The end index, excluding itself
       * \param d The dimension along which to pick
       *
       * \return The value of {x[v],...,x[u]}
       */
      tensor slice(unsigned s, unsigned e, unsigned d=0) {
        return dynet::pick_range(*this, s, e, d);
      }

      /**
       * \brief Select rows
       * \details Select a subset of rows of a matrix.
       *
       * \param x The input expression
       * \param rows The rows to extract
       *
       * \return An expression containing the selected rows
       */
      tensor select_rows(const std::vector<unsigned> &rows) {
        return dynet::select_rows(*this, rows);
      }

      /**
       * \brief Select columns
       * \details Select a subset of columns of a matrix. select_cols is more
       *          efficient than select_rows since DyNet uses column-major order.
       *
       * \param x The input expression
       * \param columns The columns to extract
       *
       * \return An expression containing the selected columns
       */
      tensor select_cols(const std::vector<unsigned> &cols) {
        return dynet::select_cols(*this, cols);
      }

      /**
       * \brief Reshape to another size
       * \details This node reshapes a tensor to another size, without changing the
       *          underlying layout of the data. The layout of the data in DyNet is
       *          column-major, so if we have a 3x4 matrix
       *
       *    \f$
       *      \begin{pmatrix}
       *        x_{1,1} & x_{1,2} & x_{1,3} & x_{1,4} \\
       *        x_{2,1} & x_{2,2} & x_{2,3} & x_{2,4} \\
       *        x_{3,1} & x_{3,2} & x_{3,3} & x_{3,4} \\
       *      \end{pmatrix}
       *    \f$
       *
       *          and transform it into a 2x6 matrix, it will be rearranged as:
       *
       *    \f$
       *      \begin{pmatrix}
       *        x_{1,1} & x_{3,1} & x_{2,2} & x_{1,3} & x_{3,3} & x_{2,4} \\
       *        x_{2,1} & x_{1,2} & x_{3,2} & x_{2,3} & x_{1,4} & x_{3,4} \\
       *      \end{pmatrix}
       *    \f$
       *
       *         **Note:** This is O(1) for forward, and O(n) for backward.
       *
       * \param x The input expression
       * \param d The new dimensions
       *
       * \return The reshaped expression
       */
      tensor reshape(const Dim &d) {
        return dynet::reshape(*this, d);
      }

      /**
       * \brief Transpose a matrix
       * \details Transpose a matrix or tensor, or if dims is specified shuffle the
       *          dimensions arbitrarily.
       *          **Note:** This is O(1) if either the row or column dimension is 1,
       *          and O(n) otherwise.
       *
       * \param x The input expression
       * \param dims The dimensions to swap. The ith dimension of the output will be equal
       *          to the dims[i] dimension of the input. dims must have the same number
       *          of dimensions as x.
       *
       * \return The transposed/shuffled expression
       */
      tensor transpose(const std::vector<unsigned> &dims = {1, 0}) {
        return dynet::transpose(*this, dims);
      }

      float as_scalar() const { return dynet::as_scalar(dy::_cg().incremental_forward(*this)); }

      std::vector<float> as_vector() const { return dynet::as_vector(dy::_cg().incremental_forward(*this)); }

      ~tensor() {
        num_exprs()--;
        if (num_exprs() == 0) dy::_renew_cg();
      }

      static std::vector<dynet::Expression> vector_cast_to_base(const std::vector<tensor> &x) {
        return std::vector<dynet::Expression>(x.begin(), x.end());
      }

      static std::vector<tensor> vector_cast_to_parent(const std::vector<dynet::Expression> &x) {
        return std::vector<tensor>(x.begin(), x.end());
      }

      static unsigned get_exprs_counter() { return num_exprs(); }

    private:
      static unsigned long &num_exprs() {
        static unsigned long _;
        return _;
      }

      void increment_cnt() { num_exprs()++; };
    };

    inline Parameter add_parameters(const Dim &dim) {
      return _pc().add_parameters(dim);
    }

    inline LookupParameter add_lookup_parameters(unsigned capacity, const Dim &dim) {
      return _pc().add_lookup_parameters(capacity, dim);
    }


    /**
     * get the const Expression associated with a given Parameter
     * (Expression can be used in computation graph to define neural network topology)
     * (Parameters under const Expression will not be updated by this Expression during training)
     * \param p the Parameter
     * \return the const Expression
     */
    inline tensor const_expr(const Parameter &p) {
      return dynet::const_parameter(_cg(), p);
    }


    /**
     * find the index of the largest value
     * \param logits must be of dim{n}
     * \return
     */
    inline unsigned argmax_index(const tensor &logits) {
      auto logits_value = as_vector(dy::_cg().incremental_forward(logits));

      float max_value = logits_value[0];
      unsigned max_index = 0;
      for (unsigned i = 1; i < logits_value.size(); ++i) {
        float val = logits_value[i];
        if (val > max_value) {
          max_value = val;
          max_index = i;
        }
      }
      return max_index;
    }

  }
}


#endif //DYNET_DYNET_WRAPPER_HPP
