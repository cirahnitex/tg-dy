//
// Created by Dekai WU and YAN Yuchen on 20190322.
//

#ifndef DYANA_TENSOR_HPP
#define DYANA_TENSOR_HPP
#include <dynet/dynet.h>
#include <dynet/expr.h>
#include "dyana_dirty_core.hpp"
#include <iostream>
namespace tg {
  namespace dyana {

    class parameter {
    public:
      std::shared_ptr<dynet::Parameter> _dynet_parameter_m;
      static std::unordered_set<std::shared_ptr<dynet::Parameter>>& alives() {
        static std::unordered_set<std::shared_ptr<dynet::Parameter>> _;
        return _;
      };

      /**
       * get the number of parameters that should be garbage collected
       * \return
       */
      static unsigned long &num_dead() {
        static unsigned long _;
        return _;
      }

      parameter() = default;
      parameter(const parameter&) = default;
      parameter(parameter&&) noexcept = default;
      parameter &operator=(const parameter&) = default;
      parameter &operator=(parameter&&) noexcept = default;
      parameter(const Dim& dim): _dynet_parameter_m(std::make_shared<dynet::Parameter>(_pc()->add_parameters(dim).p)) {
        alives().insert(_dynet_parameter_m);
      }
      ~parameter() {
        // if this object is the last one holding its parameter storage pointer, mark it as "dead".
        if(_dynet_parameter_m.use_count()==2) {
          alives().erase(_dynet_parameter_m);
          num_dead()++;
        }
      }

      bool is_nil() const {return (bool)_dynet_parameter_m;}
      Dim dim() const {return _dynet_parameter_m->dim();}
      std::vector<float> get_values() const {return dynet::as_vector(*_dynet_parameter_m->values());}
      void set_values(const std::vector<float>& vs){_dynet_parameter_m->set_value(vs);}

      template<class Archive>
      void save(Archive& archive) const {
        archive(_dynet_parameter_m);
      }

      template<class Archive> void load(Archive& archive) {
        if(_dynet_parameter_m && _dynet_parameter_m.use_count()<=2) {
          alives().erase(_dynet_parameter_m);
          num_dead()++;
        }

        archive(_dynet_parameter_m);
        alives().insert(_dynet_parameter_m);
      }
    };

    class tensor : public dynet::Expression {
    public:
      tensor() : dynet::Expression() { increment_cnt(); };

      tensor(const dynet::Expression &x) : dynet::Expression(x) { increment_cnt(); };

      tensor(const dyana::tensor &x) : dynet::Expression(x) { increment_cnt(); };

      tensor(dynet::Expression &&x) : dynet::Expression(x) { increment_cnt(); };

      tensor(dyana::tensor &&x) : dynet::Expression(x) { increment_cnt(); };

      tensor(const dyana::parameter &x) : dynet::Expression(dynet::parameter(_cg(), *x._dynet_parameter_m)) { increment_cnt(); }

      explicit tensor(float x) : dynet::Expression(dynet::input(dyana::_cg(), x)) { increment_cnt(); }

      tensor(const std::vector<float> x) : dynet::Expression(
        dynet::input(dyana::_cg(), {(unsigned) x.size()}, x)) { increment_cnt(); }

      tensor(const std::initializer_list<float> x) : dynet::Expression(
        dynet::input(dyana::_cg(), {(unsigned) x.size()}, x)) { increment_cnt(); }

      tensor(const std::vector<float> &values, const dynet::Dim &dim) : dynet::Expression(
        dynet::input(dyana::_cg(), dim, values)) { increment_cnt(); }

      tensor(const std::initializer_list<float> &values, const dynet::Dim &dim) : dynet::Expression(
        dynet::input(dyana::_cg(), dim, values)) { increment_cnt(); }

      tensor &operator=(const dynet::Expression &x) {
        dynet::Expression::operator=(x);
        return *this;
      };

      tensor &operator=(const dyana::tensor &x) {
        dynet::Expression::operator=(x);
        return *this;
      };

      tensor &operator=(dynet::Expression &&x) {
        dynet::Expression::operator=(x);
        return *this;
      };

      tensor &operator=(dyana::tensor &&x) {
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
      tensor at(unsigned v, unsigned d=0) const {
        if(dim().batch_size()<=d) throw std::runtime_error("tenso::at: required dimension out of range");
        if(dim()[d]<=v) throw std::runtime_error("tenso::at: required index out of range");
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
      tensor slice(unsigned s, unsigned e, unsigned d=0) const {
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
      tensor select_rows(const std::vector<unsigned> &rows) const {
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
      tensor select_cols(const std::vector<unsigned> &cols) const {
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
      tensor reshape(const Dim &d) const {
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
      tensor transpose(const std::vector<unsigned> &dims = {1, 0}) const {
        return dynet::transpose(*this, dims);
      }

      float as_scalar() const { return dynet::as_scalar(dyana::_cg().incremental_forward(*this)); }

      std::vector<float> as_vector() const { return dynet::as_vector(dyana::_cg().incremental_forward(*this)); }

      ~tensor() {
        num_exprs()--;
        if (num_exprs() == 0) dyana::_renew_cg();
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

    class lookup_parameter {
    public:
      std::shared_ptr<dynet::LookupParameter> _dynet_parameter_m;
      static std::unordered_set<std::shared_ptr<dynet::LookupParameter>>& alives() {
        static std::unordered_set<std::shared_ptr<dynet::LookupParameter>> _;
        return _;
      };

      /**
       * get the number of parameters that should be garbage collected
       * \return
       */
      static unsigned long &num_dead() {
        static unsigned long _;
        return _;
      }

      lookup_parameter() = default;
      lookup_parameter(const lookup_parameter&) = default;
      lookup_parameter(lookup_parameter&&) noexcept = default;
      lookup_parameter &operator=(const lookup_parameter&) = default;
      lookup_parameter &operator=(lookup_parameter&&) noexcept = default;
      lookup_parameter(unsigned size, const Dim& dim): _dynet_parameter_m(std::make_shared<dynet::LookupParameter>(_pc()->add_lookup_parameters(size, dim).p)) {
        alives().insert(_dynet_parameter_m);
      }
      ~lookup_parameter() {
        // if this object is the last one holding its parameter storage pointer, mark it as "dead".
        if(_dynet_parameter_m.use_count()==2) {
          alives().erase(_dynet_parameter_m);
          num_dead()++;
        }
      }

      Dim dim() const {return _dynet_parameter_m->dim();}
      bool is_nil() const {return (bool)_dynet_parameter_m;}
      std::vector<std::vector<float>> get_values() const {
        auto tensors = _dynet_parameter_m->values();
        std::vector<std::vector<float>> ret;
        ret.reserve(tensors->size());
        for(const auto& tensor:*tensors) {
          ret.push_back(dynet::as_vector(tensor));
        }
        return ret;
      }
      void set_values(const std::vector<std::vector<float>>& vs){
        for(unsigned i=0; i<vs.size(); ++i) {
          _dynet_parameter_m->initialize(i, vs[i]);
        }
      }

      void initialize(unsigned index, const std::vector<float>& values) {_dynet_parameter_m->initialize(index, values);}

      tensor lookup(unsigned index) const {
        return dynet::lookup(_cg(), *_dynet_parameter_m, index);
      }

      tensor const_lookup(unsigned index) const {
        return dynet::const_lookup(_cg(), *_dynet_parameter_m, index);
      }

      template<class Archive>
      void save(Archive& archive) const {
        archive(_dynet_parameter_m);
      }

      template<class Archive> void load(Archive& archive) {
        if(_dynet_parameter_m && _dynet_parameter_m.use_count()<=2) {
          alives().erase(_dynet_parameter_m);
          num_dead()++;
        }

        archive(_dynet_parameter_m);
        alives().insert(_dynet_parameter_m);
      }
    };
    /**
    * get the const Expression associated with a given Parameter
    * (Expression can be used in computation graph to define neural network topology)
    * (Parameters under const Expression will not be updated by this Expression during training)
    * \param p the Parameter
    * \return the const Expression
    */
    inline tensor const_expr(const parameter &p) {
      return dynet::const_parameter(_cg(), *p._dynet_parameter_m);
    }

    inline void _force_garbage_collection() {
      if(parameter::num_dead()<=0 && lookup_parameter::num_dead()<=0) return;
      auto new_pc = new dynet::ParameterCollection();
      for(const auto& dp:parameter::alives()) {
        auto new_p = new_pc->add_parameters(dp->dim());
        new_p.set_value(dynet::as_vector(*(dp->values())));
        *dp = new_p;
      }
      for(const auto& dp:lookup_parameter::alives()) {
        auto n = dp->get_storage().values.size();
        auto new_p = new_pc->add_lookup_parameters(n, dp->dim());
        for(unsigned i=0; i<dp->values()->size(); ++i) {
          new_p.initialize(i, dynet::as_vector(dp->values()->at(i)));
        }
        *dp = new_p;
      }
      delete dyana::_pc();
      dyana::_pc() = new_pc;
      parameter::num_dead() = 0;
      lookup_parameter::num_dead() = 0;
    }
  }
}
#endif //DYANA_TENSOR_HPP