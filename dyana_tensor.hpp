//
// Created by Dekai WU and YAN Yuchen on 20190322.
//

#ifndef DYANA_TENSOR_HPP
#define DYANA_TENSOR_HPP

#include <dynet/dynet.h>
#include <dynet/expr.h>
#include "dyana_dirty_core.hpp"
#include "dyana_serialization_helper.hpp"
#include <iostream>
#include "dyana_guard.macro.hpp"
#include "dyana_event_emitter.hpp"

namespace dyana {

  /**
   * when guarded, the model is currently under multiprocess training
   */
  DEFINE_THREAD_LOCAL_GUARAD(multiprocessing_guard)

  class parameter {
  public:
    static constexpr char MP_PARAM_INIT_ERR_MSG[] = "cannot initialize parameters while multi-process training. please call your model's compute_loss function on some good piece of data mannually before training, to ensure that all parameters have been initialized";

    std::shared_ptr<dynet::Parameter> _dynet_parameter_m;

    static std::unordered_set<std::shared_ptr<dynet::Parameter>>& alives() {
      static std::unordered_set<std::shared_ptr<dynet::Parameter>> _;
      return _;
    };

    /**
     * get the number of parameters that should be garbage collected
     * \return
     */
    static unsigned long& num_dead() {
      static unsigned long _;
      return _;
    }

    parameter() = default;

    parameter(const parameter&) = default;

    parameter(parameter&&) noexcept = default;

    parameter& operator=(const parameter&) = default;

    parameter& operator=(parameter&&) noexcept = default;

    parameter(const Dim& dim) : _dynet_parameter_m(std::make_shared<dynet::Parameter>(_pc()->add_parameters(dim).p)) {
      alives().insert(_dynet_parameter_m);
      if(multiprocessing_guard::is_guarded()) throw std::runtime_error(MP_PARAM_INIT_ERR_MSG);
    }

    ~parameter() {
      // if this object is the last one holding its parameter storage pointer, mark it as "dead".
      if (_dynet_parameter_m.use_count() == 2) {
        alives().erase(_dynet_parameter_m);
        num_dead()++;
      }
    }

    explicit operator bool() const { return (bool) _dynet_parameter_m; }

    Dim dim() const { return _dynet_parameter_m->dim(); }

    std::vector<float> get_values() const { return dynet::as_vector(*_dynet_parameter_m->values()); }

    void set_values(const std::vector<float>& vs) { _dynet_parameter_m->set_value(vs); }

    template<class Archive>
    void save(Archive& archive) const {
      auto valid = (bool) _dynet_parameter_m;
      archive(cereal::make_nvp("valid", valid));
      if (valid) dynet::save(archive, *_dynet_parameter_m);
    }

    template<class Archive>
    void load(Archive& archive) {
      using namespace std;
      if (_dynet_parameter_m && _dynet_parameter_m.use_count() <= 2) {
        alives().erase(_dynet_parameter_m);
        num_dead()++;
      }
      bool valid;
      archive(cereal::make_nvp("valid", valid));
      if (valid) {
        _dynet_parameter_m = std::make_shared<dynet::Parameter>();
        dynet::load(archive, *_dynet_parameter_m);
        alives().insert(_dynet_parameter_m);
      }
    }
  };

  template<typename NUMBER_RANGE>
  std::vector<float> _convert_to_float_range(NUMBER_RANGE&& xs) {
    std::vector<float> ret;
    ret.reserve(xs.size());
    for (auto&& x:xs) {
      ret.push_back((float) x);
    }
    return ret;
  }

  /**
   * computations defined when guarded applys no gradients to parameters
   */
  DEFINE_THREAD_LOCAL_GUARAD(const_guard)

  class tensor : public dynet::Expression {
  public:
    tensor() : dynet::Expression() { increment_cnt(); };

    tensor(const dynet::Expression& x) : dynet::Expression(x) { increment_cnt(); };

    tensor(const dyana::tensor& x) : dynet::Expression(x) { increment_cnt(); };

    tensor(dynet::Expression&& x) : dynet::Expression(x) { increment_cnt(); };

    tensor(dyana::tensor&& x) noexcept : dynet::Expression(x) { increment_cnt(); };

    tensor(const dyana::parameter& x) : dynet::Expression(
      get_param_expr(x._dynet_parameter_m)) { increment_cnt(); }

    explicit tensor(float x) : dynet::Expression(dynet::input(dyana::_cg(), x)) { increment_cnt(); }

    explicit tensor(const std::vector<float> x) : dynet::Expression(
      dynet::input(dyana::_cg(), {(unsigned) x.size()}, x)) { increment_cnt(); }

    explicit tensor(const std::vector<double> x) : dynet::Expression(
      dynet::input(dyana::_cg(), {(unsigned) x.size()}, _convert_to_float_range(x))) { increment_cnt(); }

    explicit tensor(const std::vector<bool> x) : dynet::Expression(
      dynet::input(dyana::_cg(), {(unsigned) x.size()}, _convert_to_float_range(x))) { increment_cnt(); }

    template<typename T>
    explicit tensor(const std::vector<T>&& x) : dynet::Expression(
      dynet::input(dyana::_cg(), {(unsigned) x.size()}, _convert_to_float_range(x))) { increment_cnt(); }

    tensor(const std::initializer_list<float> x) : dynet::Expression(
      dynet::input(dyana::_cg(), {(unsigned) x.size()}, x)) { increment_cnt(); }

    tensor(const std::vector<float>& values, const dynet::Dim& dim) : dynet::Expression(
      dynet::input(dyana::_cg(), dim, values)) { increment_cnt(); }

    tensor(const std::vector<double>& values, const dynet::Dim& dim) : dynet::Expression(
      dynet::input(dyana::_cg(), dim, _convert_to_float_range(values))) { increment_cnt(); }

    tensor(const std::vector<bool>& values, const dynet::Dim& dim) : dynet::Expression(
      dynet::input(dyana::_cg(), dim, _convert_to_float_range(values))) { increment_cnt(); }

    tensor(const std::initializer_list<float>& values, const dynet::Dim& dim) : dynet::Expression(
      dynet::input(dyana::_cg(), dim, values)) { increment_cnt(); }

    tensor& operator=(const dynet::Expression& x) {
      dynet::Expression::operator=(x);
      return *this;
    };

    tensor& operator=(const dyana::tensor& x) {
      dynet::Expression::operator=(x);
      return *this;
    };

    tensor& operator=(dynet::Expression&& x) {
      dynet::Expression::operator=(x);
      return *this;
    };

    tensor& operator=(dyana::tensor&& x) noexcept {
      dynet::Expression::operator=(x);
      return *this;
    };

    explicit operator bool() const {
      return pg;
    }

    /**
     * \brief Pick element
     * \details Pick a single element/row/column/sub-tensor.
     *          This will result in the dimension of the tensor being reduced
     *          by 1.
     *
     * \param v The index of the element to select
     * \param d The dimension along which to choose the element
     *
     * \return The value of x[v] along dimension d
     */
    tensor at(unsigned v, unsigned d = 0) const {
      if (dim().ndims() <= d) throw std::runtime_error("tenso::at: required dimension out of range");
      if (dim()[d] <= v) throw std::runtime_error("tenso::at: required index out of range");
      return dynet::pick(*this, v, d);
    }

    /**
     * \brief Pick range of elements
     * \details Pick a range of elements from a tensor.
     *
     * \param s The start index
     * \param e The end index, excluding itself
     * \param d The dimension along which to pick
     *
     * \return The value of {x[s],x[s+1]...,x[e-1]}
     */
    tensor slice(unsigned s, unsigned e, unsigned d = 0) const {
      return dynet::pick_range(*this, s, e, d);
    }

    /**
     * \brief split a tensor into vector of tensors
     * \param d the dimension along which to split
     * \return the value of {x[0],x[1],...,x[n-1]}
     */
    std::vector<tensor> split(unsigned d = 1) const {
      auto size = dim()[d];
      std::vector<tensor> ret;
      ret.reserve(size);
      for (unsigned i = 0; i < size; i++) {
        ret.push_back(at(i, d));
      }
      return ret;
    }

    /**
     * \brief Select rows
     * \details Select a subset of rows of a matrix.
     *
     * \param rows The rows to extract
     *
     * \return An expression containing the selected rows
     */
    tensor select_rows(const std::vector<unsigned>& rows) const {
      return dynet::select_rows(*this, rows);
    }

    /**
     * \brief Select columns
     * \details Select a subset of columns of a matrix. select_cols is more
     *          efficient than select_rows since DyNet uses column-major order.
     *
     * \param cols The columns to extract
     *
     * \return An expression containing the selected columns
     */
    tensor select_cols(const std::vector<unsigned>& cols) const {
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
     * \param d The new dimensions
     *
     * \return The reshaped expression
     */
    tensor reshape(const Dim& d) const {
      return dynet::reshape(*this, d);
    }

    /**
     * \brief Transpose a matrix
     * \details Transpose a matrix or tensor, or if dims is specified shuffle the
     *          dimensions arbitrarily.
     *          **Note:** This is O(1) if either the row or column dimension is 1,
     *          and O(n) otherwise.
     *
     * \param dims The dimensions to swap. The ith dimension of the output will be equal
     *          to the dims[i] dimension of the input. dims must have the same number
     *          of dimensions as x.
     *
     * \return The transposed/shuffled expression
     */
    tensor transpose(const std::vector<unsigned>& dims = {1, 0}) const {
      return dynet::transpose(*this, dims);
    }

    float as_scalar() const {
      return dynet::as_scalar(dyana::_cg().incremental_forward(*this));
    }

    std::vector<float> as_vector() const {
      return dynet::as_vector(dyana::_cg().incremental_forward(*this));
    }

    ~tensor() {
      num_exprs()--;
      if (num_exprs() == 0) clear_tensors();
    }

    static std::vector<dynet::Expression> vector_cast_to_base(const std::vector<tensor>& x) {
      return std::vector<dynet::Expression>(x.begin(), x.end());
    }

    static std::vector<tensor> vector_cast_to_parent(const std::vector<dynet::Expression>& x) {
      return std::vector<tensor>(x.begin(), x.end());
    }

    static unsigned get_exprs_counter() { return num_exprs(); }

    template<typename Archive>
    void save(Archive& ar) const {
      ar(cereal::make_nvp("valid", (bool) pg));
      if (!pg) return;
      ar(cereal::make_nvp("dim", dim()));
      ar(cereal::make_nvp("data", as_vector()));
    }

    template<typename Archive>
    void load(Archive& ar) {
      bool valid;
      ar(cereal::make_nvp("valid", valid));
      if (!valid) return;
      Dim dim;
      ar(cereal::make_nvp("dim", dim));
      std::vector<float> data;
      ar(cereal::make_nvp("data", data));
      operator=(tensor(data, dim));
    }

    static event_emitter<>::listener_handle_t add_before_renew_cg_listener(const event_emitter<>::listener_t& listener) {
      return before_renew_cg_evt().add_listener(listener);
    }

    static void remove_before_renew_cg_listener(const event_emitter<>::listener_handle_t& listener) {
      before_renew_cg_evt().remove_listener(listener);
    }

  private:
    static event_emitter<>& before_renew_cg_evt() {
      static event_emitter<> _;
      return _;
    };

    static unsigned long& num_exprs() {
      thread_local static unsigned long _{0};
      return _;
    }

    void increment_cnt() { num_exprs()++; };

    static std::unordered_map<std::shared_ptr<dynet::Parameter>, dynet::Expression>& param_expr_map(bool is_const) {
      static thread_local std::unordered_map<std::shared_ptr<dynet::Parameter>, dynet::Expression> c, v;
      return is_const?c:v;
    }

    static void clear_tensors() {
      before_renew_cg_evt().fire();
      param_expr_map(true).clear();
      param_expr_map(false).clear();
      dyana::_renew_cg();
    }

    static const dynet::Expression& get_param_expr(const std::shared_ptr<dynet::Parameter>& dynet_param) {
      auto&& pem = param_expr_map(const_guard::is_guarded());
      try {
        return pem.at(dynet_param);
      }
      catch(...) {
        return pem[dynet_param] =const_guard::is_guarded()?dynet::const_parameter(_cg(), *dynet_param):dynet::parameter(_cg(), *dynet_param);
      }
    }
  };

  class lookup_parameter {
    event_emitter<>::listener_handle_t renew_cg_handler;
    mutable std::unordered_map<unsigned, dynet::Expression> lookup_cache{};
    mutable std::unordered_map<unsigned, dynet::Expression> const_lookup_cache{};
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
    static unsigned long& num_dead() {
      static unsigned long _;
      return _;
    }

    lookup_parameter():renew_cg_handler(tensor::add_before_renew_cg_listener([&](){handle_renew_cg();})){
    };

    lookup_parameter(const lookup_parameter& x):renew_cg_handler(tensor::add_before_renew_cg_listener([&](){handle_renew_cg();})), _dynet_parameter_m(x._dynet_parameter_m){
    };

    lookup_parameter(lookup_parameter&& x) noexcept :renew_cg_handler(tensor::add_before_renew_cg_listener([&](){handle_renew_cg();})), _dynet_parameter_m(x._dynet_parameter_m){
      tensor::remove_before_renew_cg_listener(x.renew_cg_handler);
    };

    lookup_parameter& operator=(const lookup_parameter& x) {
      handle_renew_cg();
      _dynet_parameter_m = x._dynet_parameter_m;
      return *this;
    };

    lookup_parameter& operator=(lookup_parameter&& x) noexcept {
      handle_renew_cg();
      tensor::remove_before_renew_cg_listener(x.renew_cg_handler);
      _dynet_parameter_m = x._dynet_parameter_m;
      return *this;
    }

    lookup_parameter(unsigned size, const Dim& dim) :
      renew_cg_handler(tensor::add_before_renew_cg_listener([&](){handle_renew_cg();})),
    _dynet_parameter_m(
      std::make_shared<dynet::LookupParameter>(_pc()->add_lookup_parameters(size, dim).p)) {
      alives().insert(_dynet_parameter_m);
      if(multiprocessing_guard::is_guarded()) throw std::runtime_error(parameter::MP_PARAM_INIT_ERR_MSG);
    }

    ~lookup_parameter() {
      tensor::remove_before_renew_cg_listener(renew_cg_handler);
      // if this object is the last one holding its parameter storage pointer, mark it as "dead".
      if (_dynet_parameter_m.use_count() == 2) {
        alives().erase(_dynet_parameter_m);
        num_dead()++;
      }
    }

    Dim dim() const { return _dynet_parameter_m->dim(); }

    explicit operator bool() const { return (bool) _dynet_parameter_m; }

    std::vector<std::vector<float>> get_values() const {
      auto tensors = _dynet_parameter_m->values();
      std::vector<std::vector<float>> ret;
      ret.reserve(tensors->size());
      for (const auto& tensor:*tensors) {
        ret.push_back(dynet::as_vector(tensor));
      }
      return ret;
    }

    void set_values(const std::vector<std::vector<float>>& vs) {
      for (unsigned i = 0; i < vs.size(); ++i) {
        _dynet_parameter_m->initialize(i, vs[i]);
      }
    }

    void set_value(unsigned i, const std::vector<float>& value) {
      _dynet_parameter_m->initialize(i, value);
    }

    void initialize(unsigned index, const std::vector<float>& values) { _dynet_parameter_m->initialize(index, values); }

    tensor lookup(unsigned index) const {
      if(const_guard::is_guarded()) {
        auto cached_result = const_lookup_cache.find(index);
        if(cached_result != const_lookup_cache.end()) {
          return cached_result->second;
        }
        else {
          auto ret = dynet::const_lookup(_cg(), *_dynet_parameter_m, index);
          const_lookup_cache[index] = ret;
          return ret;
        }
      }
      else {
        auto cached_result = lookup_cache.find(index);
        if(cached_result != lookup_cache.end()) {
          return cached_result->second;
        }
        else {
          auto ret = dynet::lookup(_cg(), *_dynet_parameter_m, index);
          lookup_cache[index] = ret;
          return ret;
        }
      }
    }

    tensor lookup(const std::vector<unsigned>& indices) const {
      if(const_guard::is_guarded()) {
        return dynet::const_lookup(_cg(), *_dynet_parameter_m, indices);
      }
      else {
        return dynet::lookup(_cg(), *_dynet_parameter_m, indices);
      }
    }

    template<class Archive>
    void save(Archive& archive) const {
      using namespace std;
      auto valid = (bool) _dynet_parameter_m;
      archive(cereal::make_nvp("valid", valid));
      if (valid) dynet::save(archive, *_dynet_parameter_m);
    }

    template<class Archive>
    void load(Archive& archive) {
      if (_dynet_parameter_m && _dynet_parameter_m.use_count() <= 2) {
        alives().erase(_dynet_parameter_m);
        num_dead()++;
      }

      bool valid;
      archive(cereal::make_nvp("valid", valid));
      if (valid) {
        _dynet_parameter_m = std::make_shared<dynet::LookupParameter>();
        dynet::load(archive, *_dynet_parameter_m);
        alives().insert(_dynet_parameter_m);
      }
    }
  private:
    void handle_renew_cg() {
      lookup_cache.clear();
      const_lookup_cache.clear();
    }
  };

  /**
  * get the const Expression associated with a given Parameter
  * (Expression can be used in computation graph to define neural network topology)
  * (Parameters under const Expression will not be updated by this Expression during training)
  * \param p the Parameter
  * \return the const Expression
  */
  inline tensor const_expr(const parameter& p) {
    return dynet::const_parameter(_cg(), *p._dynet_parameter_m);
  }

  inline void _force_garbage_collection() {
    if (parameter::num_dead() <= 0 && lookup_parameter::num_dead() <= 0) return;
    using namespace std;
    auto new_pc = new dynet::ParameterCollection();
    for (const auto& dp:parameter::alives()) {
      auto new_p = new_pc->add_parameters(dp->dim());
      new_p.set_value(dynet::as_vector(*(dp->values())));
      *dp = new_p;
    }
    for (const auto& dp:lookup_parameter::alives()) {
      auto n = dp->get_storage().values.size();
      auto new_p = new_pc->add_lookup_parameters(n, dp->dim());
      for (unsigned i = 0; i < dp->values()->size(); ++i) {
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

#endif //DYANA_TENSOR_HPP
