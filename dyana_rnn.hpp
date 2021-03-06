//
// Created by YAN Yuchen on 5/1/2018.
//

#ifndef DYANA_LSTM_HPP
#define DYANA_LSTM_HPP

#include <dynet/lstm.h>
#include "dyana_common.hpp"
#include "dyana_operations.hpp"
#include "dyana_linear_layer.hpp"
#include <memory>

namespace dyana {

  typedef std::vector<dyana::tensor> rnn_cell_state_t;


  class rnn_cell_t {
  public:
    virtual std::pair<rnn_cell_state_t, dyana::tensor>
    operator()(const rnn_cell_state_t &prev_state, const dyana::tensor &x) = 0;
  };


  template<class RNN_CELL_T>
  class rnn {
  public:
    rnn() = default;

    rnn(const rnn &) = default;

    rnn(rnn &&) = default;

    rnn &operator=(const rnn &) = default;

    rnn &operator=(rnn &&) = default;

    typedef std::vector<rnn_cell_state_t> stacked_cell_state;

    rnn(unsigned num_stack, unsigned output_dim) : cells() {
      for (unsigned i = 0; i < num_stack; i++) {
        cells.emplace_back(output_dim);
      }
    }

    operator bool() const {
      return !cells.empty();
    }

    stacked_cell_state default_cell_state() const {
      stacked_cell_state ret;
      for (const auto &cell:cells) {
        ret.push_back(cell.default_cell_state());
      }
      return ret;
    }

    /**
     * apply the RNN cell for a single time step
     * \param prev_state the previous cell state.
     *                   pass {} for default initial value
     * \param x the current input
     * \return 0) the cell state after this time step
     *         1) the output
     */
    std::pair<stacked_cell_state, dyana::tensor>
    operator()(const stacked_cell_state &prev_state, const dyana::tensor &x) {
      tensor y = x;
      std::vector<rnn_cell_state_t> output_stacked_cell_state;
      for (unsigned i = 0; i < cells.size(); i++) {
        auto &cell = cells[i];
        auto _ = cell(i < prev_state.size() ? prev_state[i] : cell.default_cell_state(), y);

        if(use_residual()) {
          y = y + std::move(_.second);
        } else {
          y = std::move(_.second);
        }

        output_stacked_cell_state.push_back(std::move(_.first));
      }
      return std::make_pair(output_stacked_cell_state, y);
    }

    /**
     * apply the RNN cell for multiple time steps
     * \param prev_state the previous cell state
     *                   pass {} for default initial value
     * \param x_sequence a list of inputs to apply, in chronological order
     * \return 0) the cell state after the last time step
     *         1) the list of output in chronological order
     */
    std::pair<stacked_cell_state, std::vector<dyana::tensor>>
    operator()(const stacked_cell_state &prev_state, const std::vector<dyana::tensor> &x_sequence) {
      if (x_sequence.empty()) return std::make_pair(default_cell_state(), std::vector<dyana::tensor>());
      stacked_cell_state _state;
      dyana::tensor y;
      std::tie(_state, y) = operator()(prev_state, x_sequence[0]);
      std::vector<dyana::tensor> ys;
      ys.push_back(std::move(y));
      for (unsigned i = 1; i < x_sequence.size(); i++) {
        std::tie(_state, y) = operator()(_state, x_sequence[i]);
        ys.push_back(std::move(y));
      }
      return std::make_pair(std::move(_state), std::move(ys));
    }

    /**
     * apply the RNN cell for multiple time steps
     * \param x_sequence a list of inputs to apply, in chronological order
     * \return 0) the cell state after the last time step
     *         1) the list of output in chronological order
     */
    std::pair<stacked_cell_state, std::vector<dyana::tensor>>
    operator()(const std::vector<dyana::tensor> &x_sequence) {
      return operator()(default_cell_state(), x_sequence);
    }

    /**
     * concatenate a stacked-cell-state into a single expression,
     * useful if you need to feed this stacked-cell-state into some other network
     * \param scs the stacked-cell-state
     * \return
     */
    static dyana::tensor flatten_stacked_cell_state(const stacked_cell_state &scs) {
      std::vector<dyana::tensor> flatterned_exprs;
      for (const auto &cs:scs) {
        for (const auto &expr:cs) {
          flatterned_exprs.push_back(expr);
        }
      }
      return dyana::concatenate(flatterned_exprs);
    }

    stacked_cell_state unflatten_stacked_cell_state(const dyana::tensor &x) {
      stacked_cell_state ret;
      unsigned pivot = 0;
      unsigned length = x.dim()[0] / cells.size() / RNN_CELL_T::num_cell_state_parts;
      for (unsigned i = 0; i < cells.size(); i++) {
        rnn_cell_state_t cell_state;
        for (unsigned j = 0; j < RNN_CELL_T::num_cell_state_parts; j++) {
          cell_state.push_back(x.slice(pivot, pivot + length));
          pivot += length;
        }
        ret.push_back(cell_state);
      }
      return ret;
    }

    EASY_SERIALIZABLE(cells)

  protected:
    std::vector<RNN_CELL_T> cells;

    bool use_residual() {

      // use residual connection when # of stacks >= 4
      // reference: arXiv:1609.08144v2
      // Google’s Neural Machine Translation System: Bridging the Gap
      // between Human and Machine Translation
      return cells.size() >= 4;
    }
  };

  template<class RNN_CELL_T>
  class bidirectional_rnn {
  public:
    bidirectional_rnn() = default;

    bidirectional_rnn(const bidirectional_rnn &) = default;

    bidirectional_rnn(bidirectional_rnn &&) = default;

    bidirectional_rnn &operator=(const bidirectional_rnn &) = default;

    bidirectional_rnn &operator=(bidirectional_rnn &&) = default;

    bidirectional_rnn(unsigned num_stack, unsigned output_dim) : forward_rnn(num_stack, output_dim),
                                                                 backward_rnn(num_stack, output_dim) {}

    typedef typename rnn<RNN_CELL_T>::stacked_cell_state inner_cell_state;

    /**
     * apply the bidirectional-RNN to a sequence of time steps
     * \param x_sequence a list of inputs to apply
     * \return the outputs for each time step, concatenating outputs from both direction
     */
    std::vector<dyana::tensor>
    predict_output_sequence(const std::vector<dyana::tensor> &x_sequence) {
      auto forward_ys = forward_rnn(forward_rnn.default_cell_state(), x_sequence).second;
      auto reversed_xs = x_sequence;
      std::reverse(reversed_xs.begin(), reversed_xs.end());
      auto backward_ys = backward_rnn(backward_rnn.default_cell_state(), reversed_xs).second;
      std::reverse(backward_ys.begin(), backward_ys.end());
      std::vector<dyana::tensor> ret;
      for (unsigned i = 0; i < forward_ys.size(); i++) {
        ret.push_back(dyana::concatenate({forward_ys[i], backward_ys[i]}));
      }
      return ret;
    }

    /**
     * apply the bidirectional-RNN to a sequence of time steps
     * \param x_sequence a list of inputs to apply
     * \return concatenating the final output from both direction. i.e. forward output for t[-1] and reversed output for t[0]
     */
    dyana::tensor predict_output_final(const std::vector<dyana::tensor> &x_sequence) {
      auto forward_ys = forward_rnn(forward_rnn.default_cell_state(), x_sequence).second;
      auto reversed_xs = x_sequence;
      std::reverse(reversed_xs.begin(), reversed_xs.end());
      auto backward_ys = backward_rnn(backward_rnn.default_cell_state(), reversed_xs).second;
      return dyana::concatenate({forward_ys.back(), backward_ys.back()});
    }

  private:
    rnn<RNN_CELL_T> forward_rnn;
    rnn<RNN_CELL_T> backward_rnn;
  };


  class vanilla_lstm_cell_t : public rnn_cell_t {
  public:
    static constexpr unsigned num_cell_state_parts = 2;

    vanilla_lstm_cell_t() = default;

    vanilla_lstm_cell_t(const vanilla_lstm_cell_t &) = default;

    vanilla_lstm_cell_t(vanilla_lstm_cell_t &&) = default;

    vanilla_lstm_cell_t &operator=(const vanilla_lstm_cell_t &) = default;

    vanilla_lstm_cell_t &operator=(vanilla_lstm_cell_t &&) = default;

    vanilla_lstm_cell_t(unsigned output_dim) : output_dim(output_dim), input_dim(),
    Wx(),Wh(), b(){};

    rnn_cell_state_t default_cell_state() const {
      auto zeros = dyana::zeros({output_dim});
      return rnn_cell_state_t({zeros, zeros});
    }

    virtual std::pair<rnn_cell_state_t, dyana::tensor>
    operator()(const rnn_cell_state_t &prev_state, const dyana::tensor &x) {
      ensure_init(x);
      auto cell_state = prev_state.empty()?dyana::zeros({output_dim}):prev_state[0];
      auto hidden_state = prev_state.empty()?dyana::zeros({output_dim}):prev_state[1];

      auto gates_t = dynet::vanilla_lstm_gates({x}, hidden_state, dyana::tensor(Wx), dyana::tensor(Wh), dyana::tensor(b), 0);

      cell_state = dyana::tensor(dynet::vanilla_lstm_c(cell_state, gates_t));
      hidden_state = dynet::vanilla_lstm_h(cell_state, gates_t);
      return std::make_pair(rnn_cell_state_t{cell_state, hidden_state}, hidden_state);
    }

    EASY_SERIALIZABLE(output_dim, input_dim, Wx, Wh, b)

  private:
    void ensure_init(const dyana::tensor &x) {
      if(input_dim > 0) return;
      input_dim = x.dim()[0];
      Wx = dyana::parameter({output_dim*4, input_dim});
      Wh = dyana::parameter({output_dim*4, output_dim});
      b = dyana::parameter({output_dim*4});

      //initialize b to zeros
      b.set_values(std::vector<float>(output_dim*4, 0));
    }

    unsigned output_dim;
    unsigned input_dim;
    dyana::parameter Wx;
    dyana::parameter Wh;
    dyana::parameter b;
  };

  typedef rnn<vanilla_lstm_cell_t> vanilla_lstm;
  typedef bidirectional_rnn<vanilla_lstm_cell_t> bidirectional_vanilla_lstm;

  class coupled_lstm_cell_t : public rnn_cell_t {
  public:
    static constexpr unsigned num_cell_state_parts = 2;

    coupled_lstm_cell_t() = default;

    coupled_lstm_cell_t(const coupled_lstm_cell_t &) = default;

    coupled_lstm_cell_t(coupled_lstm_cell_t &&) = default;

    coupled_lstm_cell_t &operator=(const coupled_lstm_cell_t &) = default;

    coupled_lstm_cell_t &operator=(coupled_lstm_cell_t &&) = default;

    coupled_lstm_cell_t(unsigned output_dim) : output_dim(output_dim), input_dim(), Wx(), Wh(), b() {};

    rnn_cell_state_t default_cell_state() const {
      auto zeros = dyana::zeros({output_dim});
      return rnn_cell_state_t({zeros, zeros});
    }

    virtual std::pair<rnn_cell_state_t, dyana::tensor>
    operator()(const rnn_cell_state_t &prev_state, const dyana::tensor &x) {
      ensure_init(x);
      auto cell_state = prev_state.empty()?dyana::zeros({output_dim}):prev_state[0];
      auto hidden_state = prev_state.empty()?dyana::zeros({output_dim}):prev_state[1];

      auto gates_t = dyana::tensor(dynet::vanilla_lstm_gates({x}, hidden_state, dyana::tensor(Wx), dyana::tensor(Wh), dyana::tensor(b), 0));

      // turn the vanilla gates into coupled gates by setting input_gate to (1 - forget_gate)
      gates_t = dyana::concatenate({
        (float)1 - gates_t.slice(output_dim, output_dim*2),
        gates_t.slice(output_dim, output_dim*4)
      });

      cell_state = dyana::tensor(dynet::vanilla_lstm_c(cell_state, gates_t));
      hidden_state = dynet::vanilla_lstm_h(cell_state, gates_t);
      return std::make_pair(rnn_cell_state_t{cell_state, hidden_state}, hidden_state);
    }

    EASY_SERIALIZABLE(output_dim, input_dim, Wx, Wh, b)

  private:
    void ensure_init(const dyana::tensor &x) {
      if(input_dim > 0) return;
      input_dim = x.dim()[0];
      Wx = dyana::parameter({output_dim*4, input_dim});
      Wh = dyana::parameter({output_dim*4, output_dim});
      b = dyana::parameter({output_dim*4});

      //initialize b to zeros
      b.set_values(std::vector<float>(output_dim*4, 0));
    }

    unsigned output_dim;
    unsigned input_dim;
    dyana::parameter Wx;
    dyana::parameter Wh;
    dyana::parameter b;
  };

  typedef rnn<coupled_lstm_cell_t> coupled_lstm;
  typedef bidirectional_rnn<coupled_lstm_cell_t> bidirectional_coupled_lstm;

  class gru_cell_t : public rnn_cell_t {
  public:
    static constexpr unsigned num_cell_state_parts = 1;

    gru_cell_t() = default;

    gru_cell_t(const gru_cell_t &) = default;

    gru_cell_t(gru_cell_t &&) = default;

    gru_cell_t &operator=(const gru_cell_t &) = default;

    gru_cell_t &operator=(gru_cell_t &&) = default;

    gru_cell_t(unsigned output_dim) : output_dim(output_dim), pre_input_gate(output_dim), input_fc(output_dim),
                                      output_gate(output_dim) {};

    rnn_cell_state_t default_cell_state() const {
      return rnn_cell_state_t({dyana::zeros({output_dim})});
    }

    virtual std::pair<rnn_cell_state_t, dyana::tensor>
    operator()(const rnn_cell_state_t &prev_state, const dyana::tensor &x) {
      if (prev_state.empty()) {
        throw std::runtime_error("RNN: previous cell state empty. call default_cell_state to get a default one");
      }
      auto hidden = prev_state[0];
      auto input_for_gates = dyana::concatenate({hidden, x});
      auto pre_input_gate_coef = dyana::logistic(pre_input_gate.operator()(input_for_gates));
      auto output_gate_coef = dyana::logistic(output_gate.operator()(input_for_gates));
      auto gated_concat = dyana::concatenate({dyana::cmult(hidden, pre_input_gate_coef), x});
      auto output_candidate = dyana::tanh(input_fc.operator()(gated_concat));
      auto after_forget = dyana::cmult(hidden, (float)1.0 - output_gate_coef);
      auto output_hidden = after_forget + dyana::cmult(output_gate_coef, output_candidate);
      return std::make_pair(rnn_cell_state_t({output_hidden}), output_hidden);
    }

    EASY_SERIALIZABLE(output_dim, pre_input_gate, input_fc, output_gate)

  private:
    unsigned output_dim;
    linear_dense_layer pre_input_gate;
    linear_dense_layer input_fc;
    linear_dense_layer output_gate;
  };

  typedef rnn<gru_cell_t> gru;
  typedef bidirectional_rnn<gru_cell_t> bidirectional_gru;
}

#endif //DYANA_LSTM_HPP
