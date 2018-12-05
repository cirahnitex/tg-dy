//
// Created by YAN Yuchen on 5/1/2018.
//

#ifndef DYNET_WRAPPER_DY_LSTM_HPP
#define DYNET_WRAPPER_DY_LSTM_HPP

#include <dynet/lstm.h>
#include "dy_common.hpp"
#include "dy_operations.hpp"
#include "dy_linear_layer.hpp"
#include <memory>

namespace tg {
  namespace dy {

    typedef std::vector<dy::tensor> rnn_cell_state_t;


    class rnn_cell_t {
    public:
      virtual std::pair<rnn_cell_state_t, dy::tensor>
      forward(const rnn_cell_state_t &prev_state, const dy::tensor &x) = 0;
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

      rnn(unsigned num_stack, unsigned hidden_dim) : cells() {
        for(unsigned i=0; i<num_stack; i++) {
          cells.emplace_back(hidden_dim);
        }
      }

      stacked_cell_state default_cell_state() const {
        stacked_cell_state ret;
        for(const auto& cell:cells) {
          ret.push_back(cell.default_cell_state());
        }
        return ret;
      }

      /**
       * apply the RNN cell for a single time step
       * \param prev_state the previous cell state
       * \param x the current input
       * \return 0) the cell state after this time step
       *         1) the output
       */
      std::pair<stacked_cell_state, dy::tensor>
      predict(const stacked_cell_state &prev_state, const dy::tensor &x) {
        tensor y = x;
        std::vector<rnn_cell_state_t> output_stacked_cell_state;
        for (unsigned i = 0; i < cells.size(); i++) {
          auto &cell = cells[i];
          auto _ = cell.forward(i < prev_state.size() ? prev_state[i] : rnn_cell_state_t(), y);
          y = std::move(_.second);
          output_stacked_cell_state.push_back(std::move(_.first));
        }
        return std::make_pair(output_stacked_cell_state, y);
      }

      /**
       * apply the RNN cell for multiple time steps
       * \param prev_state the previous cell state
       * \param x_sequence a list of inputs to apply, in chronological order
       * \return 0) the cell state after the last time step
       *         1) the list of output in chronological order
       */
      std::pair<stacked_cell_state, std::vector<dy::tensor>>
      predict(const stacked_cell_state &prev_state, const std::vector<dy::tensor> &x_sequence) {
        if (x_sequence.empty()) return std::make_pair(default_cell_state(), std::vector<dy::tensor>());
        auto[_state, y] = predict(prev_state, x_sequence[0]);
        std::vector<dy::tensor> ys;
        ys.push_back(std::move(y));
        for (unsigned i = 1; i < x_sequence.size(); i++) {
          std::tie(_state, y) = predict(_state, x_sequence[i]);
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
      std::pair<stacked_cell_state, std::vector<dy::tensor>>
      predict(const std::vector<dy::tensor> &x_sequence) {
        return predict(stacked_cell_state(), x_sequence);
      }

      /**
       * concatenate a stacked-cell-state into a single expression,
       * useful if you need to feed this stacked-cell-state into some other network
       * \param scs the stacked-cell-state
       * \return
       */
      static dy::tensor flatten_stacked_cell_state(const stacked_cell_state& scs) {
        std::vector<dy::tensor> flatterned_exprs;
        for(const auto& cs:scs) {
          for(const auto& expr:cs) {
            flatterned_exprs.push_back(expr);
          }
        }
        return dy::concatenate(flatterned_exprs);
      }

      stacked_cell_state unflatten_stacked_cell_state(const dy::tensor& x) {
        stacked_cell_state ret;
        unsigned pivot = 0;
        unsigned length = x.dim()[0] / cells.size() / RNN_CELL_T::num_cell_state_parts;
        for(unsigned i=0; i<cells.size(); i++) {
          rnn_cell_state_t cell_state;
          for(unsigned j=0; j<RNN_CELL_T::num_cell_state_parts; j++) {
            cell_state.push_back(x.slice(pivot, pivot+length));
            pivot += length;
          }
          ret.push_back(cell_state);
        }
        return ret;
      }

      EASY_SERIALZABLE(cells)

    protected:
      std::vector<RNN_CELL_T> cells;
    };

    template<class RNN_CELL_T>
    class bidirectional_rnn {
    public:
      bidirectional_rnn() = default;
      bidirectional_rnn(const bidirectional_rnn&) = default;
      bidirectional_rnn(bidirectional_rnn&&) = default;
      bidirectional_rnn &operator=(const bidirectional_rnn&) = default;
      bidirectional_rnn &operator=(bidirectional_rnn&&) = default;
      bidirectional_rnn(unsigned num_stack, unsigned hidden_dim) : forward_rnn(num_stack, hidden_dim), backward_rnn(num_stack, hidden_dim) {}
      typedef typename rnn<RNN_CELL_T>::stacked_cell_state inner_cell_state;

      /**
       * apply the bidirectional-RNN to a sequence of time steps
       * \param x_sequence a list of inputs to apply
       * \return the outputs for each time step, concatenating outputs from both direction
       */
      std::vector<dy::tensor>
      predict_output_sequence(const std::vector<dy::tensor> &x_sequence) {
        auto forward_ys= forward_rnn.predict(forward_rnn.default_cell_state(), x_sequence).second;
        auto reversed_xs = x_sequence;
        std::reverse(reversed_xs.begin(), reversed_xs.end());
        auto backward_ys = backward_rnn.predict(backward_rnn.default_cell_state(), reversed_xs).second;
        std::reverse(backward_ys.begin(), backward_ys.end());
        std::vector<dy::tensor> ret;
        for(unsigned i=0; i<forward_ys.size(); i++) {
          ret.push_back(dy::concatenate({forward_ys[i], backward_ys[i]}));
        }
        return ret;
      }

      /**
       * apply the bidirectional-RNN to a sequence of time steps
       * \param x_sequence a list of inputs to apply
       * \return concatenating the final output from both direction. i.e. forward output for t[-1] and reversed output for t[0]
       */
      dy::tensor predict_output_final(const std::vector<dy::tensor> &x_sequence) {
        auto forward_ys= forward_rnn.predict(forward_rnn.default_cell_state(), x_sequence).second;
        auto reversed_xs = x_sequence;
        std::reverse(reversed_xs.begin(), reversed_xs.end());
        auto backward_ys = backward_rnn.predict(backward_rnn.default_cell_state(), reversed_xs).second;
        return dy::concatenate({forward_ys.back(), backward_ys.back()});
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

      vanilla_lstm_cell_t(unsigned hidden_dim) : hidden_dim(hidden_dim), forget_gate(hidden_dim),
                                                 input_gate(hidden_dim), input_fc(hidden_dim),
                                                 output_gate(hidden_dim) {};

      rnn_cell_state_t default_cell_state() const {
        auto zeros = dy::zeros({hidden_dim});
        return rnn_cell_state_t({zeros, zeros});
      }

      virtual std::pair<rnn_cell_state_t, dy::tensor>
      forward(const rnn_cell_state_t &prev_state, const dy::tensor &x) {
        if(prev_state.empty()) {
          throw std::runtime_error("RNN: previous cell state empty. call default_cell_state to get a default one");
        }
        auto cell_state = prev_state[0];
        auto hidden_state = prev_state[1];
        auto concat = concatenate({hidden_state, x});
        auto after_forget = dy::cmult(cell_state, dy::logistic(forget_gate.predict(concat)));
        auto input_candidate = dy::tanh(input_fc.predict(concat));
        auto input = dy::cmult(dy::logistic(input_gate.predict(concat)), input_candidate);
        auto output_cell_state = after_forget + input;
        auto output_hidden_state = dy::cmult(dy::logistic(output_gate.predict(concat)), dy::tanh(output_cell_state));
        return std::make_pair(rnn_cell_state_t({std::move(output_cell_state),output_hidden_state}),output_hidden_state);
      }

      EASY_SERIALZABLE(hidden_dim, forget_gate, input_gate, input_fc, output_gate)

    private:
      unsigned hidden_dim;
      linear_layer forget_gate;
      linear_layer input_gate;
      linear_layer input_fc;
      linear_layer output_gate;
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

      coupled_lstm_cell_t(unsigned hidden_dim) : hidden_dim(hidden_dim), forget_gate(hidden_dim), input_fc(hidden_dim),
                                                 output_gate(hidden_dim) {};

      rnn_cell_state_t default_cell_state() const {
        auto zeros = dy::zeros({hidden_dim});
        return rnn_cell_state_t({zeros, zeros});
      }

      virtual std::pair<rnn_cell_state_t, dy::tensor>
      forward(const rnn_cell_state_t &prev_state, const dy::tensor &x) {
        if(prev_state.empty()) {
          throw std::runtime_error("RNN: previous cell state empty. call default_cell_state to get a default one");
        }
        auto cell_state = prev_state[0];
        auto hidden_state = prev_state[1];

        auto concat = concatenate({hidden_state, x});
        auto forget_coef = dy::logistic(forget_gate.predict(concat));
        auto after_forget = dy::cmult(cell_state, forget_coef);
        auto input_candidate = dy::tanh(input_fc.predict(concat));
        auto input = dy::cmult(1.0 - forget_coef, input_candidate);
        auto output_cell_state = after_forget + input;
        auto output_hidden_state = dy::cmult(dy::logistic(output_gate.predict(concat)), dy::tanh(output_cell_state));

        return std::make_pair(rnn_cell_state_t({std::move(output_cell_state),output_hidden_state}),output_hidden_state);
      }

      EASY_SERIALZABLE(hidden_dim, forget_gate, input_fc, output_gate)

    private:
      unsigned hidden_dim;
      linear_layer forget_gate;
      linear_layer input_fc;
      linear_layer output_gate;
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

      gru_cell_t(unsigned hidden_dim) : hidden_dim(hidden_dim), pre_input_gate(hidden_dim), input_fc(hidden_dim),
                                        output_gate(hidden_dim) {};

      rnn_cell_state_t default_cell_state() const {
        return rnn_cell_state_t({dy::zeros({hidden_dim})});
      }

      virtual std::pair<rnn_cell_state_t, dy::tensor>
      forward(const rnn_cell_state_t &prev_state, const dy::tensor &x) {
        if(prev_state.empty()) {
          throw std::runtime_error("RNN: previous cell state empty. call default_cell_state to get a default one");
        }
        auto hidden = prev_state[0];
        auto input_for_gates = dy::concatenate({hidden, x});
        auto pre_input_gate_coef = dy::logistic(pre_input_gate.predict(input_for_gates));
        auto output_gate_coef = dy::logistic(output_gate.predict(input_for_gates));
        auto gated_concat = dy::concatenate({dy::cmult(hidden, pre_input_gate_coef), x});
        auto output_candidate = dy::tanh(input_fc.predict(gated_concat));
        auto after_forget = dy::cmult(hidden, 1 - output_gate_coef);
        auto output_hidden = after_forget + dy::cmult(output_gate_coef, output_candidate);
        return std::make_pair(rnn_cell_state_t({output_hidden}), output_hidden);
      }

      EASY_SERIALZABLE(hidden_dim, pre_input_gate, input_fc, output_gate)

    private:
      unsigned hidden_dim;
      linear_layer pre_input_gate;
      linear_layer input_fc;
      linear_layer output_gate;
    };

    typedef rnn<gru_cell_t> gru;
    typedef bidirectional_rnn<gru_cell_t> bidirectional_gru;
  }
}
#endif //DYNET_WRAPPER_DY_LSTM_HPP
