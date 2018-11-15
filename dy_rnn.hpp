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

    template<class CELL_STATE_T>
    class rnn_cell_t {
    public:
      typedef CELL_STATE_T cell_state_type;

      virtual std::pair<cell_state_type, dy::Expression>
      forward(const cell_state_type &prev_state, const dy::Expression &x) = 0;
    };


    template<class RNN_CELL_T, class CELL_STATE_T>
    class rnn {
    public:
      rnn() = default;

      rnn(const rnn &) = default;

      rnn(rnn &&) = default;

      rnn &operator=(const rnn &) = default;

      rnn &operator=(rnn &&) = default;

      typedef std::vector<CELL_STATE_T> stacked_cell_state;

      explicit rnn(unsigned num_stack) : cells() {
        cells.resize(num_stack);
      }

      /**
       * apply the RNN cell for a single time step
       * \param prev_state the previous cell state
       * \param x the current input
       * \return 0) the cell state after this time step
       *         1) the output
       */
      std::pair<stacked_cell_state, dy::Expression>
      forward(const stacked_cell_state &prev_state, const dy::Expression &x) {
        Expression y = x;
        std::vector<CELL_STATE_T> output_stacked_cell_state;
        for (unsigned i = 0; i < cells.size(); i++) {
          auto &cell = cells[i];
          auto _ = cell.forward(i < prev_state.size() ? prev_state[i] : CELL_STATE_T(), y);
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
      std::pair<stacked_cell_state, std::vector<dy::Expression>>
      forward(const stacked_cell_state &prev_state, const std::vector<dy::Expression> &x_sequence) {
        if (x_sequence.empty()) return std::make_pair(stacked_cell_state(), std::vector<dy::Expression>());
        auto[_state, y] = forward(prev_state, x_sequence[0]);
        std::vector<dy::Expression> ys;
        ys.push_back(std::move(y));
        for (unsigned i = 1; i < x_sequence.size(); i++) {
          std::tie(_state, y) = forward(_state, x_sequence[i]);
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
      std::pair<stacked_cell_state, std::vector<dy::Expression>>
      forward(const std::vector<dy::Expression> &x_sequence) {
        return forward(stacked_cell_state(), x_sequence);
      }

      EASY_SERIALZABLE(cells)

    protected:
      std::vector<RNN_CELL_T> cells;
    };

    struct lstm_cell_state {
      Expression cell_state, hidden_state;
    };

    class vanilla_lstm_cell_t : public rnn_cell_t<lstm_cell_state> {
    public:
      vanilla_lstm_cell_t() : hidden_dim(0), forget_gate(), input_gate(), input_fc(), output_gate() {};

      vanilla_lstm_cell_t(const vanilla_lstm_cell_t &) = default;

      vanilla_lstm_cell_t(vanilla_lstm_cell_t &&) = default;

      vanilla_lstm_cell_t &operator=(const vanilla_lstm_cell_t &) = default;

      vanilla_lstm_cell_t &operator=(vanilla_lstm_cell_t &&) = default;

      virtual std::pair<lstm_cell_state, dy::Expression>
      forward(const lstm_cell_state &prev_state, const dy::Expression &x) {
        ensure_init(x);
        auto cell_state = prev_state.cell_state;
        if (cell_state.pg == nullptr) cell_state = zeros(x.dim());
        auto hidden_state = prev_state.hidden_state;
        if (hidden_state.pg == nullptr) hidden_state = zeros(x.dim());

        auto concat = concatenate({hidden_state, x});
        auto after_forget = dy::cmult(cell_state, dy::logistic(forget_gate.forward(concat)));
        auto input_candidate = dy::tanh(input_fc.forward(concat));
        auto input = dy::cmult(dy::logistic(input_gate.forward(concat)), input_candidate);
        auto output_cell_state = after_forget + input;
        auto output_hidden_state = dy::cmult(dy::logistic(output_gate.forward(concat)), dy::tanh(output_cell_state));
        lstm_cell_state ret;
        ret.cell_state = std::move(output_cell_state);
        ret.hidden_state = output_hidden_state;
        return std::make_pair(std::move(ret), output_hidden_state);
      }

      EASY_SERIALZABLE(hidden_dim, forget_gate, input_gate, input_fc, output_gate)

    private:
      void ensure_init(const Expression &x) {
        if (hidden_dim > 0) return;
        hidden_dim = x.dim()[0];
        forget_gate = linear_layer(hidden_dim);
        input_gate = linear_layer(hidden_dim);
        input_fc = linear_layer(hidden_dim);
        output_gate = linear_layer(hidden_dim);
      }

      unsigned hidden_dim;
      linear_layer forget_gate;
      linear_layer input_gate;
      linear_layer input_fc;
      linear_layer output_gate;
    };

    typedef rnn<vanilla_lstm_cell_t, lstm_cell_state> vanilla_lstm;

    class coupled_lstm_cell_t : public rnn_cell_t<lstm_cell_state> {
    public:
      coupled_lstm_cell_t() : hidden_dim(0), forget_gate(), input_fc(), output_gate() {};

      coupled_lstm_cell_t(const coupled_lstm_cell_t &) = default;

      coupled_lstm_cell_t(coupled_lstm_cell_t &&) = default;

      coupled_lstm_cell_t &operator=(const coupled_lstm_cell_t &) = default;

      coupled_lstm_cell_t &operator=(coupled_lstm_cell_t &&) = default;

      virtual std::pair<lstm_cell_state, dy::Expression>
      forward(const lstm_cell_state &prev_state, const dy::Expression &x) {
        ensure_init(x);
        auto cell_state = prev_state.cell_state;
        if (cell_state.pg == nullptr) cell_state = zeros(x.dim());
        auto hidden_state = prev_state.hidden_state;
        if (hidden_state.pg == nullptr) hidden_state = zeros(x.dim());

        auto concat = concatenate({hidden_state, x});
        auto forget_coef = dy::logistic(forget_gate.forward(concat));
        auto after_forget = dy::cmult(cell_state, forget_coef);
        auto input_candidate = dy::tanh(input_fc.forward(concat));
        auto input = dy::cmult(1.0 - forget_coef, input_candidate);
        auto output_cell_state = after_forget + input;
        auto output_hidden_state = dy::cmult(dy::logistic(output_gate.forward(concat)), dy::tanh(output_cell_state));

        lstm_cell_state ret;
        ret.cell_state = std::move(output_cell_state);
        ret.hidden_state = output_hidden_state;
        return std::make_pair(std::move(ret), output_hidden_state);
      }

      EASY_SERIALZABLE(hidden_dim, forget_gate, input_fc, output_gate)

    private:
      void ensure_init(const Expression &x) {
        if (hidden_dim > 0) return;
        hidden_dim = x.dim()[0];
        forget_gate = linear_layer(hidden_dim);
        input_fc = linear_layer(hidden_dim);
        output_gate = linear_layer(hidden_dim);
      }

      unsigned hidden_dim;
      linear_layer forget_gate;
      linear_layer input_fc;
      linear_layer output_gate;
    };

    typedef rnn<coupled_lstm_cell_t, lstm_cell_state> coupled_lstm;


    class gru_cell_t : public rnn_cell_t<dy::Expression> {
    public:
      gru_cell_t() : hidden_dim(0), pre_input_gate(), input_fc(), output_gate() {};

      gru_cell_t(const gru_cell_t &) = default;

      gru_cell_t(gru_cell_t &&) = default;

      gru_cell_t &operator=(const gru_cell_t &) = default;

      gru_cell_t &operator=(gru_cell_t &&) = default;

      virtual std::pair<dy::Expression, dy::Expression>
      forward(const dy::Expression &prev_state, const dy::Expression &x) {
        ensure_init(x);
        auto hidden = (prev_state.pg == nullptr)?dy::zeros({hidden_dim}):prev_state;
        auto input_for_gates = dy::concatenate({hidden, x});
        auto pre_input_gate_coef = dy::logistic(pre_input_gate.forward(input_for_gates));
        auto output_gate_coef = dy::logistic(output_gate.forward(input_for_gates));
        auto gated_concat = dy::concatenate({dy::cmult(hidden, pre_input_gate_coef),x});
        auto output_candidate = dy::tanh(input_fc.forward(gated_concat));
        auto after_forget = dy::cmult(hidden, 1-output_gate_coef);
        auto output_hidden = after_forget + dy::cmult(output_gate_coef, output_candidate);
        return std::make_pair(output_hidden, output_hidden);
      }

      EASY_SERIALZABLE(hidden_dim, pre_input_gate, input_fc, output_gate)

    private:
      void ensure_init(const Expression &x) {
        if (hidden_dim > 0) return;
        hidden_dim = x.dim()[0];
        pre_input_gate = linear_layer(hidden_dim);
        input_fc = linear_layer(hidden_dim);
        output_gate = linear_layer(hidden_dim);
      }

      unsigned hidden_dim;
      linear_layer pre_input_gate;
      linear_layer input_fc;
      linear_layer output_gate;
    };

    typedef rnn<coupled_lstm_cell_t, lstm_cell_state> coupled_lstm;
  }
}
#endif //DYNET_WRAPPER_DY_LSTM_HPP
