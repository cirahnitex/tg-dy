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

      virtual std::pair<cell_state_type, dynet::Expression>
      forward(const cell_state_type &prev_state, const dynet::Expression &x) = 0;
    };


    template<class RNN_CELL_T, class CELL_STATE_T>
    class rnn {
    public:
      rnn() = default;

      rnn(const rnn &) = default;

      rnn(rnn &&) = default;

      rnn &operator=(const rnn &) = default;

      rnn &operator=(rnn &&) = default;

      /**
       * apply the RNN cell for a single time step
       * \param prev_state the previous cell state
       * \param x the current input
       * \return 0) the cell state after this time step
       *         1) the output
       */
      std::pair<CELL_STATE_T, dynet::Expression>
      forward(const CELL_STATE_T &prev_state, const dynet::Expression &x) {
        return cell.forward(prev_state, x);
      }

      /**
       * apply the RNN cell for multiple time steps
       * \param prev_state the previous cell state
       * \param x_sequence a list of inputs to apply, in chronological order
       * \return 0) the cell state after the last time step
       *         1) the list of output in chronological order
       */
      std::pair<CELL_STATE_T, std::vector<dynet::Expression>>
      forward(const CELL_STATE_T &prev_state, const std::vector<dynet::Expression> &x_sequence) {
        if (x_sequence.empty()) return std::make_pair(CELL_STATE_T(), std::vector<dynet::Expression>());
        auto[_state, y] = forward(prev_state, x_sequence[0]);
        std::vector<dynet::Expression> ys;
        ys.push_back(std::move(y));
        for (unsigned i = 1; i < x_sequence.size(); i++) {
          std::tie(_state, y) = forward(_state, x_sequence[i]);
          ys.push_back(std::move(y));
        }
        return std::make_pair(std::move(_state), std::move(ys));
      }

    protected:
      RNN_CELL_T cell;
    };

    struct lstm_cell_state {
      Expression cell_state, hidden_state;
    };

    class vanilla_lstm_cell_t : public rnn_cell_t<lstm_cell_state> {
    public:
      vanilla_lstm_cell_t():hidden_dim(0), forget_gate(), input_gate(), input_fc(), output_gate() {};
      vanilla_lstm_cell_t(const vanilla_lstm_cell_t&) = default;
      vanilla_lstm_cell_t(vanilla_lstm_cell_t&&) = default;
      vanilla_lstm_cell_t &operator=(const vanilla_lstm_cell_t&) = default;
      vanilla_lstm_cell_t &operator=(vanilla_lstm_cell_t&&) = default;
      virtual std::pair<lstm_cell_state, dynet::Expression> forward(const lstm_cell_state &prev_state, const dynet::Expression &x) {
        ensure_init(x);
        auto cell_state = prev_state.cell_state;
        if(cell_state.pg==nullptr) cell_state = zeros(x.dim());
        auto hidden_state = prev_state.hidden_state;
        if(hidden_state.pg==nullptr) hidden_state = zeros(x.dim());

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
      void ensure_init(const Expression& x) {
        if(hidden_dim>0) return;
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

    typedef rnn<vanilla_lstm_cell_t, lstm_cell_state> valinna_lstm;


  }
}
#endif //DYNET_WRAPPER_DY_LSTM_HPP
