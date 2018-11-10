//
// Created by YAN Yuchen on 5/1/2018.
//

#ifndef DYNET_WRAPPER_DY_LSTM_HPP
#define DYNET_WRAPPER_DY_LSTM_HPP

#include <dynet/lstm.h>
#include "dy_common.hpp"
#include <memory>

namespace tg {
  namespace dy {
    /**
     * pure functional RNN adapter that wraps dynet::RNNBuilder
     * \tparam RNN_BUILDER_TYPE inherits dynet::RNNBuilder
     */
    template<class RNN_BUILDER_TYPE>
    class rnn {
    public:
      rnn() = default;

      rnn(const rnn &) = default;

      rnn(rnn &&) = default;

      rnn &operator=(const rnn &) = default;

      rnn &operator=(rnn &&) = default;

      /**
       * cell state is the thing that gets passed between time steps.
       * it's an concatenation of all layer's hidden states followed by all layer's cell states
       * normally you shouldn't be using it in any other way.
       */
      typedef std::vector<dynet::Expression> cell_state_type;


      /**
       * apply the RNN cell for a single time step
       * \param prev_state the previous cell state
       * \param x the current input
       * \return 0) the output
       *         1) the cell state after this time step
       */
      std::pair<dynet::Expression, cell_state_type>
      operator()(const cell_state_type &prev_state, const dynet::Expression &x) {
        ensure_init(x);
        AUTO_START_THIS_GRAPH(builder->new_graph(cg()));
        builder->start_new_sequence(prev_state);
        auto y = builder->add_input(x);
        return std::make_pair(y, builder->final_s());
      }

      /**
       * apply the RNN cell for multiple time steps
       * \param prev_state the previous cell state
       * \param x_sequence a list of inputs to apply, in chronological order
       * \return 0) the list of output in chronological order
       *         1) the cell state after the last time step
       */
      std::pair<std::vector<dynet::Expression>, cell_state_type>
      operator()(const cell_state_type &prev_state, const std::vector<dynet::Expression> &x_sequence) {
        if(x_sequence.empty()) return std::make_pair(std::vector<dynet::Expression>(), default_initial_state());
        ensure_init(x_sequence[0]);
        AUTO_START_THIS_GRAPH(builder->new_graph(cg()));
        builder->start_new_sequence(prev_state);
        std::vector<dynet::Expression> y_sequence;
        for (auto itr = x_sequence.begin(); itr != x_sequence.end(); ++itr) {
          y_sequence.push_back(builder->add_input(*itr));
        }
        return std::make_pair(y_sequence, builder->final_s());
      }

      static cell_state_type default_initial_state() { return {}; }

    protected:
      std::shared_ptr<RNN_BUILDER_TYPE> builder;
      virtual void ensure_init(const dynet::Expression& x) = 0;
    };

    class coupled_lstm : public rnn<dynet::CoupledLSTMBuilder> {
      coupled_lstm() = default;
      coupled_lstm(const coupled_lstm &) = default;
      coupled_lstm(coupled_lstm &&) = default;
      coupled_lstm &operator=(const coupled_lstm &) = default;
      coupled_lstm &operator=(coupled_lstm &&) = default;

      /**
       * \brief Constructor for the LSTMBuilder
       *
       * \param layers Number of layers
       * \param hidden_dim Dimention of the hidden states \f$h_t\f$ and \f$c_t\f$
       */
      coupled_lstm(unsigned layers, unsigned hidden_dim) : rnn(), layers(layers), input_dim(0), hidden_dim(hidden_dim) {
      }

      template<class Archive>
      void save(Archive &ar) {
        ar(layers, input_dim, hidden_dim);
        if(input_dim == 0) return;
        ar(builder->params);
      }

      template<class Archive>
      void load(Archive &ar) {
        ar(layers, input_dim, hidden_dim);
        if(input_dim == 0) return;
        builder = std::make_shared<dynet::CoupledLSTMBuilder>(layers, input_dim, hidden_dim, pc());
        ar(builder->params);
      }

    protected:
      void ensure_init(const dynet::Expression& x) {
        if(input_dim != 0) return;
        input_dim = x.dim()[0];
        builder = std::make_shared<dynet::CoupledLSTMBuilder>(layers, input_dim, hidden_dim, pc());
      }

    private:
      unsigned layers;
      unsigned input_dim;
      unsigned hidden_dim;
    };

    class vanilla_lstm : public rnn<dynet::VanillaLSTMBuilder> {
      vanilla_lstm() = default;
      vanilla_lstm(const vanilla_lstm&) = default;
      vanilla_lstm(vanilla_lstm&&) = default;
      vanilla_lstm &operator=(const vanilla_lstm&) = default;
      vanilla_lstm &operator=(vanilla_lstm&&) = default;
      vanilla_lstm(unsigned layers,
                  unsigned hidden_dim,
                  bool ln_lstm = false,
                  float forget_bias = 1.f):rnn(), layers(layers), input_dim(-1), hidden_dim(hidden_dim), ln_lstm(ln_lstm), forget_bias(forget_bias){}
      template<class Archive>
      void save(Archive &ar) {
        ar(layers, input_dim, hidden_dim, ln_lstm, forget_bias);
        if(input_dim == 0) return;
        ar(builder->params);
        ar(builder->ln_params);
      }
      template<class Archive>
      void load(Archive &ar) {
        ar(layers, input_dim, hidden_dim, ln_lstm, forget_bias);
        if(input_dim == 0) return;
        builder = std::make_shared<dynet::VanillaLSTMBuilder>(layers, input_dim, hidden_dim, pc(), ln_lstm, forget_bias);
        ar(builder->params);
        ar(builder->ln_params);
      }

    protected:
      void ensure_init(const dynet::Expression& x) {
        if(input_dim != 0) return;
        input_dim = x.dim()[0];
        builder = std::make_shared<dynet::VanillaLSTMBuilder>(layers, input_dim, hidden_dim, pc(), ln_lstm, forget_bias);
      }

    private:
      unsigned layers;
      unsigned input_dim;
      unsigned hidden_dim;
      bool ln_lstm;
      float forget_bias;
    };


  }
}
#endif //DYNET_WRAPPER_DY_LSTM_HPP
