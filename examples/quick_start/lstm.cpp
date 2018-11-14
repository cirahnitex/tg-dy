//
// Created by YAN Yuchen on 11/13/2018.
//

#include "../../dy.hpp"
#include <vector>
using namespace tg;
using namespace std;

class simple_seq_to_seq_translation_model {
public:
  static constexpr unsigned MAX_OUTPUT_SIZE = 128;
  static constexpr char END_OF_SENTENCE[] = "&eos;";
  simple_seq_to_seq_translation_model(unsigned embedding_size, const unordered_set<string>& foreign_tokens, const unordered_set<string>& emit_tokens):embedding_size(embedding_size), emb_table(embedding_size, foreign_tokens), ro(), encoder(3),decoder(3) {
    unordered_set<string> enhanced_l1_tokens = emit_tokens;
    enhanced_l1_tokens.insert(END_OF_SENTENCE);
    ro = dy::mono_lookup_readout(embedding_size, enhanced_l1_tokens);
  };

  vector<string> predict(const vector<string>& foreign_sentence) {
    auto sentence_embs = emb_table.read_sentence(foreign_sentence);
    auto cell_state = encoder.forward(sentence_embs).first;
    dy::Expression curr_output_emb = dy::zeros({embedding_size});
    vector<string> ret;
    for(unsigned i=0; i<MAX_OUTPUT_SIZE; i++) {
      tie(cell_state, curr_output_emb) = decoder.forward(cell_state, curr_output_emb);
      auto output_token = ro.readout(curr_output_emb);
      if(output_token == END_OF_SENTENCE) {
        break;
      }
      ret.push_back(output_token);
    }
    return ret;
  }

  dy::Expression compute_loss(const vector<string>& foreign_sentence, const vector<string>& emit_sentence) {
    auto [sentence_embs, foreign_lookup_loss] = emb_table.read_sentence_with_loss(foreign_sentence);
    auto cell_state = encoder.forward(sentence_embs).first;
    auto [emit_embs, emit_lookup_loss] = ro.read_sentence_with_loss(emit_sentence);
    auto emit_with_ending_eos = emit_sentence;
    emit_with_ending_eos.push_back(END_OF_SENTENCE);

    vector<dy::Expression> output_embs(emit_with_ending_eos.size());
    for(unsigned i=0; i<emit_with_ending_eos.size(); i++) {
      string oracle_token = emit_with_ending_eos[i];
      auto input_emb = i==0?dy::zeros({embedding_size}):emit_embs[i-1];
      tie(cell_state, output_embs[i]) = decoder.forward(cell_state, input_emb);
    }

    return ro.compute_windowed_loss(output_embs, emit_with_ending_eos) + foreign_lookup_loss + emit_lookup_loss;
  }

  EASY_SERIALZABLE(embedding_size, emb_table, ro, encoder, decoder)
private:
  unsigned embedding_size;
  dy::mono_lookup_readout emb_table;
  dy::mono_lookup_readout ro;
  dy::vanilla_lstm encoder;
  dy::vanilla_lstm decoder;
};

struct datum_t {
  datum_t() = default;
  datum_t(const datum_t&) = default;
  datum_t(datum_t&&) = default;
  datum_t &operator=(const datum_t&) = default;
  datum_t &operator=(datum_t&&) = default;
  datum_t(const string &foreign, const string &emit) : foreign(foreign), emit(emit) {}
  string foreign;
  string emit;
};

pair<unordered_set<string>, unordered_set<string>> collect_vocab(const vector<datum_t>& data) {
  unordered_set<string> foreign_vocab;
  unordered_set<string> emit_vocab;
  for(const auto& datum:data) {
    for(const auto& foreign_token:ECMAScript_string_utils::split(datum.foreign)) {
      foreign_vocab.insert(foreign_token);
    }
    for(const auto& emit_token:ECMAScript_string_utils::split(datum.emit)) {
      emit_vocab.insert(emit_token);
    }
  }
  return make_pair(move(foreign_vocab),move(emit_vocab));
}

template<class T>
void print_helper(const T& x, std::ostream& os=std::cout) {
  for(const auto& t:x) {
    os << t << " ";
  }
  os <<endl;
}

int main() {
  vector<datum_t> training_set({
    datum_t("左 转","turn left"),
    datum_t("右 转","turn right"),
    datum_t("左 转 再 右 转","turn left and turn right"),
    datum_t("右 转 再 左 转","turn right and turn left"),
    datum_t("左 转 再 左 转","turn left and turn left"),
    datum_t("右 转 再 右 转","turn right and turn right")
  });
  const unsigned EMBEDDING_SIZE = 16;
  auto [foreign_vocab, emit_vocab] = collect_vocab(training_set);
  dy::initialize();
  simple_seq_to_seq_translation_model model(EMBEDDING_SIZE, foreign_vocab, emit_vocab);
  for(unsigned epoch = 0; epoch < 1000; epoch ++) {
    for(const auto& datum:training_set) {
      dy::renew_cg();
      auto loss = model.compute_loss(ECMAScript_string_utils::split(datum.foreign), ECMAScript_string_utils::split(datum.emit));
      dy::train_on_loss(loss);
    }
  }
  cout << "predicting" << endl;
  for(const auto& datum:training_set) {
    dy::renew_cg();
    print_helper(model.predict(ECMAScript_string_utils::split(datum.foreign)));
  }
}
