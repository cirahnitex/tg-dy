//
// Created by YAN Yuchen on 11/23/2018.
//


#include "../../dy.hpp"
#include "../../dy_training_framework.hpp"
#include <vector>
#include <string>
#include <json/json.h>

using namespace std;
using namespace tg;

struct zh_en_t {
  vector<string> zh;
  vector<string> en;

  void parse_json(const Json::Value &json) {
    for (const Json::Value &token_json:json["zh"]) {
      zh.push_back(token_json.asString());
    }
    for (const Json::Value &token_json:json["en"]) {
      en.push_back(token_json.asString());
    }
  }
};

typedef vector<zh_en_t> dataset_t;

dataset_t parse_dataset_json(const Json::Value &json) {
  dataset_t ret;
  for (const Json::Value &datum_json:json) {
    zh_en_t datum;
    datum.parse_json(datum_json);
    ret.push_back(datum);
  }
  return ret;
}

dataset_t read_dataset(const string &path) {
  Json::Value json;
  ifstream ifs(path);
  Json::Reader().parse(ifs, json);
  return parse_dataset_json(json);
}

pair<unordered_set<string>, unordered_set<string>> collect_frequent_token(const dataset_t &dataset, unsigned top_x) {
  dy::frequent_token_collector collector0;
  dy::frequent_token_collector collector1;
  for (const auto &sentence:dataset) {
    for (const auto &token:sentence.zh) {
      collector0.add_occurence(token);
    }
    for (const auto &token:sentence.en) {
      collector1.add_occurence(token);
    }
  }
  auto ret0 = collector0.list_frequent_tokens(top_x);
  auto ret1 = collector1.list_frequent_tokens(top_x);
  return make_pair(unordered_set<string>(ret0.begin(), ret0.end()), unordered_set<string>(ret1.begin(), ret1.end()));
}

class attention_model {
public:
  static constexpr char END_OF_SENTENCE[] = "&eos;";
  static unsigned constexpr MAX_OUTPUT_LENGTH = 128;
  static constexpr unsigned ATTENTION_HIDDEN_DIM = 8;
  attention_model() = default;
  attention_model(const attention_model&) = default;
  attention_model(attention_model&&) = default;
  attention_model &operator=(const attention_model&) = default;
  attention_model &operator=(attention_model&&) = default;
  attention_model(unsigned embedding_size, const unordered_set<string>& f_vocab, unordered_set<string> e_vocab, const unordered_map<string, vector<float>>& e_w2v):
    embedding_size(embedding_size), f_embedding_table(embedding_size, f_vocab), e_embedding_table(), encoder(2, embedding_size), decoder(2, embedding_size), attention_fc1(ATTENTION_HIDDEN_DIM), attention_fc2(8)
  {
    e_vocab.insert(END_OF_SENTENCE);
    e_embedding_table = dy::mono_lookup_readout(embedding_size, e_vocab, [&](const string& t){
      return e_w2v.at(t);
    });
  }
  vector<string> predict(const vector<string>& f_sentence) {
    auto f_embeddings = f_embedding_table.lookup(f_sentence);
    auto f_hiddens = encoder.predict_output_sequence(f_embeddings);
    auto cell_state = decoder.default_cell_state();
    auto y = dy::zeros({embedding_size});
    vector<string> ret;
    for(unsigned i=0; i<MAX_OUTPUT_LENGTH; i++) {
      auto context = compute_context(f_hiddens, cell_state);
      tie(cell_state, y) = decoder.predict(cell_state, dy::concatenate({context, y}));
      auto out_token = e_embedding_table.readout(y);
      if(out_token == END_OF_SENTENCE) {break;}
      ret.push_back(out_token);
      y = e_embedding_table.lookup(out_token);
    }
    return ret;
  }
  dy::tensor compute_loss(const vector<string>& f_sentence, vector<string> e_sentence) {
    auto [f_embeddings, f_lookup_loss] = f_embedding_table.lookup_with_loss(f_sentence);
    auto f_hiddens = encoder.predict_output_sequence(f_embeddings);
    e_sentence.push_back(END_OF_SENTENCE);
    auto [e_embeddings, e_lookup_loss] = e_embedding_table.lookup_with_loss(e_sentence);
    auto cell_state = decoder.default_cell_state();
    auto y = dy::zeros({embedding_size});
    vector<dy::tensor> output_embeddings;
    for(unsigned i=0; i<e_sentence.size(); i++) {
      auto input_embedding = i==0?dy::zeros({embedding_size}):e_embeddings[i-1];
      auto context = compute_context(f_hiddens, cell_state);
      tie(cell_state, y) = decoder.predict(cell_state, dy::concatenate({context, input_embedding}));
      output_embeddings.push_back(y);
    }
    return e_embedding_table.compute_readout_loss(output_embeddings, e_sentence) + f_lookup_loss + e_lookup_loss;
  }
  EASY_SERIALIZABLE(embedding_size, f_embedding_table, e_embedding_table, encoder, decoder, attention_fc)
private:
  unsigned embedding_size;
  dy::mono_lookup_readout f_embedding_table;
  dy::mono_lookup_readout e_embedding_table;
  dy::bidirectional_vanilla_lstm encoder;
  dy::vanilla_lstm decoder;
  dy::linear_layer attention_fc1;
  dy::linear_layer attention_fc2;

  dy::tensor compute_context(const vector<dy::tensor>& f_hiddens, const dy::vanilla_lstm::stacked_cell_state& prev_cell_state) {
    auto flattened = dy::vanilla_lstm::flatten_stacked_cell_state(prev_cell_state);
    vector<dy::tensor> xs;
    for(const auto& f_hidden:f_hiddens) {
      auto x = dy::tanh(attention_fc1.predict(dy::concatenate({f_hidden, flattened})));
      xs.push_back(attention_fc2.predict(x));
    }
    return dy::concatenate(f_hiddens,1) * dy::softmax(dy::concatenate(xs));
  }
};

int main() {
  const string DATASET_PATH = "/hltc/0/cl/corpora/manythings.org-anki/zh-en.json";
  const string PATH_TO_WORD2VEC_FILE = "/hltc/0/cl/tools/word_embeddings/w2vgw.d300.en.bin";
  const unsigned EMBEDDING_SIZE = 128;
  const unsigned EPOCHES = 20;
  cout << "read dataset" <<endl;
  const auto dataset = read_dataset(DATASET_PATH);
  cout << "pre-processing" <<endl;
  const auto [training_set, dev_set] = dy::shuffle_and_split_dataset(dataset);
  const auto [f_vocab, e_vocab] = collect_frequent_token(dataset, 20000);
  cout << "import word2vec" <<endl;
  const auto w2v = dy::import_word2vec(PATH_TO_WORD2VEC_FILE);
  cout << "initialize model" <<endl;
  dy::initialize(16);
  attention_model model(EMBEDDING_SIZE, f_vocab, e_vocab, w2v);
  dy::fit<zh_en_t>(EPOCHES, training_set, dev_set, [&](const zh_en_t &datum) {
    return model.compute_loss(datum.zh, datum.en);
  });

  cout << "predicting" <<endl;
  for(unsigned i=0; i<dev_set.size(); i++) {
    if(i>=32) break;
    const auto& datum = dev_set[i];
    const auto en_predict = model.predict(datum.zh);
    cout << "zh: " << ECMAScript_string_utils::join(datum.zh) << endl;
    cout << "en predict: " << ECMAScript_string_utils::join(en_predict) << endl;
  }
  return 0;
}
