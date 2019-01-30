//
// Created by YAN Yuchen on 11/21/2018.
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

template<class T>
void print_helper(const T &x, std::ostream &os = std::cout) {
  for (const auto &t:x) {
    os << t << " ";
  }
  os << endl;
}

class wseq2seq_model {
public:
  wseq2seq_model() = default;
  wseq2seq_model(const wseq2seq_model&) = default;
  wseq2seq_model(wseq2seq_model&&) = default;
  wseq2seq_model &operator=(const wseq2seq_model&) = default;
  wseq2seq_model &operator=(wseq2seq_model&&) = default;
  static constexpr char END_OF_SENTENCE[] = "&eos;";
  static constexpr unsigned MAX_OUTPUT_LENGTH = 128;
  wseq2seq_model(unsigned embedding_size, const unordered_set<string>& f_vocab, unordered_set<string> e_vocab, const unordered_map<string, vector<float>>& e_w2v):
    embedding_size(embedding_size), f_embedding_table(embedding_size, f_vocab), e_embedding_table(), encoder(2, embedding_size), decoder(2, embedding_size), output_fc(embedding_size)
  {
    e_vocab.insert(END_OF_SENTENCE);
    e_embedding_table = dy::mono_lookup_readout(embedding_size, e_vocab, [&](const string& t){
      return e_w2v.at(t);
    });
  }
  vector<string> predict(const vector<string>& f_sentence) {
    const auto sentence_emb = f_embedding_table.lookup(f_sentence);
    auto cell_state = encoder.predict(sentence_emb).first;
    auto x = dy::zeros({embedding_size});
    vector<string> ret;
    for(unsigned i=0; i<MAX_OUTPUT_LENGTH; i++) {
      tie(cell_state, x) = decoder.predict(cell_state, x);
      auto output_token = e_embedding_table.readout(dy::tanh(output_fc.predict(x)));
      if(output_token == END_OF_SENTENCE) {break;}
      ret.push_back(output_token);
      x = e_embedding_table.lookup(output_token);
    }
    return ret;
  }
  dy::tensor compute_loss(const vector<string>& f_sentence, vector<string> e_sentence) {
    const auto [f_sentence_emb, f_lookup_loss] = f_embedding_table.lookup_with_loss(f_sentence);
    auto cell_state = encoder.predict(f_sentence_emb).first;
    e_sentence.push_back(END_OF_SENTENCE);
    const auto [e_sentence_emb, e_lookup_loss] = e_embedding_table.lookup_with_loss(e_sentence);
    vector<dy::tensor> decoder_inputs({dy::zeros({embedding_size})});
    copy(e_sentence_emb.begin(), e_sentence_emb.end()-1,back_inserter(decoder_inputs));
    auto decoder_outputs = decoder.predict(cell_state, decoder_inputs).second;
    for(auto& x:decoder_outputs) {x = dy::tanh(output_fc.predict(x));}
    return e_embedding_table.compute_readout_loss(decoder_outputs, e_sentence) + f_lookup_loss + e_lookup_loss;
  }
  EASY_SERIALIZABLE(embedding_size, f_embedding_table, e_embedding_table, encoder, decoder, output_fc)
private:
  unsigned embedding_size;
  dy::mono_lookup_readout f_embedding_table;
  dy::mono_lookup_readout e_embedding_table;
  dy::vanilla_lstm encoder;
  dy::vanilla_lstm decoder;
  dy::linear_layer output_fc;
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
  wseq2seq_model model(EMBEDDING_SIZE, f_vocab, e_vocab, w2v);
  dy::fit<zh_en_t>(EPOCHES, training_set, dev_set, [&](const zh_en_t &datum) {
    return model.compute_loss(datum.zh, datum.en);
  });

  cout << "predicting" <<endl;
  for(unsigned i=0; i<dev_set.size(); i++) {
    if(i>=32) break;
    const auto& datum = dev_set[i];
    const auto en_predict = model.predict(datum.zh);
    cout << "zh: ";
    print_helper(datum.zh);
    cout << "en predict: ";
    print_helper(en_predict);
  }
  return 0;
}
