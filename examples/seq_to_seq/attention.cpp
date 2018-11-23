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
  static unsigned constexpr MAX_OUTPUT_LENGTH = 128;
  vector<string> forward(const vector<string>& f_sentence) {
    auto f_embeddings = f_embedding_table.lookup(f_sentence);
    auto f_hiddens = encoder.forward_output_sequence(f_embeddings);
    dy::vanilla_lstm::stacked_cell_state cell_state;
    for(unsigned i=0; i<MAX_OUTPUT_LENGTH; i++) {
      dy::Tensor context;
      decoder.forward(cell_state, )
    }
  }
private:
  unsigned embedding_size;
  dy::mono_lookup_readout f_embedding_table;
  dy::mono_lookup_readout e_embedding_table;
  dy::bidirectional_vanilla_lstm encoder;
  dy::vanilla_lstm decoder;
  dy::linear_layer attention_fc;
};
