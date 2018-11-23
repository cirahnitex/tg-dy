//
// Created by YAN Yuchen on 11/20/2018.
//

#include "dataset_t.hpp"
#include "../../dy.hpp"
#include "../../dy_training_framework.hpp"
#include <unordered_set>
#include <unordered_map>
using namespace tg;
using namespace std;

class poetry_model {
public:
  poetry_model() = default;
  poetry_model(const poetry_model&) = default;
  poetry_model(poetry_model&&) = default;
  poetry_model &operator=(const poetry_model&) = default;
  poetry_model &operator=(poetry_model&&) = default;
  static constexpr char START_OF_SENTENCE[] = "&sos;";
  static constexpr char END_OF_SENTENCE[] = "&eos;";
  static const unsigned MAX_GENERATED_SENTENCE_LENGTH = 128;
  poetry_model(unsigned embedding_size, unordered_set<string> vocab, const unordered_map<string, vector<float>>& w2v):embedding_table(),lstm(1) {
    vocab.insert(START_OF_SENTENCE);
    vocab.insert(END_OF_SENTENCE);
    embedding_table = dy::mono_lookup_readout(embedding_size, vocab, [&](const string& token){
      return w2v.at(token);
    });
  }
  vector<string> predict() {
    vector<string> ret;
    dy::vanilla_lstm::stacked_cell_state cell_state;
    auto x = embedding_table.lookup(START_OF_SENTENCE);
    for(unsigned i=0; i<MAX_GENERATED_SENTENCE_LENGTH; i++) {
      tie(cell_state, x) = lstm.forward(cell_state, x);
      auto token = embedding_table.random_readout(x);
      if(token == END_OF_SENTENCE) {
        break;
      }
      ret.push_back(token);
      x = embedding_table.lookup(token);
    }
    return ret;
  }
  dy::Tensor compute_loss(const vector<string>& sentence) {
    vector<string> input({START_OF_SENTENCE});
    copy(sentence.begin(), sentence.end(), back_inserter(input));
    vector<string> oracle(sentence);
    oracle.push_back(END_OF_SENTENCE);
    auto sentence_emb = embedding_table.lookup(input);
    auto output_emb = lstm.forward(sentence_emb).second;
    return embedding_table.compute_readout_loss(output_emb, oracle);
  }
  EASY_SERIALZABLE(embedding_table, lstm)
private:
  dy::mono_lookup_readout embedding_table;
  dy::vanilla_lstm lstm;
};

dataset_t read_dataset(const string& path) {
  Json::Value json;
  ifstream ifs(path);
  Json::Reader().parse(ifs, json);
  return parse_dataset_json(json);
}

unordered_set<string> collect_frequent_token(const dataset_t& dataset, unsigned top_x) {
  dy::frequent_token_collector collector;
  for(const auto& sentence:dataset) {
    for(const auto& token:sentence) {
      collector.add_occurence(token);
    }
  }
  auto ret = collector.list_frequent_tokens(top_x);
  return unordered_set<string>(ret.begin(), ret.end());
}

template<class T>
void print_helper(const T& x, std::ostream& os=std::cout) {
  for(const auto& t:x) {
    os << t << " ";
  }
  os <<endl;
}
int main() {
  const string DATASET_PATH = "/hltc/0/cl/corpora/poetry/poetry.json";
  const string PATH_TO_WORD2VEC_FILE = "/hltc/0/cl/tools/word_embeddings/w2vgw.d300.en.bin";
  const unsigned VOCAB_SIZE = 5000;
  const unsigned EMBEDDING_SIZE = 50;
  const unsigned NUM_EPOCHES = 30;
  cout << "read dataset" <<endl;
  auto training_set = read_dataset(DATASET_PATH);
  cout << "collect vocab" <<endl;
  const auto vocab = collect_frequent_token(training_set, VOCAB_SIZE);
  cout << "import word2vec" <<endl;
  const auto w2v = dy::import_word2vec(PATH_TO_WORD2VEC_FILE);
  cout << "initialze model" <<endl;
  dy::initialize();
  poetry_model model(EMBEDDING_SIZE, vocab, w2v);
  dy::fit<datum_t>(4, NUM_EPOCHES, training_set, vector<datum_t>(), [&](const datum_t &datum) {
    return model.compute_loss(datum);
  });
  cout << "testing" <<endl;
  for(unsigned i=0; i<10; i++) {
    print_helper(model.predict());
  }
}
