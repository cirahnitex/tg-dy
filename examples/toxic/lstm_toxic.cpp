//
// Created by YAN Yuchen on 11/16/2018.
//


#include "data_t.hpp"
#include "../../dy.hpp"
#include "../../dy_training_framework.hpp"
#include <fstream>

using namespace std;
using namespace tg;

unordered_set<string> collect_frequent_tokens(const dataset_t& data, unsigned top_x=20000) {
  dy::frequent_token_collector collector;
  for(const auto& datum:data.data) {
    for(const auto& token:datum.input) {
      collector.add_occurence(token);
    }
  }
  auto ret = collector.list_frequent_tokens(top_x);
  return unordered_set<string>(ret.begin(), ret.end());
}

class lstm_toxic_model {
public:
  lstm_toxic_model(const unordered_set<string>& labels, const unordered_set<string>& vocab, const unordered_map<string, vector<float>>& init_embeddings, unsigned embedding_size)
    :emb(embedding_size, vocab, [&](const string& token){return init_embeddings.at(token);}), lstm(1, 15), ro(labels){
  }

  dy::Tensor forward(const vector<string>& sentence) {
//    auto output_embs = lstm.forward(emb.lookup(sentence, true)).second;
    auto output_embs = lstm.forward_output_sequence(emb.lookup(sentence, true));
    return dy::max(output_embs);
  }

  unordered_set<string> predict(const vector<string>& sentence) {
    return ro.readout(forward(sentence));
  }

  dynet::Expression compute_loss(const vector<string>& sentence, const unordered_set<string>& labels) {
    return ro.compute_loss(forward(sentence), labels);
  }

  EASY_SERIALZABLE(emb, lstm, ro)
private:
  dy::embedding_lookup emb;
//  dy::vanilla_lstm lstm;
  dy::bidirectional_vanilla_lstm lstm;
  dy::multi_readout_layer ro;
};

template<class T>
void print_helper(const T& x, std::ostream& os=std::cout) {
  for(const auto& t:x) {
    os << t << " ";
  }
  os <<endl;
}


int main() {
  const string TRAINING_DATA_PATH = "/hltc/0/cl/corpora/jigsaw-toxic-comment-classification-challenge/processed/train.json";
  const string TEST_DATA_PATH = "/hltc/0/cl/corpora/jigsaw-toxic-comment-classification-challenge/processed/test.json";
  const string PATH_TO_WORD2VEC_FILE = "/hltc/0/cl/tools/word_embeddings/w2vgw.d300.en.bin";
  cout << "read dataset" <<endl;
  const auto trainint_set = read_dataset(TRAINING_DATA_PATH);
  const auto test_set = read_dataset(TEST_DATA_PATH);
  cout << "collect frequent tokens" <<endl;
  const auto vocab = collect_frequent_tokens(trainint_set);
  cout << "import word2vec" <<endl;
  const auto w2v = dy::import_word2vec(PATH_TO_WORD2VEC_FILE);
  cout << "initialze model" <<endl;
  dy::initialize();
  lstm_toxic_model model(trainint_set.labels, vocab, w2v, 128);

  cout << "training" <<endl;
  dy::fit<datum_t>(8, 10, trainint_set.data, test_set.data, [&](const datum_t &datum) {
    return model.compute_loss(datum.input, datum.oracle);
  }, [](const std::exception &e, const datum_t &datum) {
    cerr << e.what() << endl;
    cerr << "sentence is:";
    print_helper(datum.input, cerr);
  });

  cout << "testing" <<endl;

  for(unsigned i=0; i<64; i++) {
    const auto& datum = test_set.data[i];
    cout << "sentence:";
    print_helper(datum.input);
    cout << "oracle:";
    print_helper(datum.oracle);
    const auto result = model.predict(datum.input);
    cout << "predict:";
    print_helper(result);
    cout << endl;
  }

}
