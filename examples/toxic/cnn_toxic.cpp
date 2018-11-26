//
// Created by YAN Yuchen on 11/12/2018.
//

#include "data_t.hpp"
#include "../../dy.hpp"
#include "../../dy_training_framework.hpp"
#include <fstream>

using namespace std;
using namespace tg;

vector<string> collect_frequent_tokens(const dataset_t& data, unsigned top_x=20000) {
  dy::frequent_token_collector collector;
  for(const auto& datum:data.data) {
    for(const auto& token:datum.input) {
      collector.add_occurence(token);
    }
  }
  return collector.list_frequent_tokens(top_x);
}

class my_model {
public:
  my_model(const unordered_set<string>& labels, const unordered_set<string>& vocab, const unordered_map<string, vector<float>>& init_embeddings, unsigned embedding_size)
  :emb(embedding_size, vocab, [&](const string& token){return init_embeddings.at(token);}), conv0(embedding_size,3,1), conv1(embedding_size,3,1), conv2(embedding_size,3,1), fc(128), ro(labels){
  }

  dy::tensor forward(const vector<string>& sentence) {
    vector<dy::tensor> xs;
    xs = emb.lookup(sentence, true);

    xs = conv0.predict(xs);
    for(auto& x:xs) {x=dy::rectify(x);}
    xs = dy::maxpooling1d(conv1.predict(xs), 3, 1);

    for(auto& x:xs) {x=dy::rectify(x);}
    xs = dy::maxpooling1d(conv2.predict(xs), 3, 1);
    for(auto& x:xs) {x=dy::rectify(x);}

    auto x = dy::max(xs);
    x = dy::tanh(fc.predict(x));
    return x;
  }

  unordered_set<string> predict(const vector<string>& sentence) {
    return ro.readout(forward(sentence));
  }

  dy::tensor compute_loss(const vector<string>& sentence, const unordered_set<string>& labels) {
    auto x = ro.compute_loss(forward(sentence), labels);
    return x;
  }

  EASY_SERIALZABLE(emb, conv0, conv1, conv2, fc, ro)
private:
  dy::embedding_lookup emb;
  dy::conv1d_layer conv0;
  dy::conv1d_layer conv1;
  dy::conv1d_layer conv2;
  dy::linear_layer fc;
  dy::multi_readout_model ro;
};

template<class T>
void print_helper(const T& x, std::ostream& os=std::cout) {
  for(const auto& t:x) {
    os << t << " ";
  }
  os <<endl;
}


int main() {
  const string DATASET_PATH = "/hltc/0/cl/corpora/jigsaw-toxic-comment-classification-challenge/processed/train.json";
  const string PATH_TO_WORD2VEC_FILE = "/hltc/0/cl/tools/word_embeddings/w2vgw.d300.en.bin";
  cout << "read dataset" <<endl;
  const auto dataset = read_dataset(DATASET_PATH);
  const auto [training_set, dev_set] = dy::shuffle_and_split_dataset(dataset.data);
  cout << "collect frequent tokens" <<endl;
  const auto vocab = collect_frequent_tokens(dataset);
  cout << "import word2vec" <<endl;
  const auto w2v = dy::import_word2vec(PATH_TO_WORD2VEC_FILE);
  cout << "initialze model" <<endl;
  dy::initialize();
  my_model model(dataset.labels, unordered_set<string>(vocab.begin(), vocab.end()), w2v, 128);

  cout << "training" <<endl;
  dy::fit<datum_t>(4, 10, training_set, dev_set, [&](const datum_t &datum) {
    return model.compute_loss(datum.input, datum.oracle);
  }, [](const std::exception &e, const datum_t &datum) {
    cerr << e.what() << endl;
    cerr << "sentence is:";
    print_helper(datum.input, cerr);
  });

  cout << "predicting" <<endl;
  for(unsigned i=0; i<64; i++) {
    const auto& datum = dev_set[i];
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
