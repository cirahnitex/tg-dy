//
// Created by YAN Yuchen on 11/6/2018.
//

#include "../../dy.hpp"
#include <string>
#include <unordered_map>
#include <vector>
#include <dict.hpp>
#include <word2vec.hpp>

using namespace tg;
using namespace std;

using namespace util;

unordered_map<string, vector<float>> import_word2vec(const string &path) {
  unordered_map<string, vector<float> > embeddings;
  read_word2vec()(true,
                  util::id_func(),
                  curry(forced_map_lookup(), embeddings),
                  path);
  return embeddings;
}

int main() {
  dy::initialize();
  const string PATH_TO_WORD2VEC_FILE = "/hltc/0/cl/tools/word_embeddings/w2vgw.d300.en.bin";
  const unsigned EMBEDDING_SIZE = 8;

  // here are all the words we are interested in
  const vector<string> MY_VOCAB({"king", "queen", "man", "woman"});

  // load all word embeddings from w2v
  const auto all_w2v_embeddings = import_word2vec(PATH_TO_WORD2VEC_FILE);

  // construct embedding lookup table from embedding size, vocab and initial embedding getter
  dy::embedding_lookup embedding_lookup(EMBEDDING_SIZE, MY_VOCAB, [&](const string& token){
    return all_w2v_embeddings.at(token);
  });

  // compute embedding
  dy::_renew_cg();
  auto embedding = dy::as_vector(embedding_lookup.lookup("king"));

  // output
  for (const auto &value:embedding) {
    cout << value << " ";
  }
  cout << endl;

}
