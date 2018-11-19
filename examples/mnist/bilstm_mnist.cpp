//
// Created by YAN Yuchen on 11/19/2018.
//

#include <json/json.h>
#include "dataset_t.hpp"
#include <fstream>
#include <iostream>
#include "../../dy.hpp"

using namespace tg;
using namespace std;

dataset_t read_dataset(const string &path) {
  Json::Value json;
  ifstream ifs(path);
  Json::Reader().parse(ifs, json);
  dataset_t ret;
  ret.parse_json(json);
  return ret;
}

class bilstm_maxpool_layer {
public:
  bilstm_maxpool_layer() = default;
  bilstm_maxpool_layer(const bilstm_maxpool_layer &) = default;
  bilstm_maxpool_layer(bilstm_maxpool_layer &&) = default;
  bilstm_maxpool_layer &operator=(const bilstm_maxpool_layer &) = default;
  bilstm_maxpool_layer &operator=(bilstm_maxpool_layer &&) = default;

  bilstm_maxpool_layer(unsigned hidden_dim):bilstm(1, hidden_dim) {}

  dy::Expression forward(const vector<dy::Expression> &features) {
    return dy::max(bilstm.forward_output_sequence(features));
  }

  EASY_SERIALZABLE(bilstm)
private:
  dy::bidirectional_vanilla_lstm bilstm;
};

class bilstm_mnist_model {
public:
  bilstm_mnist_model() = default;
  bilstm_mnist_model(const bilstm_mnist_model&) = default;
  bilstm_mnist_model(bilstm_mnist_model&&) = default;
  bilstm_mnist_model &operator=(const bilstm_mnist_model&) = default;
  bilstm_mnist_model &operator=(bilstm_mnist_model&&) = default;

  bilstm_mnist_model(unsigned width, unsigned height, const unordered_set<string>& labels, unsigned hidden_dim):width(width),height(height),split_by_row_pass(hidden_dim), split_by_column_pass(hidden_dim), ro(labels) {}

  string predict(const vector<float> &image) {
    return ro.readout(forward(image));
  }

  dy::Expression compute_loss(const vector<float>& image, const string& oracle) {
    return ro.compute_loss(forward(image), oracle);
  }

  EASY_SERIALZABLE(width, height, split_by_row_pass, split_by_column_pass, ro)
private:
  unsigned width;
  unsigned height;
  bilstm_maxpool_layer split_by_row_pass;
  bilstm_maxpool_layer split_by_column_pass;
  dy::readout_layer ro;

  vector<dy::Expression> split_by_row(const vector<float> &image) {
    vector<dy::Expression> ret;
    for (unsigned row = 0; row < height; row++) {
      vector<float> row_vals;
      for (unsigned column = 0; column < width; column++) {
        row_vals.push_back(image[row * width + column]);
      }
      ret.push_back(dy::const_expr(row_vals));
    }
    return ret;
  }

  vector<dy::Expression> split_by_column(const vector<float> &image) {
    vector<dy::Expression> ret;
    for (unsigned column = 0; column < width; column++) {
      vector<float> column_vals;
      for (unsigned row = 0; row < height; row++) {
        column_vals.push_back(image[row * width + column]);
      }
      ret.push_back(dy::const_expr(column_vals));
    }
    return ret;
  }

  dy::Expression forward(const vector<float>& image) {
    return dy::concatenate({
                             split_by_row_pass.forward(split_by_row(image)),
                             split_by_column_pass.forward(split_by_column(image))
                           });
  }
};

pair<vector<datum_t>, vector<datum_t>> shuffle_and_split(const dataset_t& dataset) {
  vector<datum_t> data = dataset.data;
  random_shuffle(data.begin(), data.end());
  unsigned cut = data.size()/10;
  return make_pair(vector<datum_t>(data.begin()+cut, data.end()), vector<datum_t>(data.begin(), data.begin()+cut));
}

int main() {
  const string DATASET_PATH = "/hltc/0/cl/corpora/mnist/train.json";
  const unsigned HIDDEN_DIM = 15;

  cout << "reading dataset" <<endl;
  auto dataset = read_dataset(DATASET_PATH);

  cout << "splitting" <<endl;
  auto [training_set, dev_set] = shuffle_and_split(dataset);
  bilstm_mnist_model model(dataset.width, dataset.height, dataset.labels, HIDDEN_DIM);

  cout << "training" <<endl;
  for(unsigned i=0; i<10; i++) {
    cout << "epoch: "<<i <<endl;
    dy::mp_train<datum_t>(1, training_set, [&](const datum_t& datum){
      return model.compute_loss(datum.input, datum.oracle);
    });
  }

  cout << "testing. dev set size: "<<dev_set.size() <<endl;
  unsigned num_correct = 0;
  for(const auto& datum:dev_set) {
    if(datum.oracle == model.predict(datum.input)) num_correct++;
  }
  cout << "correct predictions:"<< num_correct <<endl;
  return 0;
}
