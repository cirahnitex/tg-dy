#include <iostream>
#include "dy.hpp"
#include <chrono>
using namespace tg;
using namespace std;

using datum_t = pair<vector<float>, string>;

class model {
  dy::normalization_layer norm;
  dy::readout_model ro;
  dy::tensor shared_op(const vector<float> &x) {
    return norm.predict(x);
  }
public:
  model():norm(),ro({"small","medium","big"}){};
  model(const model&) = default;
  model(model&&) = default;
  model &operator=(const model&) = default;
  model &operator=(model&&) = default;
  string predict(const vector<float>& x) {
    return ro.predict(shared_op(x));
  }
  dy::tensor compute_loss(const vector<float>& x, const string& oracle) {
    return ro.compute_loss(shared_op(x), oracle);
  }
};

int main() {
  dy::initialize(1, dy::trainer_type::SIMPLE_SGD);


  vector<datum_t> dataset({
                            make_pair<vector<float>, string>({11,16,19,19,17,16,10,19,19,14},"big"),
                            make_pair<vector<float>, string>({12,11,19,11,14,18,19,12,18,11},"medium"),
                            make_pair<vector<float>, string>({19,12,10,18,17,17,10,15,16,19},"big"),
                            make_pair<vector<float>, string>({17,19,14,11,16,17,17,18,12,18},"big"),
                            make_pair<vector<float>, string>({12,18,19,17,11,18,10,10,17,19},"big"),
                            make_pair<vector<float>, string>({19,12,14,16,19,13,12,12,15,16},"medium"),
                            make_pair<vector<float>, string>({15,17,16,12,15,10,18,15,11,17},"medium"),
                            make_pair<vector<float>, string>({14,15,10,10,14,15,14,13,11,10},"small"),
                            make_pair<vector<float>, string>({10,18,17,10,15,15,10,18,11,10},"small"),
                            make_pair<vector<float>, string>({16,16,16,11,15,18,15,19,19,14},"big"),
                            make_pair<vector<float>, string>({11,19,14,11,18,15,18,16,13,13},"medium"),
                            make_pair<vector<float>, string>({18,11,18,12,13,17,10,16,10,10},"small"),
                            make_pair<vector<float>, string>({11,13,16,14,17,17,13,19,14,16},"medium"),
                            make_pair<vector<float>, string>({19,18,11,11,14,11,16,11,12,15},"small"),
                            make_pair<vector<float>, string>({16,11,16,12,11,12,18,16,14,14},"small"),
                            make_pair<vector<float>, string>({15,13,13,12,13,19,10,15,18,14},"medium"),
                            make_pair<vector<float>, string>({18,19,14,17,19,10,19,13,17,13},"big"),
                            make_pair<vector<float>, string>({12,17,12,11,18,18,12,17,13,19},"medium"),
                            make_pair<vector<float>, string>({19,10,16,15,15,13,19,13,19,18},"big"),
                            make_pair<vector<float>, string>({17,17,18,19,11,19,12,18,19,18},"big")
  });

  cout << "training first model" <<endl;
  model m;
  dy::fit<datum_t>(20, dataset, [&](const datum_t& datum){
    return m.compute_loss(datum.first, datum.second);
  });
  for(const datum_t& datum:dataset) {
    cout << "oracle: "<< datum.second <<"\t";
    cout << "predicted: "<< m.predict(datum.first) <<endl;
  }
  {
    cout << "training victim model" <<endl;
    model m;
    dy::fit<datum_t>(20, dataset, [&](const datum_t& datum){
      return m.compute_loss(datum.first, datum.second);
    });
  }
  {
    cout << "training second victimmodel" <<endl;
    model m;
    dy::fit<datum_t>(10, dataset, [&](const datum_t& datum){
      return m.compute_loss(datum.first, datum.second);
    });
  }
  for(const datum_t& datum:dataset) {
    cout << "oracle: "<< datum.second <<"\t";
    cout << "predicted: "<< m.predict(datum.first) <<endl;
  }


  return 0;
}
