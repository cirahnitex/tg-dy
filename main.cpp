#include <iostream>
#include "dy.hpp"

using namespace tg;
using namespace std;

class height_prediction_model {
  dy::linear_layer fc_m;
public:
  height_prediction_model(const height_prediction_model&) = default;
  height_prediction_model(height_prediction_model&&) = default;
  height_prediction_model &operator=(const height_prediction_model&) = default;
  height_prediction_model &operator=(height_prediction_model&&) = default;

  height_prediction_model():fc_m(1) {}

  /**
   * given the weight, predict his/her height
   * \param weight
   * \return predicted height
   */
  dy::tensor predict(const dy::tensor& weight) {
    return fc_m.predict(weight).as_scalar();
  }

  /**
   * given weight and height, compute loss
   * \param weight
   * \param oracle_height
   * \return the loss
   */
  dy::tensor compute_loss(const dy::tensor& weight, const dy::tensor& oracle_height) {
    return dy::squared_distance(fc_m.predict(weight), oracle_height);
  }
};

int main() {
  dy::initialize();

  // define training set
  typedef pair<float, float> datum;
  vector<datum> training_set({
    make_pair(44.0, 165.0),
    make_pair(55.0, 172.0),
    make_pair(60.0, 170.0),
    make_pair(73.0, 181.0),
    make_pair(88.0, 175.0),
    make_pair(45.0, 155.0),
    make_pair(55.0, 172.0),
    make_pair(62.0, 175.0),
    make_pair(71.0, 182.0),
    make_pair(97.0, 172.0)
  });

  // initialize model
  height_prediction_model model;

  dy::_trainer().learning_rate = 0.01;

  // training
  dy::fit<datum>(4, 100, training_set, vector<datum>(), [&](const datum& datum){
    return model.compute_loss(datum.first, datum.second);
  });

  // testing
  for(float val:{86.0, 40.0, 72.0}) {
    cout << "weight:"<< val <<" predicted height:"<< model.predict(val).as_scalar() << endl;
  }

  return 0;
}
