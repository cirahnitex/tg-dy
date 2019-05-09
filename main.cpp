#include <iostream>
#include "dyana.hpp"
#include <xml_archive.hpp>

using namespace std;

struct labeled_image {
  std::vector<float> pixels;
  std::string oracle;

};

struct dataset_t {
  unsigned width;
  unsigned height;
  std::vector<std::string> labels;
  std::vector<labeled_image> data;

  template<typename Archive>
  void serialize(Archive &ar) {
    ar(cereal::make_nvp("width", width));
    ar(cereal::make_nvp("height", height));
    ar.nest_list("labels", labels, [&](auto &&label) {
      ar(cereal::make_nvp("label", label));
    });
    ar.nest_list("images", data, [&](auto &&datum) {
      ar.attribute("digit", datum.oracle);
      ar.nest_list("image", datum.pixels, [&](float &pixel) {
        ar(cereal::make_nvp("px", pixel));
      });
    });
  }
};

class mnist_model {
  unsigned image_width_m;
  unsigned image_height_m;
  dyana::conv2d_layer conv0_m;
  dyana::conv2d_layer conv1_m;
  dyana::readout_model final_readout_m;
public:
  EASY_SERIALIZABLE(conv0_m, conv1_m, final_readout_m)

  template<typename RANGE_EXP>
  mnist_model(unsigned image_width, unsigned image_height, RANGE_EXP &&labels)
    : image_width_m(image_width), image_height_m(image_height),  conv0_m(16, 5, 5), conv1_m(32, 3, 3), final_readout_m(labels) {}

  mnist_model(const mnist_model &) = default;

  mnist_model(mnist_model &&) noexcept = default;

  mnist_model &operator=(const mnist_model &) = default;

  mnist_model &operator=(mnist_model &&) noexcept = default;

  dyana::tensor operator()(dyana::tensor img2d) {
    auto x = img2d;
    x = dyana::tanh(conv0_m(x));
    x = dyana::maxpooling2d(x, 3, 3, 3, 3);
    x = dyana::tanh(conv1_m(x));
    x = dyana::max_dim(dyana::max_dim(x));
    return x;
  }

  string operator()(const std::vector<float> &pixels) {
    return final_readout_m(this->operator()(dyana::tensor(pixels, {image_height_m, image_width_m})));
  }

  dyana::tensor compute_loss(const labeled_image &img) {
    if(img.pixels.size() != image_width_m*image_height_m) return (dyana::tensor)0;
    return final_readout_m.compute_loss(
      this->operator()(dyana::tensor(img.pixels, {image_height_m, image_width_m})),
      img.oracle
    );
  }
};

int main() {
  dyana::initialize();

  dataset_t dataset;
  {
    cout << "loading mnist.xml" << endl;
    ifstream ifs("mnist.xml");
    hltc_xml_input_archive ia(ifs);
    ia >> cereal::make_nvp("dataset", dataset);
    cout << "# of images loaded " << dataset.data.size() << endl;
  }

  dataset.data.resize(dataset.data.size()/10);
  cout << "dataset resized to "<<dataset.data.size() <<endl;

  mnist_model model(dataset.width, dataset.height, dataset.labels);
  unsigned cut = dataset.data.size()*9/10;
  vector<labeled_image> dev_set;
  std::move(dataset.data.begin() + cut, dataset.data.end(), back_inserter(dev_set));

  dyana::adam_trainer trainer;
  trainer.num_workers = 4;
  trainer.num_epochs = 4;
  trainer.train_reporting_dev_score(model, dataset.data, dev_set);

  cout << "dev testing" <<endl;

  unsigned total_cnt{};
  unsigned correct_cnt{};
  for(auto &&dev_img:dev_set) {
    if(dev_img.pixels.empty()) continue;
    auto prediction = model(dev_img.pixels);
    total_cnt++;
    if(prediction == dev_img.oracle) correct_cnt++;
  }

  cout << "accuracy = "<< (correct_cnt/(double)total_cnt) << endl;

  return 0;
}
