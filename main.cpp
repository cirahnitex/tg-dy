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
    ar.nest_list("digits", labels, [&](string &label) {
      ar(cereal::make_nvp("digit", label));
    });
    ar.nest_list("images", data, [&](labeled_image &datum) {
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
  EASY_SERIALIZABLE(image_width_m, image_height_m, conv0_m, conv1_m, final_readout_m)

  template<typename RANGE_EXP>
  mnist_model(unsigned image_width, unsigned image_height, RANGE_EXP &&labels)
    : image_width_m(image_width), image_height_m(image_height), conv0_m(16, 5, 5), conv1_m(32, 3, 3),
      final_readout_m(labels) {}

  mnist_model(const mnist_model &) = default;

  mnist_model(mnist_model &&) noexcept = default;

  mnist_model &operator=(const mnist_model &) = default;

  mnist_model &operator=(mnist_model &&) noexcept = default;

  dyana::tensor get_image_embedding(const vector<float> &pixels) {
    if (pixels.size() != image_width_m * image_height_m) throw std::runtime_error("unexpected image resolution");
    dyana::tensor x(pixels, {image_width_m, image_height_m});
    x = dyana::tanh(conv0_m(x));
    x = dyana::maxpooling2d(x, 3, 3, 3, 3);
    x = dyana::tanh(conv1_m(x));
    return dyana::max_dim(dyana::max_dim(x));
  }

  string operator()(const vector<float> &pixels) {
    return final_readout_m(get_image_embedding(pixels));
  }

  dyana::tensor compute_loss(const labeled_image &img) {
    return final_readout_m.compute_loss(get_image_embedding(img.pixels), img.oracle);
  }
};

int main() {
  dyana::initialize();

  dataset_t trainingset;
  {
    cout << "loading mnist.xml" << endl;
    ifstream ifs("mnist.xml");
    hltc_xml_input_archive ia(ifs);
    ia >> trainingset;
    cout << "# of images loaded " << trainingset.data.size() << endl;
  }

  unsigned cut = trainingset.data.size() * 9 / 10;
  vector<labeled_image> devset;
  std::move(trainingset.data.begin() + cut, trainingset.data.end(), back_inserter(devset));
  trainingset.data.resize(cut);

  mnist_model model(trainingset.width, trainingset.height, trainingset.labels);

  cout << "training" << endl;
  dyana::adam_trainer trainer;
  trainer.num_workers = 4;
  trainer.num_epochs = 4;
  trainer.train_reporting_dev_score(model, trainingset.data, devset);

  cout << "dev testing" << endl;

  unsigned total_cnt{};
  unsigned correct_cnt{};
  for (auto &&dev_img:devset) {
    auto prediction = model(dev_img.pixels);
    total_cnt++;
    if (prediction == dev_img.oracle) correct_cnt++;
  }

  cout << "dev accuracy = " << (correct_cnt / (double) total_cnt) << endl;

  return 0;
}
