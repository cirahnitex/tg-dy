//
// Created by YAN Yuchen on 11/19/2018.
//

#ifndef DYNET_WRAPPER_DATA_T_HPP
#define DYNET_WRAPPER_DATA_T_HPP
#include <vector>
#include <string>
#include <unordered_set>
#include <json/json.h>
struct datum_t {
  std::vector<float> input;
  std::string oracle;
  void parse_json(const Json::Value& json) {
    auto json_input = json["input"];
    for(unsigned i=0; i<json_input.size(); i++) {
      input.push_back(json_input[i].asUInt()/255.0);
    }
    oracle = json["oracle"].asString();
  }
};
struct dataset_t {
  std::unordered_set<std::string> labels;
  unsigned width;
  unsigned height;
  std::vector<datum_t> data;
  void parse_json(const Json::Value& json) {
    auto labels_json = json["labels"];
    for(unsigned i=0; i<labels_json.size(); i++) {
      labels.insert(labels_json[i].asString());
    }
    width = json["width"].asUInt();
    height = json["height"].asUInt();

    auto data_json = json["data"];
    for(unsigned i=0; i<data_json.size(); i++) {
      datum_t datum;
      datum.parse_json(data_json[i]);
      this->data.push_back(datum);
    }
  }
};
#endif //DYNET_WRAPPER_DATA_T_HPP
