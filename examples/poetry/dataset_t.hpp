//
// Created by YAN Yuchen on 11/20/2018.
//

#ifndef DYNET_WRAPPER_DATASET_T_HPP
#define DYNET_WRAPPER_DATASET_T_HPP

#include <vector>
#include <string>
#include <json/json.h>

typedef std::vector<std::string> datum_t;
datum_t parse_datum_json(const Json::Value& json) {
  datum_t ret;
  for(const Json::Value& token_json:json) {
    ret.push_back(token_json.asString());
  }
  return ret;
}
typedef std::vector<datum_t> dataset_t;
dataset_t parse_dataset_json(const Json::Value& json) {
  dataset_t ret;
  for(const Json::Value& token_json:json) {
    ret.push_back(parse_datum_json(token_json));
  }
  return ret;
}

#endif //DYNET_WRAPPER_DATASET_T_HPP
