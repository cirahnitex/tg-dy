//
// Created by YAN Yuchen on 12/4/2018.
//

#ifndef DYANA_BI_LOOKUP_READOUT_HPP
#define DYANA_BI_LOOKUP_READOUT_HPP
#include "dyana_common.hpp"
#include "dyana_operations.hpp"
#include <dynet/dynet.h>
#include <cereal/types/base_class.hpp>
#include "dyana_embedding_lookup.hpp"
#include "dyana_serialization_helper.hpp"
#include "dyana_utils.hpp"
namespace tg {
  namespace dyana {
    class bi_lookup_readout {
      dyana::embedding_lookup l0_lookup;
      dyana::embedding_lookup l1_lookup;
      dyana::parameter l0_readout;
      dyana::parameter l1_readout;
      static const unsigned SAMPLE_THRESHOLD = 128;
      static std::pair<std::function<dyana::tensor()>, std::unordered_map<std::string, unsigned>> create_readout_window(const dyana::embedding_lookup& lookup, const dyana::parameter& readout, const std::unordered_set<std::string>& tokens_involved) {
        if(lookup.real_dict_size()<=SAMPLE_THRESHOLD) {
          std::unordered_map<std::string, unsigned> window_readout_dict;
          const auto& tokens = lookup.list_real_tokens();
          for(unsigned i=0; i<tokens.size(); i++) {
            window_readout_dict[tokens[i]] = i;
          }
          return std::make_pair([&](){return dyana::tensor(readout);}, window_readout_dict);
        }
        else {
          std::unordered_map<std::string, unsigned> window_readout_dict;
          std::vector<unsigned> sampled_ori_ids;
          const auto capacity = lookup.real_dict_size();
          for(const auto& token:tokens_involved) {
            unsigned ori_id = lookup.token_to_id(token);
            unsigned remapped_id = sampled_ori_ids.size();
            sampled_ori_ids.push_back(ori_id);
            window_readout_dict[token] = remapped_id;
          }
          for(unsigned i=0; i<SAMPLE_THRESHOLD; i++) {
            const auto rand_id = dynet::rand0n(capacity);
            const auto token = lookup.id_to_token(rand_id);
            if(window_readout_dict.count(token)>0) continue;
            unsigned remapped_id = sampled_ori_ids.size();
            sampled_ori_ids.push_back(rand_id);
            window_readout_dict[token] = remapped_id;
          }
          return std::make_pair([&readout, sampled_ori_ids](){return dyana::tensor(readout).select_cols(sampled_ori_ids);}, window_readout_dict);
        }
      }
    public:
      EASY_SERIALIZABLE(l0_lookup, l1_lookup, l0_readout, l1_readout)
      bi_lookup_readout() = default;
      bi_lookup_readout(const bi_lookup_readout&) = default;
      bi_lookup_readout(bi_lookup_readout&&) = default;
      bi_lookup_readout &operator=(const bi_lookup_readout&) = default;
      bi_lookup_readout &operator=(bi_lookup_readout&&) = default;
      bi_lookup_readout(unsigned embedding_size, const std::unordered_set<std::string>& l0_tokens, const std::unordered_set<std::string>& l1_tokens):l0_lookup(embedding_size, l0_tokens), l1_lookup(embedding_size, l1_tokens), l0_readout({embedding_size+1, l0_lookup.real_dict_size()}), l1_readout({embedding_size+1, l1_lookup.real_dict_size()}) {

      }
      bi_lookup_readout(unsigned embedding_size, const std::unordered_set<std::string>& l0_tokens, const std::unordered_set<std::string>& l1_tokens, std::function<std::vector<float>(const std::string&)> get_l0_init_embedding, std::function<std::vector<float>(const std::string&)> get_l1_init_embedding): l0_lookup(embedding_size, l0_tokens, get_l0_init_embedding), l1_lookup(embedding_size, l1_tokens, get_l1_init_embedding), l0_readout({embedding_size+1, l0_lookup.real_dict_size()}), l1_readout({embedding_size+1, l1_lookup.real_dict_size()}) {

      }
      bi_lookup_readout(unsigned embedding_size, const std::unordered_map<std::string, std::vector<float>>& l0_token_and_embeddings, const std::unordered_map<std::string, std::vector<float>>& l1_token_and_embeddings): l0_lookup(embedding_size, l0_token_and_embeddings), l1_lookup(embedding_size, l1_token_and_embeddings), l0_readout({embedding_size+1, l0_lookup.real_dict_size()}), l1_readout({embedding_size+1, l1_lookup.real_dict_size()}) {}
      dyana::tensor lookup(const std::string& l0_token, const std::string& l1_token) {
        return dyana::max(l0_lookup.transduce(l0_token), l1_lookup.transduce(l1_token));
      }
      std::pair<std::string, std::string> readout(const dyana::tensor& embedding) {
        auto padded_embedding = dyana::concatenate({embedding, dyana::tensor(1)});
        auto l0_token = l0_lookup.id_to_token(dyana::argmax_index((padded_embedding.transpose() * l0_readout).transpose()));
        auto l1_token = l1_lookup.id_to_token(dyana::argmax_index((padded_embedding.transpose() * l1_readout).transpose()));
        return std::make_pair(l0_token, l1_token);
      }
      std::pair<dyana::tensor, dyana::tensor> lookup_with_loss_slow(const std::string& l0_token, const std::string& l1_token) {
        const auto embedding = lookup(l0_token, l1_token);
        auto padded_embedding = dyana::concatenate({embedding, dyana::tensor(1)});
        auto l0_loss = dyana::pickneglogsoftmax((padded_embedding.transpose() * l0_readout).transpose(), l0_lookup.token_to_id(l0_token));
        auto l1_loss = dyana::pickneglogsoftmax((padded_embedding.transpose() * l1_readout).transpose(), l1_lookup.token_to_id(l1_token));
        return std::make_pair(embedding, l0_loss + l1_loss);
      }
      std::function<std::pair<dyana::tensor, dyana::tensor>(const std::string& l0_token, const std::string& l1_token)> create_lookup_with_loss_computer(
        const std::unordered_set<std::string> &l0_tokens_involved,
        const std::unordered_set<std::string> &l1_tokens_involved) {
        auto p0 = create_readout_window(l0_lookup, l0_readout, l0_tokens_involved);
        auto l0_windowed_readout_weight = std::move(p0.first);
        auto l0_windowed_readout_dict = std::make_shared<std::unordered_map<std::string, unsigned>>(std::move(p0.second));
        auto p1 = create_readout_window(l1_lookup, l1_readout, l1_tokens_involved);
        auto l1_windowed_readout_weight = std::move(p1.first);
        auto l1_windowed_readout_dict = std::make_shared<std::unordered_map<std::string, unsigned>>(std::move(p1.second));

        return [this, l0_windowed_readout_weight, l0_windowed_readout_dict, l1_windowed_readout_weight, l1_windowed_readout_dict](const std::string& l0_token, const std::string& l1_token) {
          const auto embedding = lookup(l0_token, l1_token);
          auto padded_embedding = dyana::concatenate({embedding, dyana::tensor(1)});
          auto l0_oracle = l0_windowed_readout_dict->at(l0_token);
          auto l0_loss = dyana::pickneglogsoftmax((padded_embedding.transpose() * l0_windowed_readout_weight()).transpose(), l0_oracle);
          auto l1_oracle = l1_windowed_readout_dict->at(l1_token);
          auto l1_loss = dyana::pickneglogsoftmax((padded_embedding.transpose() * l1_windowed_readout_weight()).transpose(), l1_oracle);
          return std::make_pair(embedding, l0_loss + l1_loss);
        };
      }
      std::vector<std::pair<std::string, float>> translate_l0_to_l1_slow(const std::string& l0_token, unsigned top_X) {
        using namespace std;
        typedef std::pair<std::string, float> entry_type;
        dyana::beam_bucket<entry_type> bucket(top_X, [](const entry_type& a, const entry_type& b){return a.second<b.second;});
        for(const auto l1_token:l1_lookup.list_real_tokens()) {
          bucket.insert(std::make_pair(l1_token, lookup_with_loss_slow(l0_token, l1_token).second.as_scalar()));
        }
        return bucket.move_sorted_values();
      }
      std::vector<std::pair<std::string, float>> translate_l1_to_l0_slow(const std::string& l1_token, unsigned top_X) {
        typedef std::pair<std::string, float> entry_type;
        dyana::beam_bucket<entry_type> bucket(top_X, [](const entry_type& a, const entry_type& b){return a.second<b.second;});
        for(const auto l0_token:l0_lookup.list_real_tokens()) {
          bucket.insert(std::make_pair(l0_token, lookup_with_loss_slow(l0_token, l1_token).second.as_scalar()));
        }
        return bucket.move_sorted_values();
      }
      const std::vector<std::string>& list_l1_real_tokens() {return l0_lookup.list_real_tokens();}
      const std::vector<std::string>& list_l0_real_tokens() {return l1_lookup.list_real_tokens();}
    };
  }
}
#endif //DYANA_BI_LOOKUP_READOUT_HPP
