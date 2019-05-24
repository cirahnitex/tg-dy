//
// Created by YAN Yuchen on 5/1/2018.
//

#ifndef DYANA_LOOKUP_TABLE_HPP
#define DYANA_LOOKUP_TABLE_HPP

#include "dyana_common.hpp"
#include <dynet/dynet.h>
#include <dynet/dict.h>
#include <regex>
#include "dyana_serialization_helper.hpp"


namespace dyana {
  class embedding_lookup {
  public:
    embedding_lookup() = default;

    embedding_lookup(const embedding_lookup &) = default;

    embedding_lookup(embedding_lookup &&) = default;

    embedding_lookup &operator=(const embedding_lookup &) = default;

    embedding_lookup &operator=(embedding_lookup &&) = default;

    /**
     * construct with a list of tokens. there is no garentee for the internal ID of the tokens.
     * but constructing with the same list of tokens will result in the same internal node ID.
     * in addition, EPSILON and UNK will be automatically added in the dictionary
     * \param embedding_size the size of embedding
     * \param tokens the list of tokens
     */
    template<typename RANGE_EXP>
    embedding_lookup(unsigned embedding_size, RANGE_EXP &&tokens) :
      dict(std::make_shared<dynet::Dict>()),
      capacity(),
      embedding_size(embedding_size),
      lookup_table() {
      token_to_id(""); // force epsilon to be #0 token
      for (auto itr = tokens.begin(); itr != tokens.end(); ++itr) {
        token_to_id(*itr);
      }
      dict->freeze();
      dict->set_unk(_DYNET_WRAPPER_DEFAULT_UNK);
      capacity = dict->size();
      lookup_table = lookup_parameter(capacity, {embedding_size});
    }

    /**
     * construct with a list of tokens, also specifiing how to get initial embeddings
     * \param embedding_size the size of embedding
     * \param tokens the list of tokens
     * \param lookup_init_embedding how to get initial embedding from a token. this function may return an empty std::vector or even throw an exception to indicate that the initial embedding of a given token is unknown
     */
    template<typename RANGE_EXP>
    embedding_lookup(unsigned embedding_size, RANGE_EXP &&tokens,
                     std::function<std::vector<float>(const std::string &)> lookup_init_embedding) : embedding_lookup(
      embedding_size, tokens) {
      for (const auto &token:list_tokens()) {
        auto id = token_to_id(token);
        try {
          lookup_table.initialize(id, resize_fill_random(lookup_init_embedding(token), embedding_size));
        }
        catch (...) {
          lookup_table.initialize(id, resize_fill_random(std::vector<float>(), embedding_size));
        }

      }
    }

    /**
     * construct with a tokens and their initial embeddings
     * \tparam map_args_T unordered map template parameters. left empty if you have no idea.
     * \param embedding_size the size of embedding
     * \param token_embeddings the token to initial embedding map. to indicate that the initial embedding of a given token is unknown, set its initial embedding to be an empty std::vector.
     */
    template<typename... map_args_T>
    embedding_lookup(unsigned embedding_size,
                     const std::unordered_map<std::string, std::vector<float>, map_args_T...> &token_embeddings) :
      dict(std::make_shared<dynet::Dict>()),
      capacity(),
      embedding_size(embedding_size),
      lookup_table() {
      token_to_id(""); // force epsilon to be #0 token
      for (const auto &token_embedding:token_embeddings) {
        token_to_id(token_embedding.first);
      }
      dict->freeze();
      dict->set_unk(_DYNET_WRAPPER_DEFAULT_UNK);
      capacity = dict->size();
      lookup_table = lookup_parameter(capacity, {embedding_size});
      for (const auto &token_embedding:token_embeddings) {
        auto id = token_to_id(token_embedding.first);
        lookup_table.initialize(id, resize_fill_random(token_embedding.second, embedding_size));
      }
    }

    dyana::tensor operator()(const std::string &token, bool as_constant = false) const {
      return lookup(token_to_id(token), as_constant);
    }

    std::vector<dyana::tensor> operator()(const std::vector<std::string> &tokens, bool as_constant = false) const {
      std::vector<dyana::tensor> ret;
      for (auto itr = tokens.begin(); itr != tokens.end(); ++itr) {
        ret.push_back(operator()(*itr, as_constant));
      }
      return ret;
    }

    unsigned token_to_id(const std::string &token) const {
      return (unsigned) (dict)->convert(token);
    }

    std::string id_to_token(unsigned id) const {
      return dict->convert(id);
    }

    unsigned real_dict_size() const { return dict->size(); }

    /**
     * list all the tokens stored in dictionary, excluding EPSILON and UNK.
     */
    std::vector<std::string> list_tokens() const {
      std::vector<std::string> ret;
      for (const auto &word:dict->get_words()) {
        if (word.empty()) continue;
        if (word == _DYNET_WRAPPER_DEFAULT_UNK) continue;
        ret.push_back(word);
      }
      return ret;
    }

    const std::vector<std::string> &list_real_tokens() const {
      return dict->get_words();
    }

    template<class Archive>
    void save(Archive &a) const {
      a(cereal::make_nvp("vocab", dict->get_words()));
      a(cereal::make_nvp("capacity", capacity));
      a(cereal::make_nvp("embedding_size", embedding_size));
      a(cereal::make_nvp("lookup_table", lookup_table));
    }

    template<class Archive>
    void load(Archive &a) {
      std::vector<std::string> tokens;
      a(tokens);
      dict = std::make_shared<dynet::Dict>();
      for (const auto &entry:tokens) {
        dict->convert(entry);
      }
      dict->freeze();
      dict->set_unk(_DYNET_WRAPPER_DEFAULT_UNK);
      a(capacity, embedding_size, lookup_table);
    }

  protected:
    std::shared_ptr<dynet::Dict> dict;
    unsigned capacity;
    unsigned embedding_size;
    dyana::lookup_parameter lookup_table;

    dyana::tensor lookup(unsigned token_id, bool as_constant = false) const {
      return as_constant ? lookup_table.const_lookup(token_id) : lookup_table.lookup(token_id);
    }

    static std::vector<float> resize_fill_random(const std::vector<float> &arr, unsigned size) {
      if (arr.size() == size) {
        return arr;
      } else if (arr.size() > size) { return std::vector<float>(arr.begin(), arr.begin() + size); }
      else {
        std::vector<float> ret(arr);
        for (auto i = arr.size(); i < size; i++) {
          ret.push_back(dynet::rand01());
        }
        return ret;
      }
    }
  };
}

#endif //DYANA_LOOKUP_TABLE_HPP
