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
      lookup_table = dyana::lookup_parameter(capacity, {embedding_size});
    }

    /**
     * Set the embedding of a token.
     * \param token the token to set embedding
     * \param embedding the embedding
     */
    void set_value(const std::string& token, const std::vector<float>& embedding) {
      if(embedding.size() != embedding_size) {
        std::stringstream msg;
        msg << "embedding_lookup::set_value(): provided embedding size ("
            << embedding.size()
            << ") doesn't match the layer embedding size ("
            << embedding_size << ")";
        throw std::runtime_error(msg.str());
      }
      auto id = token_to_id(token);
      lookup_table.set_value(id, embedding);
    }

    explicit operator bool() const {
      return embedding_size > 0;
    }

    dyana::tensor operator()(const std::string &token) const {
      return lookup(token_to_id(token));
    }

    /**
     * lookup multiple tokens
     * \param tokens the list of tokens to lookup
     * \return token embeddings column by column.
     *         specifically, tensor<D,N>
     *         where D = embedding-size
     *         and   N = sentence-length
     */
    dyana::tensor operator()(const std::vector<std::string> &tokens) const {
      std::vector<unsigned> ids;
      for(auto itr = tokens.begin(); itr != tokens.end(); ++itr) {
        ids.push_back(token_to_id(*itr));
      }
      return lookup(ids);
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

    unsigned get_embedding_size() const {
      return embedding_size;
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

    dyana::tensor lookup(unsigned token_id) const {
      return lookup_table.lookup(token_id);
    }

    dyana::tensor lookup(const std::vector<unsigned>& token_ids) const {
      return lookup_table.lookup(token_ids).reshape({embedding_size, (unsigned)token_ids.size()});
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
