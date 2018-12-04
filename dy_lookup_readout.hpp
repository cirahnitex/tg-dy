//
// Created by YAN Yuchen on 5/1/2018.
//

#ifndef DYNET_WRAPPER_DY_LOOKUP_READOUT_HPP
#define DYNET_WRAPPER_DY_LOOKUP_READOUT_HPP
#include "dy_common.hpp"
#include "dy_operations.hpp"
#include <dynet/dynet.h>
#include <dynet/dict.h>
#include <cereal/types/base_class.hpp>
#include "dy_embedding_lookup.hpp"
namespace tg {
  namespace dy {
    class mono_lookup_readout: public embedding_lookup {
    public:
      mono_lookup_readout() = default;
      mono_lookup_readout(const mono_lookup_readout&) = default;
      mono_lookup_readout(mono_lookup_readout&&) = default;
      mono_lookup_readout &operator=(const mono_lookup_readout&) = default;
      mono_lookup_readout &operator=(mono_lookup_readout&&) = default;
      static const unsigned SAMPLE_THRESHOLD = 128;
      mono_lookup_readout(unsigned embedding_size, const std::unordered_set<std::string>& tokens):
        embedding_lookup(embedding_size, tokens),
        readout_table(add_parameters({embedding_size+1,capacity})) // embedding +1 for bias
      {}
      mono_lookup_readout(unsigned embedding_size, const std::unordered_set<std::string>& tokens, std::function<std::vector<float>(const std::string&)> lookup_init_embedding):
        embedding_lookup(embedding_size, tokens, lookup_init_embedding),
        readout_table(add_parameters({embedding_size+1,capacity})) // embedding +1 for bias
      {}
      mono_lookup_readout(unsigned embedding_size, const std::unordered_map<std::string, std::vector<float>>& token_embeddings):
          embedding_lookup(embedding_size, token_embeddings),
          readout_table(add_parameters({embedding_size+1,capacity})) // embedding +1 for bias
      {}

      std::pair<dy::tensor, dy::tensor> lookup_with_loss(const std::string& token) const {
        auto id = token_to_id(token);
        auto ret_embedding = lookup(id);
        return std::make_pair(ret_embedding, compute_windowed_readout_loss(ret_embedding, std::vector<unsigned>({id})));
      }

      /**
       * read in a sentence, get its embeddings and a lookup-loss
       * minimizing this lookup-loss can avoide the embedding table from collapsing
       * \param tokens the sentence
       * \return 0) the embeddings
       *         1) the lookup-loss
       */
      std::pair<std::vector<dy::tensor>, dy::tensor> lookup_with_loss(
        const std::vector<std::string> &tokens) const {
        std::vector<unsigned> ids;
        std::vector<dy::tensor> ret_embeddings;
        for(auto itr = tokens.begin(); itr!=tokens.end(); ++itr) {
          auto id = token_to_id(*itr);
          ids.push_back(id);
          ret_embeddings.push_back(lookup(id));
        }
        return std::make_pair(std::move(ret_embeddings), compute_windowed_readout_loss(dy::concatenate_to_batch(ret_embeddings), ids));
      }

      /**
       * get back the token from an embedding
       * \param embedding the embedding, dim(1) expression
       * \return the token
       */
      std::string readout(const dy::tensor& embedding) const { return dict->convert(dy::argmax_index(forward(embedding))); }

      /**
       * given an embedding, generate a token according to all token's weight distribution
       * \param embedding dim(1) expression
       * \return the generated token
       */
      std::string random_readout(const dy::tensor& embedding) const {
        auto weights = dy::softmax(forward(embedding)).as_vector();
        std::discrete_distribution<unsigned> d(weights.begin(), weights.end());
        return dict->convert(d(*dynet::rndeng));
      }

      /**
       * get back a sentence from a list of embeddings
       * \param embeddings a list of embeddings
       * \return the sentence
       */
      std::vector<std::string> readout(const std::vector<dy::tensor>& embeddings) const {
        std::vector<std::string> ret;
        for(auto itr = embeddings.begin(); itr!=embeddings.end(); ++itr) {
          ret.push_back(readout(*itr));
        }
        return ret;
      }

      /**
       * genreate a random sentence from a list of embeddings
       * \param embeddings a list of embeddings
       * \return the sentence
       */
      std::vector<std::string> random_readout(const std::vector<dy::tensor>& embeddings) const {
        std::vector<std::string> ret;
        for(auto itr = embeddings.begin(); itr!=embeddings.end(); ++itr) {
          ret.push_back(random_readout(*itr));
        }
        return ret;
      }

      /**
       * train the model to read out a token from an embedding
       * \param embedding the embedding
       * \param oracle the desired token
       * \return the loss to train on
       */
      dy::tensor compute_readout_loss(const dy::tensor &embedding, const std::string &oracle) const {
        return compute_windowed_readout_loss(embedding, std::vector<unsigned>({token_to_id(oracle)}));
      }

      /**
       * train the model to read out a sentence from its token embeddings.
       * this function is much faster than calling compute_readout_loss on each individual token embeddings.
       * \param embeddings
       * \param oracle
       * \return the loss to train on
       */
      dy::tensor compute_readout_loss(const std::vector<dy::tensor> &embeddings,
                                          const std::vector<std::string> &oracle) const {
        std::vector<unsigned> oracle_ids;
        for(const auto& token:oracle) {
          oracle_ids.push_back(token_to_id(token));
        }
        return compute_windowed_readout_loss(dy::concatenate_to_batch(embeddings), oracle_ids);
      }

      template <class Archive>
      void serialize(Archive& ar)
      {
        ar(cereal::base_class<embedding_lookup>(this), readout_table);
      }
    protected:
      dy::Parameter readout_table;

      dy::tensor forward(const dy::tensor &embedding) const {
        return (dy::concatenate({embedding, dy::tensor(1)}).transpose() * dy::tensor(readout_table)).transpose();
      }

      dy::tensor compute_windowed_readout_loss(const dy::tensor& embeddings_batch, const std::vector<unsigned>& oracles) const {
        auto one = dy::ones({1});
        if(capacity<=SAMPLE_THRESHOLD) {
          auto logits = (dy::concatenate({embeddings_batch, one}).transpose() * dy::tensor(readout_table)).transpose();
          return dy::sum_batches(dy::pickneglogsoftmax(logits, oracles));
        }
        else {
          std::vector<unsigned> sampled_token_ids;
          std::vector<unsigned> remapped_oracles;
          {
            std::unordered_map<unsigned, unsigned> ori_to_remapped_oracle;

            // put all oracle tokens in sample. the neural network should learn to correctly select them
            for(auto token_id:oracles) {
              try {
                remapped_oracles.push_back(ori_to_remapped_oracle.at(token_id));
              }
              catch(...) {
                ori_to_remapped_oracle[token_id] = sampled_token_ids.size();
                remapped_oracles.push_back(sampled_token_ids.size());
                sampled_token_ids.push_back(token_id);
              }
            }

            // randomly sample some other unrelated tokens. the neural network should learn to avoid them
            for(unsigned i=0; i<SAMPLE_THRESHOLD; i++) {
              const auto rand_id = dynet::rand0n(capacity);
              if(ori_to_remapped_oracle.count(rand_id)<=0) {
                ori_to_remapped_oracle[rand_id] = sampled_token_ids.size();
                sampled_token_ids.push_back(rand_id);
              }
            }

          }

          // fetch the readouts involved in sample
          auto remapped_readout_table = dy::tensor(readout_table).select_cols(sampled_token_ids);

          auto logits = (dy::concatenate({embeddings_batch, one}).transpose() * remapped_readout_table).transpose();
          return dy::sum_batches(dy::pickneglogsoftmax(logits, remapped_oracles));
        }
      }
    };
  }
}

#endif //DYNET_WRAPPER_DY_LOOKUP_READOUT_HPP
