//
// Created by YAN Yuchen on 5/1/2018.
//

#ifndef DYANA_LOOKUP_READOUT_HPP
#define DYANA_LOOKUP_READOUT_HPP

#include "dyana_common.hpp"
#include "dyana_operations.hpp"
#include <dynet/dynet.h>
#include <dynet/dict.h>
#include <cereal/types/base_class.hpp>
#include "dyana_embedding_lookup.hpp"

namespace dyana {
  class mono_lookup_readout : public embedding_lookup {
  public:
    mono_lookup_readout() = default;

    mono_lookup_readout(const mono_lookup_readout &) = default;

    mono_lookup_readout(mono_lookup_readout &&) = default;

    mono_lookup_readout &operator=(const mono_lookup_readout &) = default;

    mono_lookup_readout &operator=(mono_lookup_readout &&) = default;

    static const unsigned SAMPLE_THRESHOLD = 128;

    mono_lookup_readout(unsigned embedding_size, const std::unordered_set<std::string> &tokens) :
      embedding_lookup(embedding_size, tokens),
      readout_table({embedding_size + 1, capacity}) // embedding +1 for bias
    {}

    mono_lookup_readout(unsigned embedding_size, const std::unordered_set<std::string> &tokens,
                        std::function<std::vector<float>(const std::string &)> lookup_init_embedding) :
      embedding_lookup(embedding_size, tokens, lookup_init_embedding),
      readout_table({embedding_size + 1, capacity}) // embedding +1 for bias
    {}

    mono_lookup_readout(unsigned embedding_size,
                        const std::unordered_map<std::string, std::vector<float>> &token_embeddings) :
      embedding_lookup(embedding_size, token_embeddings),
      readout_table({embedding_size + 1, capacity}) // embedding +1 for bias
    {}

    std::pair<dyana::tensor, dyana::tensor> transduce_with_loss(const std::string &token) const {
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
    std::pair<std::vector<dyana::tensor>, dyana::tensor> transduce_with_loss(
      const std::vector<std::string> &tokens) const {
      std::vector<unsigned> ids;
      std::vector<dyana::tensor> ret_embeddings;
      for (auto itr = tokens.begin(); itr != tokens.end(); ++itr) {
        auto id = token_to_id(*itr);
        ids.push_back(id);
        ret_embeddings.push_back(lookup(id));
      }
      return std::make_pair(std::move(ret_embeddings),
                            compute_windowed_readout_loss(dyana::concatenate_to_batch(ret_embeddings), ids));
    }

    /**
     * get back the token from an embedding
     * \param embedding the embedding, dim(1) expression
     * \return the token
     */
    std::string operator()(const dyana::tensor &embedding) const {
      return dict->convert(dyana::argmax_index(forward(embedding)));
    }

    /**
     * given an embedding, generate a token according to all token's weight distribution
     * \param embedding dim(1) expression
     * \return the generated token
     */
    std::string generate(const dyana::tensor &embedding) const {
      auto weights = dyana::softmax(forward(embedding)).as_vector();
      std::discrete_distribution<unsigned> d(weights.begin(), weights.end());
      return dict->convert(d(*dynet::rndeng));
    }

    /**
     * get back a sentence from a list of embeddings
     * \param embeddings a list of embeddings
     * \return the sentence
     */
    std::vector<std::string> operator()(const std::vector<dyana::tensor> &embeddings) const {
      std::vector<std::string> ret;
      for (auto itr = embeddings.begin(); itr != embeddings.end(); ++itr) {
        ret.push_back(operator()(*itr));
      }
      return ret;
    }

    /**
     * genreate a random sentence from a list of embeddings
     * \param embeddings a list of embeddings
     * \return the sentence
     */
    std::vector<std::string> generate(const std::vector<dyana::tensor> &embeddings) const {
      std::vector<std::string> ret;
      for (auto itr = embeddings.begin(); itr != embeddings.end(); ++itr) {
        ret.push_back(generate(*itr));
      }
      return ret;
    }

    /**
     * get the topX readouts from an embedding
     * \param embedding
     * \param topX
     * \return a list of tokens, together with a loss for each token. tokens are sorted from best to worst
     */
    std::vector<std::pair<std::string, float>> top_readouts(const dyana::tensor &embedding, unsigned topX) {
      auto losses = (-dyana::log_softmax(forward(embedding))).as_vector();
      std::vector<unsigned> ids(real_dict_size());
      std::iota(ids.begin(), ids.end(), 0);
      std::sort(ids.begin(), ids.end(), [&](unsigned a, unsigned b) { return losses[a] < losses[b]; });
      std::vector<std::pair<std::string, float>> ret;
      for (unsigned i = 0; i < topX; i++) {
        if (i >= ids.size()) break;
        ret.push_back(std::make_pair(dict->convert(ids[i]), losses[ids[i]]));
      }
      return ret;
    }

    /**
     * train the model to read out a token from an embedding
     * \param embedding the embedding
     * \param oracle the desired token
     * \return the loss to train on
     */
    dyana::tensor compute_readout_loss(const dyana::tensor &embedding, const std::string &oracle) const {
      return compute_windowed_readout_loss(embedding, std::vector<unsigned>({token_to_id(oracle)}));
    }

    /**
     * train the model to readout (any one from a token set) from an embedding
     * \param embedding the embedding
     * \param possible_oracles the desired token set. reading out any of them is considered correct.
     * \return
     */
    dyana::tensor compute_readout_loss_multi_oracle(const dyana::tensor &embedding,
                                                    const std::vector<std::string> &possible_oracles) const {
      std::vector<unsigned> oracle_ids;
      for (const auto &token:possible_oracles) {
        oracle_ids.push_back(token_to_id(token));
      }
      auto one = dyana::ones({1});
      if (capacity <= SAMPLE_THRESHOLD) {
        auto logits = (dyana::concatenate({embedding, one}).transpose() * dyana::tensor(readout_table)).transpose();
        return -dyana::max_dim(dyana::log_softmax(logits).select_rows(oracle_ids));
      } else {
        std::vector<unsigned> sampled_token_ids;
        std::vector<unsigned> remapped_oracles;
        {
          std::unordered_map<unsigned, unsigned> ori_to_remapped_oracle;

          // put all oracle tokens in sample. the neural network should learn to correctly select them
          for (auto token_id:oracle_ids) {
            try {
              remapped_oracles.push_back(ori_to_remapped_oracle.at(token_id));
            }
            catch (...) {
              ori_to_remapped_oracle[token_id] = sampled_token_ids.size();
              remapped_oracles.push_back(sampled_token_ids.size());
              sampled_token_ids.push_back(token_id);
            }
          }

          // randomly sample some other unrelated tokens. the neural network should learn to avoid them
          for (unsigned i = 0; i < SAMPLE_THRESHOLD; i++) {
            const auto rand_id = dynet::rand0n(capacity);
            if (ori_to_remapped_oracle.count(rand_id) <= 0) {
              ori_to_remapped_oracle[rand_id] = sampled_token_ids.size();
              sampled_token_ids.push_back(rand_id);
            }
          }

        }

        // fetch the readouts involved in sample
        auto remapped_readout_table = dyana::tensor(readout_table).select_cols(sampled_token_ids);

        auto logits = (dyana::concatenate({embedding, one}).transpose() * remapped_readout_table).transpose();
        return -dyana::max_dim(dyana::log_softmax(logits).select_rows(remapped_oracles));
      }
    }

    /**
     * train the model to read out a sentence from its token embeddings.
     * this function is much faster than calling compute_readout_loss on each individual token embeddings.
     * \param embeddings
     * \param oracle
     * \return the loss to train on
     */
    dyana::tensor compute_readout_loss(const std::vector<dyana::tensor> &embeddings,
                                       const std::vector<std::string> &oracle) const {
      std::vector<unsigned> oracle_ids;
      for (const auto &token:oracle) {
        oracle_ids.push_back(token_to_id(token));
      }
      return compute_windowed_readout_loss(dyana::concatenate_to_batch(embeddings), oracle_ids);
    }

    template<class Archive>
    void save(Archive &ar) const {
      embedding_lookup::save(ar);
      ar(readout_table);
    }

    template<class Archive>
    void load(Archive &ar) {
      embedding_lookup::load(ar);
      ar(readout_table);
    }

  protected:
    dyana::parameter readout_table;

    dyana::tensor forward(const dyana::tensor &embedding) const {
      return (dyana::concatenate({embedding, dyana::tensor(1)}).transpose() * dyana::tensor(readout_table)).transpose();
    }

    dyana::tensor
    compute_windowed_readout_loss(const dyana::tensor &embeddings_batch, const std::vector<unsigned> &oracles) const {
      auto one = dyana::ones({1});
      if (capacity <= SAMPLE_THRESHOLD) {
        auto logits = (dyana::concatenate({embeddings_batch, one}).transpose() *
                       dyana::tensor(readout_table)).transpose();
        return dyana::sum_batches(dyana::pickneglogsoftmax(logits, oracles));
      } else {
        std::vector<unsigned> sampled_token_ids;
        std::vector<unsigned> remapped_oracles;
        {
          std::unordered_map<unsigned, unsigned> ori_to_remapped_oracle;

          // put all oracle tokens in sample. the neural network should learn to correctly select them
          for (auto token_id:oracles) {
            try {
              remapped_oracles.push_back(ori_to_remapped_oracle.at(token_id));
            }
            catch (...) {
              ori_to_remapped_oracle[token_id] = sampled_token_ids.size();
              remapped_oracles.push_back(sampled_token_ids.size());
              sampled_token_ids.push_back(token_id);
            }
          }

          // randomly sample some other unrelated tokens. the neural network should learn to avoid them
          for (unsigned i = 0; i < SAMPLE_THRESHOLD; i++) {
            const auto rand_id = dynet::rand0n(capacity);
            if (ori_to_remapped_oracle.count(rand_id) <= 0) {
              ori_to_remapped_oracle[rand_id] = sampled_token_ids.size();
              sampled_token_ids.push_back(rand_id);
            }
          }

        }

        // fetch the readouts involved in sample
        auto remapped_readout_table = dyana::tensor(readout_table).select_cols(sampled_token_ids);

        auto logits = (dyana::concatenate({embeddings_batch, one}).transpose() * remapped_readout_table).transpose();
        return dyana::sum_batches(dyana::pickneglogsoftmax(logits, remapped_oracles));
      }
    }
  };
}


#endif //DYANA_LOOKUP_READOUT_HPP
