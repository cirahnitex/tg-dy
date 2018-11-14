//
// Created by YAN Yuchen on 5/1/2018.
//

#ifndef DYNET_WRAPPER_DY_LOOKUP_READOUT_HPP
#define DYNET_WRAPPER_DY_LOOKUP_READOUT_HPP
#include "dy_common.hpp"
#include "dy_operations.hpp"
#include <dynet/dynet.h>
#include <dict.hpp>
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
      static const unsigned SAMPLE_THRESHOLD = 256;
      mono_lookup_readout(unsigned embedding_size, const std::unordered_set<std::string>& tokens):
        embedding_lookup(embedding_size, tokens),
        readout_table(add_lookup_parameters(capacity, {embedding_size+1})) // embedding +1 for bias
      {}
      mono_lookup_readout(unsigned embedding_size, const std::unordered_set<std::string>& tokens, std::function<std::vector<float>(const std::string&)> lookup_init_embedding):
        embedding_lookup(embedding_size, tokens, lookup_init_embedding),
        readout_table(add_lookup_parameters(capacity, {embedding_size+1})) // embedding +1 for bias
      {}
      mono_lookup_readout(unsigned embedding_size, const std::unordered_map<std::string, std::vector<float>>& token_embeddings):
          embedding_lookup(embedding_size, token_embeddings),
          readout_table(add_lookup_parameters(capacity, {embedding_size+1})) // embedding +1 for bias
      {}

      /**
       * read in a sentence, get its embeddings and a lookup-loss
       * minimizing this lookup-loss can avoide the embedding table from collapsing
       * \param tokens the sentence
       * \return 0) the embeddings
       *         1) the lookup-loss
       */
      std::pair<std::vector<dynet::Expression>, dynet::Expression> read_sentence_with_loss(const std::vector<std::string>& tokens) const {
        std::vector<unsigned> ids;
        std::vector<dynet::Expression> ret_embeddings;
        for(auto itr = tokens.begin(); itr!=tokens.end(); ++itr) {
          auto id = token_to_id(*itr);
          ids.push_back(id);
          ret_embeddings.push_back(lookup(id));
        }
        return std::make_pair(std::move(ret_embeddings), windowed_readout_loss(ids));
      };

      /**
       * get back the token from an embedding
       * \param embedding the embedding
       * \return the token
       */
      std::string readout(const dynet::Expression& embedding) const { return dict->convert(embedding_to_id(embedding)); }

      /**
       * get back a sentence from a list of embeddings
       * \param embeddings a list of embeddings
       * \return the sentence
       */
      std::vector<std::string> readout(const std::vector<dynet::Expression>& embeddings) const {
        std::vector<std::string> ret;
        for(auto itr = embeddings.begin(); itr!=embeddings.end(); ++itr) {
          ret.push_back(readout(*itr));
        }
        return ret;
      }

      /**
       * train the model to read out a token from an embedding
       * \param embedding the embedding
       * \param oracle the desired token
       * \return the loss to train on
       */
      dy::Expression compute_sampled_loss(const dy::Expression& embedding, const std::string& oracle) const {
        return compute_sampled_readout_loss(embedding, token_to_id(oracle));
      }

      /**
       * train the model to read out a sentence from its token embeddings.
       * this function is much faster than calling compute_sampled_loss on each individual token embeddings.
       * \param embeddings
       * \param oracle
       * \return the loss to train on
       */
      dy::Expression compute_windowed_loss(const std::vector<dy::Expression>& embeddings, const std::vector<std::string>& oracle) const {
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
      dynet::LookupParameter readout_table;

      unsigned embedding_to_id(const dynet::Expression &embedding) const {

        // slice the first vocab_size number of readout_table
        std::vector<unsigned> sampled_ids(dict->size());
        iota(sampled_ids.begin(), sampled_ids.end(), 0);
        auto sampled_readout =  dynet::lookup(dy::cg(), readout_table, sampled_ids);

        auto one = dynet::input(dy::cg(), {1}, {(dynet::real)1});
        auto logits = dynet::dot_product(sampled_readout, dynet::concatenate({embedding, one}));
        return dy::argmax_index(dynet::reshape(logits, {(unsigned)sampled_ids.size()}));
      }

      dynet::Expression compute_windowed_readout_loss(const dy::Expression& embedding_batch, const std::vector<unsigned>& oracles) const {
        std::vector<unsigned> unique_oracles;
        std::vector<unsigned> remapped_oracles;
        {
          std::unordered_map<unsigned, unsigned> ori_to_remapped_oracle;
          for(auto token_id:oracles) {
            try {
              remapped_oracles.push_back(ori_to_remapped_oracle.at(token_id));
            }
            catch(...) {
              ori_to_remapped_oracle[token_id] = unique_oracles.size();
              remapped_oracles.push_back(unique_oracles.size());
              unique_oracles.push_back(token_id);
            }
          }
        }

        // fetch the readouts involved in sample
        auto remapped_readout_table = dynet::transpose(dynet::reshape(dynet::lookup(dy::cg(), readout_table, unique_oracles),{embedding_size+1,(unsigned)unique_oracles.size()}));

        // multiply to get readout logits
        auto one = dynet::input(dy::cg(), {1}, {(dynet::real)1});
        auto logit_batch = remapped_readout_table * dynet::concatenate({embedding_batch, one});
        // return the loss
        return dynet::sum_batches(dynet::pickneglogsoftmax(logit_batch, remapped_oracles));
      }

      dynet::Expression windowed_readout_loss(const std::vector<unsigned>& selected_ids) const {

        // fetch the embeddings involved in sample
        auto embedding_batch = dynet::lookup(dy::cg(), lookup_table, selected_ids);

        return compute_windowed_readout_loss(embedding_batch, selected_ids);
      }

      dy::Expression compute_sampled_readout_loss(const dy::Expression& embedding, unsigned token_id) const {
        unsigned vocab_size = dict->size();

        // a placeholder for sampled token IDs
        std::vector<unsigned> sampled_ids(vocab_size);

        // in the beginning, sampled token IDs contains all token IDs.
        // random sampling will happen later if vocab size is greater than sampling threshold
        iota(sampled_ids.begin(), sampled_ids.end(), 0);

        // swap the desired token ID to the front for easy book-keeping
        sampled_ids[0] = token_id;
        sampled_ids[token_id] = 0;

        // random sampling will happen if vocab size is greater than sampling threshold
        if(vocab_size > SAMPLE_THRESHOLD) {

          // random swap the token IDs except the first ID.
          // because the first ID is the desired ID, we just keep it there
          // the swapping stops at index SAMPLE_THRESHOLD, because all the rest will be clipped away,
          // so no need to swap them
          for(unsigned i=1; i<SAMPLE_THRESHOLD; i++) {
            unsigned to_swap = i + dynet::rand0n(vocab_size - i);
            unsigned t = sampled_ids[i];
            sampled_ids[i] = sampled_ids[to_swap];
            sampled_ids[to_swap] = t;
          }

          // clip the sampled IDs to the size SAMPLE_THRESHOLD
          sampled_ids.resize(SAMPLE_THRESHOLD);
        }

        // readout and calculate loss
        auto sampled_readout_table = dynet::lookup(dy::cg(), readout_table, sampled_ids);
        auto one = dynet::input(dy::cg(), {1}, {(dynet::real)1});
        auto logits = dynet::dot_product(sampled_readout_table, dynet::concatenate({embedding, one}));
        logits = dynet::reshape(logits, {(unsigned)sampled_ids.size()});
        auto loss = dy::pickneglogsoftmax(logits, (unsigned)0);

        return loss;
      };
      std::pair<dynet::Expression, dynet::Expression> lookup_with_sampled_loss(unsigned token_id) const {
        auto embedding = lookup(token_id);
        return std::make_pair(embedding, compute_sampled_readout_loss(embedding, token_id));
      }
    };
  }
}

#endif //DYNET_WRAPPER_DY_LOOKUP_READOUT_HPP
