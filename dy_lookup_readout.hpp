//
// Created by YAN Yuchen on 5/1/2018.
//

#ifndef DYNET_WRAPPER_DY_LOOKUP_READOUT_HPP
#define DYNET_WRAPPER_DY_LOOKUP_READOUT_HPP
#include "dy_common.hpp"
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
      mono_lookup_readout(unsigned embedding_size, const std::vector<std::string>& tokens):
        embedding_lookup(embedding_size, tokens),
        readout_table(add_lookup_parameters(capacity, {embedding_size+1})) // embedding +1 for bias
      {}
      mono_lookup_readout(unsigned embedding_size, const std::vector<std::string>& tokens, std::function<std::vector<float>(const std::string&)> lookup_init_embedding):
        embedding_lookup(embedding_size, tokens, lookup_init_embedding),
        readout_table(add_lookup_parameters(capacity, {embedding_size+1})) // embedding +1 for bias
      {}
      template<typename ...map_args_T>
      mono_lookup_readout(unsigned embedding_size, const std::unordered_map<std::string, std::vector<float>, map_args_T...>& token_embeddings):
          embedding_lookup(embedding_size, token_embeddings),
          readout_table(add_lookup_parameters(capacity, {embedding_size+1})) // embedding +1 for bias
      {}
      std::pair<std::vector<dynet::Expression>, dynet::Expression> read_sentence_with_loss(const std::string& line) const {
        auto tokens = embedding_lookup::re_split(line);
        return read_sentence_with_loss(tokens);
      };

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

      std::string readout(const dynet::Expression& embedding) const { return dict->convert(embedding_to_id(embedding)); }
      std::string readout(const std::vector<dynet::Expression>& embeddings) const {
        std::string ret;
        for(auto itr = embeddings.begin(); itr!=embeddings.end(); ++itr) {
          ret += readout(*itr) + " ";
        }
        return ret;
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

      dynet::Expression windowed_readout_loss(const std::vector<unsigned>& selected_ids) const {

        // remove duplicates
        // we call the duplicate-removed IDs "sample"
        std::set<unsigned> t(selected_ids.begin(), selected_ids.end());

        // training windowed loss on a sentence typically doesn't involve epsilon
        // training epsilon is important. add epsilon into training anyway.
        t.insert(token_to_id(""));

        std::vector<unsigned> sampled_ids(t.begin(), t.end());

        // alias the sample size, because its frequently referred
        unsigned sampled_size = (unsigned)sampled_ids.size();

        // fetch the embeddings involved in sample
        auto embedding_batch = dynet::lookup(dy::cg(), lookup_table, sampled_ids);

        // fetch the readouts involved in sample
        auto sampled_readout_table = dynet::reshape(dynet::lookup(dy::cg(), readout_table, sampled_ids),{embedding_size+1,(unsigned)sampled_ids.size()});
        sampled_readout_table = dynet::transpose(sampled_readout_table);

        // multiply to get readout logits
        auto one = dynet::input(dy::cg(), {1}, {(dynet::real)1});
        auto logit_batch = sampled_readout_table * dynet::concatenate({embedding_batch, one});

        // every sample should readout as itself
        std::vector<unsigned> desired_readout(sampled_size);
        iota(desired_readout.begin(), desired_readout.end(), 0);

        // return the loss
        return dynet::sum_batches(dynet::pickneglogsoftmax(logit_batch, desired_readout));
      }
      std::pair<dynet::Expression, dynet::Expression> lookup_with_sampled_loss(unsigned token_id) const {
        unsigned vocab_size = dict->size();

        // lookup the embedding
        auto embedding = lookup(token_id);

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
        auto loss = pickneglogsoftmax(logits, (unsigned)0);

        return std::make_pair(std::move(embedding), std::move(loss));
      }
    };
  }
}

#endif //DYNET_WRAPPER_DY_LOOKUP_READOUT_HPP
