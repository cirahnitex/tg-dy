//
// Created by YAN Yuchen on 5/8/2018.
//

#ifndef DYANA_READOUT_LAYER_HPP
#define DYANA_READOUT_LAYER_HPP

#include <dynet/dynet.h>
#include <random>
#include "dyana.hpp"

namespace dyana {
  /**
   * a model that predicts a label given a tensor<X>
   * sometimes this input tensor is called embedding
   */
  class readout_model {
  public:
    readout_model() = default;

    readout_model(const readout_model &) = default;

    readout_model(readout_model &&) = default;

    readout_model &operator=(const readout_model &) = default;

    readout_model &operator=(readout_model &&) = default;

    /**
     * construct given a set of labels
     * \param labels the set of all possible labels
     */
    template<typename RANGE_EXP>
    explicit readout_model(RANGE_EXP &&labels) :
      dict(labels), fc(dict.size()) {
        if(dict.size()<=1) throw std::runtime_error("dyana::readout_model: label set cannot be empty");
    }

    operator bool() const {
      return (bool)fc;
    }

    /**
     * given an embedding, predict a label
     * \param embedding tensor<X>
     * \return the predicted label
     */
    std::string operator()(const dyana::tensor &embedding) {
      return dict.convert(dyana::argmax_index(fc.operator()(embedding)));
    }

    /**
     * given an embedding, generate a label according to all label's weight distribution
     * \param embedding tensor<X>
     * \return the generated label
     */
    std::string generate(const dyana::tensor &embedding) {
      auto weights = dyana::softmax(fc.operator()(embedding)).as_vector();
      std::discrete_distribution<unsigned> d(weights.begin(), weights.end());
      return dict.convert(d(*dynet::rndeng));
    }

    /**
     * compute the probability of generating some label from some embedding
     * \param embedding embedding tensor<x>
     * \param label the label to generate
     * \return
     */
    dyana::tensor recognize(const dyana::tensor &embedding, const std::string &label) {
      return dyana::softmax(fc.operator()(embedding)).at(get_internal_label_id(label));
    }

    std::vector<unsigned> randomly_sample_ids(unsigned num_samples) {
      std::vector<unsigned> ids(size());
      std::iota(ids.begin(), ids.end(), 0);
      if(num_samples >= size()) {
        return ids;
      }
      for(unsigned i=0; i<num_samples; i++) {
        unsigned swap_target = i + dynet::rand0n(size() - i);
        unsigned t = ids[swap_target];
        ids[swap_target] = ids[i];
        ids[i] = t;
      }
      ids.resize(num_samples);
      return std::vector<unsigned>(ids.begin(), ids.begin() + num_samples);
    }

    dyana::tensor sampled_readout_loss(const dyana::tensor &embedding, const std::string &oracle, const std::vector<unsigned> sampled_ids) {

      auto oracle_id = get_internal_label_id(oracle);

      std::vector<unsigned> filtered_ids{oracle_id};
      filtered_ids.reserve(sampled_ids.size() + 1);
      for(auto&& sampled_id:sampled_ids) {
        if(sampled_id != oracle_id) {
          filtered_ids.push_back(sampled_id);
        }
      }

      auto logits = fc.predict_given_output_positions(embedding, filtered_ids);
      return dyana::pickneglogsoftmax(logits, (unsigned) 0);
    }

    /**
     * compute the readout loss.
     * only a fixed number of false candidates are randomly selected
     * the number equals to ( SAMPLED_READOUT_NUM_SAMPLES - 1 )
     * \param embedding the embedding to readout from
     * \param oracle the true answer
     * \param num_false_samples number of false samples to provide
     * \return
     */
    dyana::tensor sampled_readout_loss(const dyana::tensor &embedding, const std::string &oracle, unsigned num_false_samples) {
      if(num_false_samples >= size()) return compute_loss(embedding, oracle);
      return sampled_readout_loss(embedding, oracle, randomly_sample_ids(num_false_samples));
    }

    /**
     * given an embedding and a desired label, compute the loss
     * \param embedding tensor<X>
     * \param oracle the desired label
     * \return the loss
     */
    dyana::tensor compute_loss(const dyana::tensor &embedding, const std::string &oracle) {
      return dyana::pickneglogsoftmax(fc.operator()(embedding), get_internal_label_id(oracle));
    }

    /**
     * batched compute_loss
     * given N embedding and N desired labels, compute the loss
     * \param batched_embedding tensor<X>, batch N
     * \param oracles N desired label
     * \return the loss
     */
    dyana::tensor compute_loss(const dyana::tensor &batched_embedding, const std::vector<std::string> &oracles) {
      std::vector<unsigned> ids;
      for(auto&& oracle:oracles) {
        ids.push_back(get_internal_label_id(oracle));
      }
      return dyana::pickneglogsoftmax(fc.operator()(batched_embedding), ids);
    }

    /**
     * given an embedding and a desired label, compute the loss
     * normalized in such a way that a random model is expected to have loss = 1
     * \param embedding
     * \param oracle
     * \return
     */
    dyana::tensor compute_normalized_loss(const dyana::tensor &embedding, const std::string& oracle) {
      return compute_loss(embedding, oracle) / get_normalization_divider();
    }

    /**
     * batched compute_loss_normalized
     * given an embedding and a desired label, compute the loss
     * normalized in such a way that a random model is expected to have loss = 1
     * \param batched_embedding
     * \param oracles
     * \return
     */
    dyana::tensor compute_normalized_loss(const dyana::tensor &batched_embedding, const std::vector<std::string>& oracles) {
      return compute_loss(batched_embedding, oracles) / get_normalization_divider() / (float)oracles.size();
    }

    /**
     * get the size of label dictionary
     * this might be different from the size of labels it constructs with
     * because the labels it constructs might can have duplicates
     * \return the size of label dictionary
     */
    unsigned size() const {
      return dict.size();
    }

    /**
     * get the ID of the label that is used internally of this readout structure
     * \param label the label to get internal ID
     * \return the ID of the label
     */
    unsigned get_internal_label_id(const std::string &label) const {
      return dict.convert(label);
    }

    /**
     * get the one-hot representation of the label, according to the internal label ID
     * \param label the label to represent
     * \return tensor<#-of-labels>, all values are 0 except 1 at the position of internal label ID
     */
    dyana::tensor one_hot(const std::string &label) const {
      return dict.one_hot(label);
    }

    /**
     * list all the labels.
     * the list might be different from what is feed into the constructor
     * but constructing another readout_layer with this label list garentees the same internal label ID
     * \return a list of labels
     */
    const std::vector<std::string> &list_labels() const {
      return dict.list_entries();
    }

    EASY_SERIALIZABLE(dict, fc)

  private:

    dyana::immutable_dict dict;
    dyana::linear_dense_layer fc;
    float normalization_divider{};

    float get_normalization_divider() {
      if(normalization_divider > 0) return normalization_divider;
      normalization_divider = log(dyana::tensor((float)size())).as_scalar();
      return normalization_divider;
    }
  };
}


#endif //DYANA_READOUT_LAYER_HPP
