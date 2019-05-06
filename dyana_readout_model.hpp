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
    readout_model(const std::unordered_set<std::string> &labels) :
      dict(labels), fc(dict.size()) {
    }

    /**
     * construct given two sets of labels.
     * this is the same as calling the previous constructor with two sets concatenated together
     * \param labels the set of all possible labels, part-I
     * \param more_labels the set of all possible labels, part-II
     */
    readout_model(const std::unordered_set<std::string> &labels, const std::unordered_set<std::string> &more_labels) :
      dict(labels, more_labels), fc(dict.size()) {
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

    /**
     * given an embedding and a desired label, compute the loss
     * if there are too many labels, it will compute sampled_readout_loss instead
     * two private constants control the behavior of sampled_readout_loss, see later documentations
     * \param embedding tensor<X>
     * \param oracle the desired label
     * \return the loss
     */
    dyana::tensor compute_loss(const dyana::tensor &embedding, const std::string &oracle) {
      if (size() > SAMPLED_READOUT_THRESHOLD) return sampled_readout_loss(embedding, oracle);
      return dyana::pickneglogsoftmax(fc.operator()(embedding), get_internal_label_id(oracle));
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
    /**
     * if #labels greater than this number, loss will be computed by sampled_readout_loss
     */
    static constexpr unsigned SAMPLED_READOUT_THRESHOLD = 128;

    /**
     * controls how many samples to take when computing sampled_readout_loss
     */
    static constexpr unsigned SAMPLED_READOUT_NUM_SAMPLES = 32;

    dyana::immutable_dict dict;
    dyana::linear_dense_layer fc;

    /**
     * compute the readout loss.
     * only a fixed number of false candidates are randomly selected
     * the number equals to ( SAMPLED_READOUT_NUM_SAMPLES - 1 )
     * \param embedding the embedding to readout from
     * \param oracle the true answer
     * \return
     */
    dyana::tensor sampled_readout_loss(const dyana::tensor &embedding, const std::string &oracle) {
      std::vector<unsigned> sampled_ids(SAMPLED_READOUT_NUM_SAMPLES);
      auto oracle_id = get_internal_label_id(oracle);
      sampled_ids[0] = oracle_id;
      for (unsigned i = 1; i < SAMPLED_READOUT_NUM_SAMPLES; i++) {
        unsigned false_option = dynet::rand0n(size() - 1); // -1 because false options exclude oracle
        if (false_option >= oracle_id) false_option++;
        sampled_ids[i] = false_option;
      }

      auto logits = fc.predict_given_output_positions(embedding, sampled_ids);
      return dyana::pickneglogsoftmax(logits, (unsigned) 0);
    }
  };
}


#endif //DYANA_READOUT_LAYER_HPP
