//
// Created by YAN Yuchen on 11/9/2018.
//

#ifndef DYNET_WRAPPER_DY_UTILS_HPP
#define DYNET_WRAPPER_DY_UTILS_HPP

#include <string>
#include <vector>
#include <unordered_map>
#include <algorithm>
#include <dirent.h>
#include <sys/types.h>
#include <sys/stat.h>
#include <ECMAScript_string_utils.hpp>

namespace tg {
  namespace dy {
    /**
     * a token container that give the X most frequent token it has seen
     */
    class frequent_token_collector {
    public:

      /**
       * +1 to a token's occurence
       * \param token
       */
      void add_occurence(const std::string &token) {
        if (counts.count(token) > 0) {
          counts[token]++;
        } else {
          counts[token] = 1;
        }
      }

      /**
       * get the X most frequent tokens.
       * if the tokens seens are less than X, all tokens will be returned.
       * \param size top X
       * \return the list of tokens, ordered from most frequent to most infrequent.
       */
      std::vector<std::string> list_frequent_tokens(unsigned size) const {
        std::vector<std::pair<std::string, unsigned>> pairs;
        for (const auto &p:counts) {
          pairs.push_back(p);
        }
        std::sort(pairs.begin(), pairs.end(), [](const auto &x, const auto &y) { return x.second > y.second; });
        if (pairs.size() > size) pairs.resize(size);
        std::vector<std::string> ret;
        for (const auto &p:pairs) {
          ret.push_back(p.first);
        }
        return ret;
      }

    private:
      std::unordered_map<std::string, unsigned> counts;
    };

    template<class DATUM>
    std::pair<std::vector<DATUM>, std::vector<DATUM>>
    shuffle_and_split_dataset(const std::vector<DATUM> &dataset, float first_to_all_ratio = 0.9) {
      std::vector<DATUM> first(dataset);
      std::random_shuffle(first.begin(), first.end());
      unsigned first_size = (unsigned) (dataset.size() * first_to_all_ratio);
      std::vector<DATUM> second;
      std::move(first.begin() + first_size, first.end(), std::back_inserter(second));
      first.resize(first_size);
      return std::make_pair(std::move(first), std::move(second));
    }
  }
}


#endif //DYNET_WRAPPER_DY_UTILS_HPP
