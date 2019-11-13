//
// Created by Dekai WU and YAN Yuchen on 20191113.
//

#ifndef DYANA_CPP_DYANA_RANDOM_HPP
#define DYANA_CPP_DYANA_RANDOM_HPP

#endif //DYANA_CPP_DYANA_RANDOM_HPP
#include <random>
#include <dynet/dynet.h>
namespace dyana {
  namespace random {
    inline float uniform_real(float min, float max){
      std::uniform_real_distribution<float> distribution(min, max);
      return distribution(*dynet::rndeng);
    }

    inline int uniform_int(int min, int max) {
      std::uniform_int_distribution<int> distribution(min,max);
      return distribution(*dynet::rndeng);
    }

    inline bool float_equal(float A, float B, float epsilon = 0.001)
    {
      return (fabs(A - B) < epsilon);
    }

    inline unsigned uniform_int_0n(unsigned n) {
      if(n == 0) throw std::runtime_error("uniform_int_0n: n must > 0");
      return (unsigned)uniform_int(0, n - 1);
    }

    inline std::vector<unsigned> weighted_categories(const std::vector<float>& probs_for_each_side, unsigned num_rolls) {
      using namespace std;
      vector<float> prefix_probs;
      {
        float cumulated_prob = 0;
        for(auto&& prob:probs_for_each_side) {
          cumulated_prob += prob;
          prefix_probs.push_back(cumulated_prob);
        }

        if(!float_equal(cumulated_prob, 1.0)) {
          throw std::runtime_error("dice roll: probabilities must sum up to 1");
        }
      }

      vector<unsigned> ret;
      for(unsigned _=0; _<num_rolls; _++) {
        float rand_num = uniform_real(0, 1);

        bool pushed = false;
        for(unsigned i=0; i<prefix_probs.size(); i++) {
          if(rand_num < prefix_probs[i]) {
            ret.push_back(i);
            pushed = true;
            continue;
          }
        }
        if(!pushed) ret.push_back(0);
      }
      return ret;
    }

    inline unsigned weighted_category(const std::vector<float>& probs_for_each_side) {
      return weighted_categories(probs_for_each_side, 1).front();
    }

    inline bool weighted_coinflip(float success_chance) {
      return weighted_category({(float)1 - success_chance, success_chance});
    }

    inline std::vector<bool> weighted_coinflips(float success_chance, unsigned num_flips) {
      auto categories = weighted_categories({(float)1 - success_chance, success_chance}, num_flips);
      std::vector<bool> ret;
      ret.reserve(num_flips);
      for(auto&& c:categories) {
        ret.emplace_back(c);
      }
      return ret;
    }

    template<typename T>
    inline const T& choose(const std::vector<T>& range) {
      return range.at(uniform_int_0n(range.size()));
    }
  }
}
