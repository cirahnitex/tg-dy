//
// Created by YAN Yuchen on 11/7/2018.
//

#ifndef DYNET_WRAPPER_DY_SERIALIZATION_HELPER_HPP
#define DYNET_WRAPPER_DY_SERIALIZATION_HELPER_HPP
#include <dynet/dynet.h>
#include "dy_common.hpp"
#include <dynet/dict.h>
#include <cereal/archives/binary.hpp>
#include <cereal/types/vector.hpp>
#include <cereal/types/unordered_set.hpp>
#include <cereal/types/string.hpp>

#define EASY_SERIALIZABLE(...) template<class Archive> \
void serialize(Archive & archive) \
{archive( __VA_ARGS__);} \

namespace dynet {

  template<class Archive>
  void save(Archive & archive, Dim const & dim)
  {
    std::vector<long> dim_vec;
    for(unsigned i=0; i<dim.ndims(); i++) {
      dim_vec.push_back(dim[i]);
    }
    archive(dim_vec);

    archive(dim.batch_elems());
  }

  template<class Archive>
  void load(Archive & archive, Dim & dim)
  {
    std::vector<long> dim_vec;
    archive(dim_vec);

    unsigned num_batches;
    archive(num_batches);

    dim = dynet::Dim(dim_vec, num_batches);
  }

  template<class Archive>
  void save(Archive & archive, Parameter const & p)
  {
    bool isValid = (bool)p.p;
    archive(isValid);
    if(!isValid) return;

    archive(p.dim());
    archive(dynet::as_vector(p.get_storage().values));
  }

  template<class Archive>
  void load(Archive & archive, Parameter & p)
  {
    bool isValid;
    archive(isValid);
    if(!isValid) return;

    dynet::Dim dim;
    archive(dim);
    std::vector<dynet::real> values;
    archive(values);
    if(!p.p) p = tg::dy::_pc().add_parameters(dim);
    if(p.dim() != dim) p = tg::dy::_pc().add_parameters(dim);
    p.set_value(values);
  }

  template<class Archive>
  void save(Archive& archive, LookupParameter const & p) {
    bool isValid = (bool)p.p;
    archive(isValid);
    if(!isValid) return;

    const auto& values = p.get_storage().values;
    archive((unsigned)values.size());
    archive(p.dim());
    for(const auto& value:values) {
      archive(dynet::as_vector(value));
    }
  }

  template<class Archive>
  void load(Archive& archive, LookupParameter& p) {
    bool isValid;
    archive(isValid);
    if(!isValid) return;

    unsigned size;
    archive(size);
    Dim dim;
    archive(dim);
    p = tg::dy::_pc().add_lookup_parameters(size, dim);
    for(unsigned i=0; i<size; i++) {
      std::vector<dynet::real> value;
      archive(value);
      p.initialize(i,value);
    }
  }

}


#endif //DYNET_WRAPPER_DY_SERIALIZATION_HELPER_HPP
