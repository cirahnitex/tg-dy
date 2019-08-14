//
// Created by YAN Yuchen on 11/7/2018.
//

#ifndef DYANA_SERIALIZATION_HELPER_HPP
#define DYANA_SERIALIZATION_HELPER_HPP
#include <dynet/dynet.h>
#include "dyana_common.hpp"
#include <dynet/dict.h>
#include <cereal/archives/binary.hpp>
#include <cereal/types/vector.hpp>
#include <cereal/types/unordered_set.hpp>
#include <cereal/types/string.hpp>
#include "FOR_EACH.cmacros.hpp"

#define TRIVIAL_SERIALIZABLE template<class Archive> \
void serialize(Archive & archive) {} \

#define CALL_ARCHIVE_NVP(name) archive(cereal::make_nvp(#name, name));

#define EASY_SERIALIZABLE(...) template<class Archive> \
void serialize(Archive & archive) \
{FOR_EACH(CALL_ARCHIVE_NVP, __VA_ARGS__)} \

#define INHERITANCE_TRIVIAL_SERIALIZABLE(BaseClass) template<class Archive> \
void serialize(Archive & archive) \
{archive(cereal::base_class< BaseClass >(this));} \

#define INHERITANCE_EASY_SERIALIZABLE(BaseClass, ...) template<class Archive> \
void serialize(Archive & archive) \
{archive(cereal::base_class< BaseClass >(this)); FOR_EACH(CALL_ARCHIVE_NVP, __VA_ARGS__)} \

namespace dynet {

  template<class Archive>
  void save(Archive & archive, Dim const & dim)
  {
    std::vector<long> ds;
    for(unsigned i=0; i<dim.ndims(); i++) {
      ds.push_back(dim[i]);
    }
    archive(cereal::make_nvp("ds", ds));

    archive(cereal::make_nvp("batch", dim.batch_elems()));
  }

  template<class Archive>
  void load(Archive & archive, Dim & dim)
  {
    std::vector<long> ds;
    archive(cereal::make_nvp("ds", ds));

    unsigned num_batches;
    archive(cereal::make_nvp("batch", num_batches));

    dim = dynet::Dim(ds, num_batches);
  }

  template<class Archive>
  void save(Archive & archive, Parameter const & p)
  {
    using namespace std;
    bool isValid = (bool)p.p;
    archive(cereal::make_nvp("valid", isValid));
    if(!isValid) return;
    archive(cereal::make_nvp("dim",p.dim()));
    archive(cereal::make_nvp("data", dynet::as_vector(p.get_storage().values)));
  }

  template<class Archive>
  void load(Archive & archive, Parameter & p)
  {
    using namespace std;
    bool isValid;
    archive(cereal::make_nvp("valid", isValid));
    if(!isValid) return;
    dynet::Dim dim;
    archive(cereal::make_nvp("dim", dim));
    std::vector<dynet::real> values;
    archive(cereal::make_nvp("data", values));
    if(!p.p) p = dyana::_pc()->add_parameters(dim);
    if(p.dim() != dim) p = dyana::_pc()->add_parameters(dim);
    p.set_value(values);
  }

  template<class Archive>
  void save(Archive& archive, LookupParameter const & p) {
    bool isValid = (bool)p.p;
    archive(cereal::make_nvp("valid", isValid));
    if(!isValid) return;

    const auto& values = p.get_storage().values;
    archive(cereal::make_nvp("size", (unsigned)values.size()));
    archive(cereal::make_nvp("dim",p.dim()));
    for(const auto& value:values) {
      archive(cereal::make_nvp("data", dynet::as_vector(value)));
    }
  }

  template<class Archive>
  void load(Archive& archive, LookupParameter& p) {
    bool isValid;
    archive(cereal::make_nvp("valid", isValid));
    if(!isValid) return;

    unsigned size;
    archive(cereal::make_nvp("size", size));
    Dim dim;
    archive(cereal::make_nvp("dim", dim));
    p = dyana::_pc()->add_lookup_parameters(size, dim);
    for(unsigned i=0; i<size; i++) {
      std::vector<dynet::real> value;
      archive(cereal::make_nvp("data", value));
      p.initialize(i,value);
    }
  }

}


#endif //DYANA_SERIALIZATION_HELPER_HPP
