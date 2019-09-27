//
// Created by Dekai WU and YAN Yuchen on 20190927.
//

#ifndef DYANA_CPP_DYANA_GUARD_MACRO_HPP
#define DYANA_CPP_DYANA_GUARD_MACRO_HPP

#define DEFINE_THREAD_LOCAL_GUARAD(guard_name) \
class guard_name { \
  static unsigned& num_instances() { \
    thread_local static unsigned _; \
    return _; \
  }; \
public: \
  static bool is_guarded() { \
    return num_instances() > 0; \
  } \
  guard_name() {num_instances()++;} \
  guard_name(const guard_name&) = delete; \
  guard_name(guard_name&&) noexcept = delete; \
  guard_name &operator=(const guard_name&) = delete; \
  guard_name &operator=(guard_name&&) noexcept = delete; \
  ~guard_name() {num_instances()--;} \
}; \



#endif //DYANA_CPP_DYANA_GUARD_MACRO_HPP
