//
// Created by Dekai WU and YAN Yuchen on 20190906.
//

#ifndef DYANA_CPP_DYANA_TIMER_HPP
#define DYANA_CPP_DYANA_TIMER_HPP
#include <functional>
#include <chrono>
namespace dyana {
  inline unsigned long milliseconds_elapsed(const std::function<void()>& task) {
    using namespace std;
    auto start = chrono::steady_clock::now();
    task();
    auto end = chrono::steady_clock::now();
    return chrono::duration_cast<chrono::milliseconds>(end - start).count();
  }
}

#endif //DYANA_CPP_DYANA_TIMER_HPP
