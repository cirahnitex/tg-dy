//
// Created by Dekai WU and YAN Yuchen on 20190825.
//

#ifndef DYANA_CPP_DYANA_EVENT_EMITTER_HPP
#define DYANA_CPP_DYANA_EVENT_EMITTER_HPP
#include <functional>
#include <memory>
#include <unordered_set>
namespace dyana {
  template<typename ...Args>
  class event_emitter {
  public:
    using listener_t = std::function<void(Args...)>;
    using listener_ptr = std::shared_ptr<listener_t>;
  private:
    std::unordered_set<listener_ptr> listeners;
    std::unordered_set<listener_ptr> once_listeners;
  public:
    void add_listener(const listener_ptr& listener) {
      listeners.insert(listener);
    }
    void add_once_listener(const listener_ptr& listener) {
      once_listeners.insert(listener);
    }
    void remove_listener(const listener_ptr& listener) {
      listeners.erase(listener);
      once_listeners.erase(listener);
    }
    void fire(Args ...args) {

      // call listeners
      {
        // make a copy so that removing listeners within listener is safe
        std::unordered_set<listener_ptr> listeners_cp(listeners);
        for(auto&& listener:listeners_cp) {
          (*listener)(args...);
        }
      }

      // call once listeners
      {
        std::unordered_set<listener_ptr> listeners_cp(once_listeners);
        for(auto&& listener:listeners_cp) {
          (*listener)(args...);
        }
      }

      // clear once listeners
      once_listeners.clear();
    }
  };
}

#endif //DYANA_CPP_DYANA_EVENT_EMITTER_HPP
