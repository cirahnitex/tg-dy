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
    using listener_handle_t = std::shared_ptr<listener_t>;
  private:
    std::unordered_set<listener_handle_t> listeners;
    std::unordered_set<listener_handle_t> once_listeners;
    const listener_handle_t& add_listener(const listener_handle_t& listener) {
      listeners.insert(listener);
      return listener;
    }
    const listener_handle_t& add_once_listener(const listener_handle_t& listener) {
      once_listeners.insert(listener);
      return listener;
    }
  public:
    listener_handle_t add_listener(const listener_t& listener) {
      return add_listener(std::make_shared<listener_t>(listener));
    }

    listener_handle_t add_once_listener(const listener_t& listener) {
      return add_once_listener(std::make_shared<listener_t>(listener));
    }

    void remove_listener(const listener_handle_t& listener) {
      listeners.erase(listener);
      once_listeners.erase(listener);
    }
    void fire(Args ...args) {

      // call listeners
      {
        // make a copy so that removing listeners within listener is safe
        std::unordered_set<listener_handle_t> listeners_cp(listeners);
        for(auto&& listener:listeners_cp) {
          (*listener)(args...);
        }
      }

      // call once listeners
      {
        std::unordered_set<listener_handle_t> listeners_cp(once_listeners);
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
