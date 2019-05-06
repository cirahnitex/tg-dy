#include <iostream>
#include "dyana.hpp"
#include <chrono>
using namespace tg;
using namespace std;

int main() {
  dyana::initialize(1, dyana::trainer_type::SIMPLE_SGD);

  return 0;
}
