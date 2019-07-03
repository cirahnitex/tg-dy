#include <iostream>
#include "dyana.hpp"
#include <xml_archive.hpp>
#include <sstream>
#include <vector>
using namespace std;


int main() {
  dyana::initialize();

  dyana::parameter x({3});

  std::stringstream ss;
  {
    cereal::BinaryOutputArchive oa(ss);
    oa << x;
  }
  {
    dyana::parameter x;
    cereal::BinaryInputArchive ia(ss);
    ia >> x;
  }

  return 0;
}
