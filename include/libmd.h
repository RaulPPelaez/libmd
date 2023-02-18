#pragma once

#include <CL/sycl.hpp>
namespace md{
  using namespace cl;

  using arithmetic_type = float;
  using vector_type = sycl::vec<arithmetic_type, 3>;

}
