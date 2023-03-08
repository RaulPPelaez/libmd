#pragma once
#include<CL/sycl.hpp>
#include<concepts>

namespace md{
  using namespace cl;
  using namespace sycl::access;

  using arithmetic_type = float;
  using scalar = float;
  template<class T> using vec3 = sycl::vec<T, 3>;

  static inline sycl::queue get_default_qeue(){
    static sycl::default_selector selector;
    static sycl::queue q{selector};
    return q;
  }


  template<std::floating_point T>
  struct Box{
    sycl::marray<vec3<T>,3> size{}; //box vectors

    //Orthorhombic box
    Box(vec3<T> orth_size){
      size[0] = {orth_size[0], 0, 0};
      size[1] = {0, orth_size[1], 0};
      size[2] = {0, 0, orth_size[2]};
    }

    //Cubic box
    Box(T orth_size){
      size[0] = {orth_size, 0, 0};
      size[1] = {0, orth_size, 0};
      size[2] = {0, 0, orth_size};
    }

    //Triclinic box
    Box(sycl::marray<vec3<T>,3> size) : size(size){}
    constexpr Box() : size(){}
  };

  template<class T> static const Box<T> empty_box = Box<T>();

  //This function is used to apply periodic boundary conditions to a position
  //Takes into account wether the box is triclinic or not
  template<std::floating_point T>
  inline auto apply_periodic_boundary_conditions(vec3<T> distance, Box<T> box){
    vec3<T> result;
    const T scale3 = sycl::floor(distance[2] / box.size[2][2] + T(0.5));
    result[0] = distance[0] - scale3 * box.size[2][0];
    result[1] = distance[1] - scale3 * box.size[2][1];
    result[2] = distance[2] - scale3 * box.size[2][2];
    const T scale2 = sycl::floor(distance[1] / box.size[1][1] + T(0.5));
    result[0] -= scale2 * box.size[1][0];
    result[1] -= scale2 * box.size[1][1];
    const T scale1 = sycl::round(distance[0] / box.size[0][0] + T(0.5));
    result[0] -= scale1 * box.size[0][0];
    return result;
  }
}
