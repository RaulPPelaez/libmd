#pragma once
#include <CL/sycl.hpp>
#include <concepts>

namespace md {
  using namespace cl;
  using namespace sycl::access;

  using arithmetic_type = float;
  using scalar = float;
  template <class T> using vec3 = sycl::vec<T, 3>;

  /**
   * @brief Returns the default queue used by libMD
   *
   */
  static inline sycl::queue get_default_queue() {
    static sycl::default_selector selector;
    static sycl::queue q{selector};
    return q;
  }

  /**
   * @brief A simple 3D box. Can be orthorhombic, cubic or triclinic
   * The member size is a 3x3 matrix, where the first row is the x vector, the
   * second row is the y vector and the third row is the z vector of the box
   * @tparam T The floating point type of the box vectors
   *
   */
  template <std::floating_point T> struct Box {
    sycl::marray<vec3<T>, 3> size{}; // box vectors

    constexpr Box() : size() {}

    // Triclinic box
    Box(sycl::marray<vec3<T>, 3> size) : size(size) {}

    // Orthorhombic box
    Box(vec3<T> orth_size) {
      size[0] = {orth_size[0], 0, 0};
      size[1] = {0, orth_size[1], 0};
      size[2] = {0, 0, orth_size[2]};
    }

    // Cubic box
    Box(T orth_size): Box(vec3<T>{orth_size, orth_size, orth_size}) {}


  };

  template <class T> static const Box<T> empty_box = Box<T>();

  /**
   * @brief Applies periodic boundary conditions to a distance vector
   *
   * @tparam T The floating point type of the distance vector
   * @param distance The distance vector
   * @param box The box to apply the boundary conditions to. Can be orthorhombic, cubic or triclinic
   * @return vec3<T> The distance vector with the boundary conditions applied
   */
  template <std::floating_point T>
  inline auto apply_periodic_boundary_conditions(vec3<T> distance, Box<T> box) {
    vec3<T> result = distance;
    const T scale3 = sycl::round(distance[2] / box.size[2][2]);
    result[0] -= scale3 * box.size[2][0];
    result[1] -= scale3 * box.size[2][1];
    result[2] -= scale3 * box.size[2][2];
    const T scale2 = sycl::round(distance[1] / box.size[1][1]);
    result[0] -= scale2 * box.size[1][0];
    result[1] -= scale2 * box.size[1][1];
    const T scale1 = sycl::round(distance[0] / box.size[0][0]);
    result[0] -= scale1 * box.size[0][0];
    return result;
  }

} // namespace md
