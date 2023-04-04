#pragma once
#include "common.hpp"

namespace md {

  // Computes the distance between two positions, taking into account periodic
  // boundary conditions
  template <std::floating_point T>
  static inline auto particleDistance(const vec3<T>& pos_i, const vec3<T>& pos_j,
                                   const Box<T>& box) {
    auto pos_diff = pos_i - pos_j;
    if (box.isPeriodic()) {
      pos_diff = apply_periodic_boundary_conditions(pos_diff, box);
    }
    return pos_diff;
  }

  // Checks if two positions are neighbors given a cutoff distance
  template <std::floating_point T>
  static inline bool isNeighbor(const vec3<T>& pos_i, const vec3<T>& pos_j,
                                T cutoff, const Box<T>& box) {
    auto pos_diff = particleDistance(pos_i, pos_j, box);
    const auto dist = sycl::length(pos_diff);
    return dist <= cutoff;
  }


} // namespace md
