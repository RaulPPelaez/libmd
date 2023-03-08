#pragma once

#include "common.hpp"
#include "log.hpp"
#include <concepts>

namespace md {


  /**
   * @brief Compute the neighbors of each particle
   *
   * @param q The SYCL queue to use
   * @param positions The positions of the particles with shape (num_particles, 3)
   * @param cutoff The cutoff distance
   * @param box The box to use. If the box is empty, the system is assumed to be non-periodic
   * @param max_num_neighbors The maximum number of neighbors allowed per particle
   * @return std::tuple<sycl::buffer<int>, sycl::buffer<int>, int, int> The number of
   * neighbors, the neighbor indices, the maximum number of neighbors used and an error flag.
   * If the maximum number of neighbors is not enough, the function will return the index of the particle with too many nighbors in the error flag.
   * The format of the neighbor indices is as follows:
   * neighbor_indices[i * max_num_neighbors + j] is the index of the jth neighbor of particle i
   * Contents of neighbor_indices beyond the number of neighbors for a particle are undefined.
   */
  template <std::floating_point T>
  auto tryComputeNeighbors(sycl::queue& q, sycl::buffer<vec3<T>>& positions,
                           T cutoff, Box<T> box = empty_box<T>,
                           int max_num_neighbors = 32) {
    // This function starts with a  max_num_neighbors of 32 by default,
    // and if it finds a particle  with more than 32 neighbors, it will
    // increase the  max_num_neighbors and recompute the  neighbors.
    auto num_particles = positions.get_count();
    auto neighbors = sycl::buffer<int>(num_particles);
    auto neighbor_indices =
        sycl::buffer<int>(num_particles * max_num_neighbors);
    int too_many_neighbors = -1;
    auto too_many_neighbors_atomic =
      sycl::atomic_ref<int, sycl::memory_order::relaxed,
		       sycl::memory_scope::device,
		       sycl::access::address_space::global_space>(
								  too_many_neighbors);

    bool isPeriodic = false;
    for (int i = 0; i < 3; i++) {
      if (box.size[i][i] > 0)
        isPeriodic = true;
    }
    if(isPeriodic)
      log<level::DEBUG7>("Using periodic boundary conditions");

    q.submit([&](sycl::handler& h) {
      sycl::accessor positions_acc{positions, h, sycl::read_only};
      sycl::accessor neighbors_acc{neighbors, h, sycl::write_only,
                                   sycl::no_init};
      sycl::accessor neighbor_indices_acc{neighbor_indices, h, sycl::write_only,
                                          sycl::no_init};

      h.parallel_for<class computeNeighbors>(
          sycl::range<1>(num_particles), [=](sycl::item<1> item) {
            auto i = item.get_id(0);
            auto pos_i = positions_acc[i];
            int num_neighbors = 0;
            for (int j = 0; j < num_particles; j++) {
              if (i == j)
                continue;
              auto pos_j = positions_acc[j];
              auto pos_diff = pos_i - pos_j;
              if (isPeriodic) {
                pos_diff = apply_periodic_boundary_conditions(pos_diff, box);
              }
	      auto dist = sycl::length(pos_diff);
              if (dist <= cutoff) {
                if (num_neighbors >= max_num_neighbors) {
                  too_many_neighbors_atomic = i;
                  break;
                }
                neighbor_indices_acc[i * max_num_neighbors + num_neighbors] = j;
                num_neighbors++;
              }
            }
            neighbors_acc[i] = num_neighbors;
          });
    });
    q.wait();
    return std::make_tuple(neighbors, neighbor_indices, max_num_neighbors,
                           too_many_neighbors);
  }

  /**
   * @brief Compute the neighbors of each particle
   *
   * @param q The SYCL queue to use
   * @param positions The positions of the particles with shape (num_particles, 3)
   * @param cutoff The cutoff distance
   * @param box The box to use. If the box is empty, the system is assumed to be non-periodic
   * @param max_num_neighbors The maximum number of neighbors allowed per particle. This will be increased if necessary to fit all necessary neighbors.
   * @return std::tuple<sycl::buffer<int>, sycl::buffer<int>, int> The number of
   * neighbors, the neighbor indices and the maximum number of neighbors used.
   * The format of the neighbor indices is as follows:
   * neighbor_indices[i * max_num_neighbors + j] is the index of the jth neighbor of particle i
   * Contents of neighbor_indices beyond the number of neighbors for a particle are undefined.
   */
  template <std::floating_point T>
  auto computeNeighbors(sycl::queue& q, sycl::buffer<vec3<T>>& positions,
                        T cutoff, Box<T> box = empty_box<T>,
                        int max_num_neighbors = 32) {
    // This function starts with a  max_num_neighbors of 32 by default,
    // and if it finds a particle  with more than 32 neighbors, it will
    // increase the  max_num_neighbors and recompute the  neighbors.
    int num_particles = positions.get_count();
    int errorFlag;
    do {
      auto [neighbors, neighbor_indices, ignored, errorFlag] =
          tryComputeNeighbors(q, positions, cutoff, box, max_num_neighbors);
      if (errorFlag == -1) {
        return std::make_tuple(neighbors, neighbor_indices, max_num_neighbors);
      }
      if (max_num_neighbors > num_particles) {
	if(errorFlag == -1)
	  throw std::runtime_error("Error flag is -1, but max_num_neighbors > num_particles. This should not happen.");
        int particle_with_too_many_neighbors = errorFlag;
        sycl::host_accessor neighbor_acc(neighbors, sycl::read_only);
        int number_of_neighbors =
            neighbor_acc[particle_with_too_many_neighbors];
        throw std::runtime_error(
            "Too many neighbors, particle i: " +
            std::to_string(particle_with_too_many_neighbors) + " has " +
            std::to_string(number_of_neighbors) + " neighbors");
      }
      // If  we  found  a  particle with  more  than  max_num_neighbors
      // neighbors, increase max_num_neighbors and recompute
      max_num_neighbors += 32;
      log<level::DEBUG4>("Trying max_num_neighbors = " + std::to_string(max_num_neighbors));
    } while (true);
  }
} // namespace md
