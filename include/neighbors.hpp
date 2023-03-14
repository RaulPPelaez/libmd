/*Raul P. Pelaez 2023. Neighbor list computation.
  This file contains functions to compute the neighbor of a group of 3D points.
 */
#pragma once
#include "common.hpp"
#include "queue.hpp"
#include "log.hpp"
#include "allocator.hpp"
#include <limits>
namespace md {

  /**
   * @brief Compute the neighbors of each particle
   *
   * @param q The SYCL queue to use
   * @param positions The positions of the particles with shape (num_particles,
   * 3)
   * @param cutoff The cutoff distance.
   * @param box The box to use. If the box is empty, the system is assumed to be
   * non-periodic
   * @param check_error If true, the function will wait for the kernel and
   * return an error flag
   * @param max_num_neighbors The maximum number of neighbors allowed per
   * particle
   * @return std::tuple<sycl::buffer<int>, sycl::buffer<int>, int> The
   * number of neighbors, the neighbor indices, and an error flag. If the
   * maximum number of neighbors is not enough and check_errors is true , the
   * function will return the index of the particle with too many nighbors in
   * the error flag. The format of the neighbor indices is as follows:
   * neighbor_indices[i * max_num_neighbors + j] is the index of the jth
   * neighbor of particle i Contents of neighbor_indices beyond the number of
   * neighbors for a particle are undefined.
   */
  template <std::floating_point T>
  auto tryComputeNeighbors(sycl::buffer<vec3<T>>& positions,
                           T cutoff = std::numeric_limits<T>::infinity(),
                           Box<T> box = empty_box<T>,
                           int max_num_neighbors = 32,
                           bool check_error = false) {
    // This function starts with a  max_num_neighbors of 32 by default,
    // and if it finds a particle  with more than 32 neighbors, it will
    // increase the  max_num_neighbors and recompute the  neighbors.
    auto q = get_default_queue();
    const auto num_particles = positions.get_count();
    usm_vector<int> neighbors(num_particles);
    auto* neighbors_acc = neighbors.data();
    usm_vector<int> neighbor_indices(num_particles * max_num_neighbors);
    auto* neighbor_indices_acc = neighbor_indices.data();
    usm_vector<int> too_many_neighbors(1);
    auto* too_many_neighbors_acc = too_many_neighbors.data();
    if (check_error) {
      too_many_neighbors[0] = -1;
    }
    bool isPeriodic = box.isPeriodic();
    auto event = q.submit([&](sycl::handler& h) {
      sycl::accessor positions_acc{positions, h, sycl::read_only};
      h.parallel_for<class computeNeighbors>(
          sycl::range<1>(num_particles), [=](sycl::item<1> item) {
            const auto i = item.get_id(0);
            const auto pos_i = positions_acc[i];
            int num_neighbors = 0;
            for (int j = 0; j < num_particles; j++) {
              if (i == j) {
                continue;
              }
              const auto pos_j = positions_acc[j];
              auto pos_diff = pos_i - pos_j;
              if (isPeriodic) {
                pos_diff = apply_periodic_boundary_conditions(pos_diff, box);
              }
              const auto dist = sycl::length(pos_diff);
              if (dist <= cutoff) {
                if (num_neighbors >= max_num_neighbors) {
		  auto atom =
		    sycl::atomic_ref<int, sycl::memory_order::relaxed,
				     sycl::memory_scope::device>(
								 too_many_neighbors_acc[0]);
		  atom = i;
                  break;
                }
                neighbor_indices_acc[i * max_num_neighbors + num_neighbors] = j;
                num_neighbors++;
              }
            }
            neighbors_acc[i] = num_neighbors;
          });
    });
    int too_many_neighbors2 = -1;
    if (check_error) {
      event.wait();
      too_many_neighbors2 = too_many_neighbors[0];
    }
    return std::make_tuple(neighbors, neighbor_indices, too_many_neighbors2);
  }

  /**
   * @brief Compute the neighbors of each particle
   *
   * @param q The SYCL queue to use
   * @param positions The positions of the particles with shape (num_particles,
   * 3)
   * @param cutoff The cutoff distance.
   * @param box The box to use. If the box is empty, the system is assumed to be
   * non-periodic
   * @param max_num_neighbors The maximum number of neighbors allowed per
   * particle. This will be increased if necessary to fit all necessary
   * neighbors if resize_to_fit is true.
   * @param resize_to_fit If true, the neighbor indices buffer will be resized
   * until are neighbors fit.
   * @return std::tuple<sycl::buffer<int>, sycl::buffer<int>, int> The number of
   * neighbors, the neighbor indices and the maximum number of neighbors used.
   * The format of the neighbor indices is as follows:
   * neighbor_indices[i * max_num_neighbors + j] is the index of the jth
   * neighbor of particle i Contents of neighbor_indices beyond the number of
   * neighbors for a particle are undefined.
   */
  template <std::floating_point T>
  auto computeNeighbors(sycl::buffer<vec3<T>>& positions,
                        T cutoff = std::numeric_limits<T>::infinity(),
                        Box<T> box = empty_box<T>, int max_num_neighbors = 32,
                        bool resize_to_fit = false) {
    // This function starts with a  max_num_neighbors of 32 by default,
    // and if it finds a particle  with more than 32 neighbors, it will
    // increase the  max_num_neighbors and recompute the  neighbors.
    int num_particles = positions.get_count();
    do {
      auto [neighbors, neighbor_indices, errorFlag] = tryComputeNeighbors(
          positions, cutoff, box, max_num_neighbors, resize_to_fit);
      if (resize_to_fit) {
        int too_many_neighbors = errorFlag;
        if (too_many_neighbors != -1) {
          if (max_num_neighbors > num_particles) {
            throw std::runtime_error("Too many neighbors for particle " +
                                     std::to_string(too_many_neighbors) +
                                     ". This should not had happened.");
          }
          // If  we  found  a  particle with  more  than  max_num_neighbors
          // neighbors, increase max_num_neighbors and recompute
          max_num_neighbors += 32;
          log<level::DEBUG4>("Trying max_num_neighbors = " +
                             std::to_string(max_num_neighbors));
          continue;
        }
      }
      return std::make_tuple(neighbors, neighbor_indices, max_num_neighbors);
    } while (true);
  }
} // namespace md
