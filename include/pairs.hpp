/*Raul P. Pelaez 2023. Neighbor list computation.
  This file contains functions to compute the neighbor of a group of 3D points.
  As opposed to the neighbor list in the neighbor_list.hpp file, this one computes a compacted list of pairs.
 */
#pragma once
#include "common.hpp"
#include "utils.hpp"
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
   * @param max_num_pairs The maximum number of pairs. If the number of pairs is larger than max_num_pairs, the function will throw.
   * @return std::tuple<sycl::buffer<int>, sycl::buffer<vec3<T>>, sycl::buffer<T>, sycl::buffer<int>>:
              - A list of pairs (i,j), shape (max_num_pairs, 2)
	      - A list of vectors between particles (r_ij), shape (max_num_pairs, 3)
	      - A list with the norms of the distances (|r_ij|), shape (max_num_pairs)
	      - The number of pairs found. Shape (1). Can be larger than max_num_pairs.
   * Note that the contents of the list after the number of pairs is undefined.
   *
   */
  template <std::floating_point T>
  auto tryComputeNeighborPairs(sycl::buffer<vec3<T>>& positions,
			       T cutoff = std::numeric_limits<T>::infinity(),
			       Box<T> box = empty_box<T>,
			       int max_num_pairs = 32,
			       bool check_error = false) {
    auto q = get_default_queue();
    const auto num_particles = positions.get_count();
    usm_vector<int> neighbor_pairs(2*max_num_pairs);
    auto* neighbor_pairs_acc = neighbor_pairs.data();
    usm_vector<vec3<T>> neighbor_vectors(max_num_pairs);
    auto* neighbor_vectors_acc = neighbor_vectors.data();
    usm_vector<T> neighbor_distances(max_num_pairs);
    auto* neighbor_distances_acc = neighbor_distances.data();
    usm_vector<int> num_pairs(1);
    auto* num_pairs_acc = num_pairs.data();
    q.fill(num_pairs_acc, 0, 1);
    size_t local_size = std::min<size_t>(num_particles, 128);
    size_t num_tiles = (num_particles + local_size - 1) / local_size;
    auto execution_range = sycl::range<1>{num_particles*(num_particles-1)/2};
    auto event = q.submit([&](sycl::handler& h) {
      sycl::accessor positions_acc{positions, h, sycl::read_only};
      h.parallel_for<class computeNeighbors>(
          execution_range, [=](sycl::item<1> tid) {
            const size_t index = tid.get_linear_id();
	    int32_t row = floor((sqrtf(8 * index + 1) + 1) / 2);
	    if (row * (row - 1) > 2 * index) row--;
	    const int32_t column = index - row * (row - 1) / 2;
	    auto pos_diff = particleDistance(positions_acc[row], positions_acc[column], box);
	    const auto dist = sycl::length(pos_diff);
	    const bool isNeighbor = dist <= cutoff;
	    if(isNeighbor){
	      auto num_pair_atom =
		sycl::atomic_ref<int, sycl::memory_order::relaxed,
				 sycl::memory_scope::device>(
							     num_pairs[0]);
	      const auto npairs = num_pair_atom++;
	      if(npairs < max_num_pairs){
		neighbor_pairs_acc[2*npairs] = row;
		neighbor_pairs_acc[2*npairs+1] = column;
		neighbor_vectors_acc[npairs] = pos_diff;
		neighbor_distances_acc[npairs] = dist;
	      }
	    }
          });
    });
    if(check_error){
      event.wait_and_throw();
      if(num_pairs[0] > max_num_pairs){
	throw std::runtime_error("Too many pairs found, expected at most "+std::to_string(max_num_pairs)+", found "+std::to_string(num_pairs[0]));
      }
    }
    return std::make_tuple(neighbor_pairs, neighbor_vectors, neighbor_distances, num_pairs);
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
   * @param max_num_pairs The maximum number of pairs.
   * @param resize_to_fit If true, the neighbor indices buffer will be resized
   * until are neighbors fit.
   * @return tuple(sycl::buffer<int>, sycl::buffer<vec3<T>>, sycl::buffer<T>, sycl::buffer<int>):
   * - A list of neighbors for each particle, shape (num_particles,
   * max_num_neighbors)
   * - A list of vectors between particles (r_ij), shape (num_particles,
   * max_num_neighbors, 3)
   * - A list with the norms of the distances (|r_ij|), shape (num_particles,
   * max_num_neighbors)
   * - The total number of pairs found. Shape (1).
   */
  template <std::floating_point T>
  auto computeNeighborPairs(sycl::buffer<vec3<T>>& positions,
                        T cutoff = std::numeric_limits<T>::infinity(),
                        Box<T> box = empty_box<T>, int max_num_pairs = 32,
                        bool resize_to_fit = false) {
    // This function starts with a  max_num_neighbors of 32 by default,
    // and if it finds a particle  with more than 32 neighbors, it will
    // increase the  max_num_neighbors and recompute the  neighbors.
    int num_particles = positions.get_count();
    do {
      auto [neighbors, deltas, distances, num_pairs] = tryComputeNeighborPairs(positions, cutoff, box, max_num_pairs);
      if (resize_to_fit) {
	auto q = get_default_queue();
	q.wait_and_throw();
        if (num_pairs[0] > max_num_pairs) {
          // If  we  found  a  particle with  more  than  max_num_neighbors
          // neighbors, increase max_num_neighbors and recompute
	  max_num_pairs = ((num_pairs[0] + 31) / 32) * 32;
          log<level::DEBUG4>("Trying max_num_neighbors = " +
                             std::to_string(max_num_pairs));
          continue;
        }
      }
      return std::make_tuple(neighbors, deltas, distances, num_pairs);
    } while (true);
  }
} // namespace md
