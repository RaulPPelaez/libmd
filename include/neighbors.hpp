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

  // Checks if two positions are neighbors given a cutoff distance
  template <std::floating_point T>
  bool isNeighbor(const vec3<T>& pos_i, const vec3<T>& pos_j, T cutoff,
                  const Box<T>& box) {
    auto pos_diff = pos_i - pos_j;
    if (box.isPeriodic()) {
      pos_diff = apply_periodic_boundary_conditions(pos_diff, box);
    }
    const auto dist = sycl::length(pos_diff);
    return dist <= cutoff;
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
    sycl::buffer<int> neighbors(num_particles);
    sycl::buffer<int> neighbor_indices(num_particles * max_num_neighbors);
    sycl::buffer<int> too_many_neighbors(1);
    size_t local_size = std::min<size_t>(num_particles, 128);
    size_t num_tiles = (num_particles + local_size - 1) / local_size;
    auto execution_range = sycl::nd_range<1>{
      sycl::range<1>{num_tiles * local_size}, sycl::range<1>{local_size}};
    if(check_error){
      q.submit([&](sycl::handler& h){
	sycl::accessor too_many_neighbors_acc{too_many_neighbors, h, sycl::write_only, sycl::no_init};
	h.single_task<class initTooManyNeighbors>([=](){
	  too_many_neighbors_acc[0] = -1;
	});
      }
	);
    }
    auto event = q.submit([&](sycl::handler& h) {
      sycl::accessor positions_acc{positions, h, sycl::read_only};
      sycl::local_accessor<vec3<T>> shared{sycl::range<1>{local_size}, h};
      sycl::accessor neighbors_acc{neighbors, h, sycl::write_only, sycl::no_init};
      sycl::accessor neighbor_indices_acc{neighbor_indices, h, sycl::write_only, sycl::no_init};
      sycl::accessor too_many_neighbors_acc{too_many_neighbors, h,
					    sycl::write_only, sycl::no_init};
    h.parallel_for<class computeNeighbors>(
          execution_range, [=](sycl::nd_item<1> tid) {
            const size_t global_id = tid.get_global_id().get(0);
            const size_t local_id = tid.get_local_id().get(0);
            const bool is_active = global_id < num_particles;
            const vec3<T> pos_i =
                is_active ? positions_acc[global_id] : vec3<T>{0, 0, 0};
            int num_neighbors = 0;
            for (size_t offset = 0; offset < num_particles;
                 offset += local_size) {
              const size_t j_load = offset + local_id;
              if (j_load < num_particles)
                shared[local_id] = positions_acc[j_load];
              else
                shared[local_id] = vec3<T>{0, 0, 0};
              tid.barrier();
              for (size_t k = 0; k < local_size; k++) {
                const vec3<T> pos_j = shared[k];
                const bool is_neighbor = isNeighbor(pos_i, pos_j, cutoff, box);
                const bool is_same_particle = global_id == offset + k;
                const bool is_active_j = offset + k < num_particles;
                if (is_neighbor and is_active and is_active_j and
                    not is_same_particle) {
                  if (num_neighbors >= max_num_neighbors) {
                    auto atom =
                        sycl::atomic_ref<int, sycl::memory_order::relaxed,
                                         sycl::memory_scope::device>(
                            too_many_neighbors_acc[0]);
                    atom = global_id;
                  } else {
                    neighbor_indices_acc[global_id * max_num_neighbors +
                                         num_neighbors] = offset + k;
                  }
                  num_neighbors++;
                }
              }
              tid.barrier();
            }
            if (is_active) {
              neighbors_acc[global_id] = num_neighbors;
            }
          });
    });
    int too_many_neighbors2 = -1;
    if (check_error) {
      event.wait();
      sycl::host_accessor too_many_neighbors_acc{too_many_neighbors};
      too_many_neighbors2 = too_many_neighbors_acc[0];
    }
    usm_vector<int> neighbors2(num_particles);
    usm_vector<int> neighbor_indices2(num_particles * max_num_neighbors);
    sycl::host_accessor neighbors_acc{neighbors};
    std::copy(neighbors_acc.get_pointer(), neighbors_acc.get_pointer() + neighbors_acc.size(), neighbors2.begin());
    sycl::host_accessor neighbor_indices_acc{neighbor_indices};
    std::copy(neighbor_indices_acc.get_pointer(), neighbor_indices_acc.get_pointer() + neighbor_indices_acc.size(), neighbor_indices2.begin());
    return std::make_tuple(neighbors2, neighbor_indices2, too_many_neighbors2);
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
