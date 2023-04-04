Neighbors
----------

This page describes two functions for computing the neighbor pairs of a group of 3D points, :cpp:any:`tryComputeNeighborPairs` and :cpp:any:`computeNeighborPairs`. These functions are declared in the header file pairs.hpp.
It generates a compacted list of pairs, as opposed to the neighbor list in the neighbor_list.hpp file.

.. _compute-neighbor-pairs-try:

tryComputeNeighborPairs
=======================

.. cpp:function:: template <std::floating_point T> auto tryComputeNeighborPairs(sycl::buffer<vec3<T>>& positions, T cutoff = std::numeric_limits<T>::infinity(), Box<T> box = empty_box<T>, int max_num_pairs = 32, bool check_error = false)

   Note that the contents of the list after the reported number of pairs is undefined.
  :param positions: The positions of the particles with shape (num_particles, 3).
  :param cutoff: The cutoff distance. Default is infinity.
  :param box: The box to use. If the box is empty, the system is assumed to be non-periodic. Default is an empty box.
  :param max_num_pairs: The maximum number of pairs. Default is 32.
  :param check_error: If true, the function will wait for the kernel and throw if the number of pairs exceeds the maximum provided. Default is false.
  :return: A tuple of :cpp:any:`usm_vector` s consisting of the following:

      * A list of neighbors for each particle, shape (max_num_pairs, 2). Type :cpp:`int`.
      * A list of vectors between particles (r_ij), shape (max_num_pairs). Type :cpp:any:`vec3<T>`.
      * A list with the norms of the distances, shape (max_num_pairs). Type :cpp:any:`T`.
      * The total number of pairs found. Shape (1). Type :cpp:`int`.
      
    
      
.. _compute-neighbor-pairs:

Compute Neighbor Pairs (Alternative)
====================================

This module provides an alternative function to compute the neighbor pairs of a group of 3D points. This version allows to automatically increase the maximum number of neighbors until all required pairs fit.

.. cpp:function:: template <std::floating_point T> auto computeNeighborPairs(sycl::buffer<vec3<T>>& positions, T cutoff = std::numeric_limits<T>::infinity(), Box<T> box = empty_box<T>, int max_num_pairs = 32, bool resize_to_fit = false)

   This function computes the neighbors of each particle, with an option to resize the neighbor indices buffer to fit all neighbors.
   Note that the size of the output might be larger than the max_num_pairs provided if resize_to_fit=true.
   
   :param positions: The positions of the particles with shape (num_particles, 3).
   :param cutoff: The cutoff distance. (optional, default: `std::numeric_limits<T>::infinity()`)
   :param box: The box to use. If the box is empty, the system is assumed to be non-periodic. (optional, default: `empty_box<T>`)
   :param max_num_pairs: The maximum number of pairs. (optional, default: 32)
   :param resize_to_fit: If true, the neighbor indices buffer will be resized until all neighbors fit. (optional, default: false)

   :return: A tuple of :cpp:any:`usm_vector` s consisting of the following:

      * A list of neighbors for each particle, shape (max_num_pairs, 2). Type :cpp:`int`.
      * A list of vectors between particles (r_ij), shape (max_num_pairs). Type :cpp:any:`vec3<T>`.
      * A list with the norms of the distances, shape (max_num_pairs). Type :cpp:any:`T`.
      * The total number of pairs found. Shape (1). Type :cpp:`int`.

