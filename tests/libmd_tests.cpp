#include "common.hpp"
#include <concepts>
#include <gtest/gtest.h>
#include "gmock/gmock.h"
#include <libmd.h>
#include<iostream>

using namespace cl;
using namespace md;

TEST(Queue, IsCreatedCorrectly){
  auto q = md::get_default_qeue();
  //List properties of the queue with the log function
  log("Queue has device called %s", q.get_device().get_info<sycl::info::device::name>().c_str());
  log("Queue has %d compute units", q.get_device().get_info<sycl::info::device::max_compute_units>());
}

TEST(Queue, BufferCanBeCreated){
  auto positions = sycl::buffer<vec3<float>>(2);
  //Get a host accessor for the property with write and sycl::no_init
  sycl::host_accessor positions_acc {positions, sycl::write_only, sycl::no_init};
  //Get a host pointer to the property
  auto positions_ptr = positions_acc.get_pointer();
  //Assert the pointer is not null
  ASSERT_NE(positions_ptr, nullptr);
}

TEST(NeighborList, IsCorrectForTwoParticlesOpenBox){
  auto q = md::get_default_qeue();
  auto positions = sycl::buffer<vec3<float>>(2);
  {
    sycl::host_accessor positions_acc{positions, sycl::write_only, sycl::no_init};
    positions_acc[0] = vec3<float>(0, 0, 0);
    positions_acc[1] = vec3<float>(1, 0, 0);
  }
  auto cutoff = 1.5f;
  auto [neighbors, neighbor_indices, max_num_neighbors] =
    md::computeNeighbors(q, positions, cutoff, empty_box<float>);
  sycl::host_accessor neighbors_acc{neighbors, sycl::read_only};
  sycl::host_accessor neighbor_indices_acc{neighbor_indices, sycl::read_only};
  EXPECT_EQ(neighbors_acc[0], 1);
  EXPECT_EQ(neighbors_acc[1], 1);
  EXPECT_EQ(neighbor_indices_acc[0], 1);
  EXPECT_EQ(neighbor_indices_acc[1], 0);
}

template <std::floating_point T>
void nbody_test(int num_particles, vec3<T> box_size, bool periodic, T cutoff){
  auto q = md::get_default_qeue();
  auto positions = sycl::buffer<vec3<T>>(num_particles);
  {
    sycl::host_accessor positions_acc{positions, sycl::write_only, sycl::no_init};
    //Positions are placed randomly inside a cubic box
    std::mt19937 gen(0xBADA55D00D);
    std::uniform_real_distribution<T> dis(0, 1);
    for(int i = 0; i < num_particles; i++){
      positions_acc[i] = vec3<T>(dis(gen), dis(gen), dis(gen))*box_size;
    }
  }
  auto box = empty_box<T>;
  if(periodic) box = Box(box_size);
  auto [neighbors, neighbor_indices, max_num_neighbors] =
    md::computeNeighbors(q, positions, cutoff, box);
  sycl::host_accessor neighbors_acc{neighbors, sycl::read_only};
  sycl::host_accessor neighbor_indices_acc{neighbor_indices, sycl::read_only};
  //Check that all particles have num_particles - 1 neighbors
  for(int i = 0; i < num_particles; i++){
    ASSERT_EQ(neighbors_acc[i], num_particles - 1)<< "Particle " << i << " has wrong number of neighbors";
  }
  //Check that all pairs are included as neighbors
  for(int i = 0; i < num_particles; i++){
    for(int j = 0; j < num_particles; j++){
      if(i == j) continue;
      ASSERT_TRUE(std::find(neighbor_indices_acc.get_pointer() + i * max_num_neighbors,
			    neighbor_indices_acc.get_pointer() + (i+1) * max_num_neighbors,
			    j))<< "Particle " << i << " does not have " << j << " as neighbor";
    }
  }
}

TEST(NeighborList, IsCorrectForNParticlesNBody){
  int num_particles = 100;
  float box_size = 128;
  nbody_test<float>(num_particles, vec3<float>(box_size), false, sqrt(3)*box_size);
}

TEST(NeighborList, IsCorrectForNParticlesNBodyPeriodicBox){
  int num_particles = 100;
  float box_size = 128;
  nbody_test<float>(num_particles, vec3<float>(box_size), true, 0.5*box_size);
}
