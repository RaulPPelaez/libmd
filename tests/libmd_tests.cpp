#include "common.hpp"
#include <concepts>
#include <gtest/gtest.h>
#include "gmock/gmock.h"
#include <libmd.h>
#include<iostream>

using namespace cl;
using namespace md;

void assert_eq_vec3(vec3<float> a, vec3<float> b){
  for(int i = 0; i < 3; i++){
    ASSERT_EQ(a[i], b[i]);
  }
}

void assert_near_vec3(vec3<float> a, vec3<float> b, float tol = 1e-6){
  for(int i = 0; i < 3; i++){
    ASSERT_THAT(a[i], ::testing::FloatNear(b[i], tol));
  }
}

//Gives a different vec every time
auto get_random_vec3(vec3<float> min, vec3<float> max){
  static std::mt19937 gen(0xBADA55D00D);
  static std::uniform_real_distribution<float> dis(0, 1);
  return vec3<float>(dis(gen), dis(gen), dis(gen))*(max - min) + min;
}

TEST(Queue, IsCreatedCorrectly){
  auto q = md::get_default_queue();
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

TEST(Queue, CanUseAtomicRef){
  //Create a queue with the default device
  sycl::queue q = md::get_default_queue();
  constexpr int N = 100;
  int v = 0;
  sycl::buffer<int> buf(&v, 1);
  q.submit([&](sycl::handler& h) {
    sycl::accessor acc{buf, h, sycl::read_write};
    h.parallel_for<class AtomicExample>(sycl::range<1>{N}, [=](sycl::id<1> i) {
      // Define an atomic reference to an integer in the buffer
      sycl::atomic_ref<int, sycl::memory_order::relaxed,
		       sycl::memory_scope::device> atomic_ref(acc[0]);
      // Use the atomic reference to increment the value in the buffer atomically
      atomic_ref++;
    });
  }).wait();
  sycl::host_accessor acc{buf};
  ASSERT_EQ(acc[0], N);
}


TEST(NeighborList, IsCorrectForTwoParticlesOpenBox){
  auto q = md::get_default_queue();
  auto positions = sycl::buffer<vec3<float>>(2);
  {
    sycl::host_accessor positions_acc{positions, sycl::write_only, sycl::no_init};
    positions_acc[0] = vec3<float>(0, 0, 0);
    positions_acc[1] = vec3<float>(1, 0, 0);
  }
  auto cutoff = 1.5f;
  auto [neighbors, neighbor_indices, max_num_neighbors] =
    md::computeNeighbors(q, positions, cutoff, empty_box<float>, 32, true);
  sycl::host_accessor neighbors_acc{neighbors, sycl::read_only};
  sycl::host_accessor neighbor_indices_acc{neighbor_indices, sycl::read_only};
  EXPECT_EQ(neighbors_acc[0], 1);
  EXPECT_EQ(neighbors_acc[1], 1);
  EXPECT_EQ(neighbor_indices_acc[0], 1);
  EXPECT_EQ(neighbor_indices_acc[1], 0);
}

template <std::floating_point T>
void nbody_test(int num_particles, vec3<T> box_size, bool periodic, T cutoff){
  auto q = md::get_default_queue();
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
    md::computeNeighbors(q, positions, cutoff, box, 32, true);
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

template <std::floating_point T>
auto apply_periodic_boundary_conditions_naive(vec3<T> pos, Box<T> box){
  vec3<T> result;
  for(int i = 0; i < 3; i++){
    result[i] = pos[i];
    while(result[i] <= -box.size[i][i]*0.5) result[i] += box.size[i][i];
    while(result[i] >= box.size[i][i]*0.5) result[i] -= box.size[i][i];
  }
  return result;
}

TEST(Box, PeriodicBoundaryConditionsOrthorhombic){
  vec3<float> lbox(10, 20, 30);
  auto box = Box(lbox);
  auto result = apply_periodic_boundary_conditions({1.5f, 0.f, 0.f}, box);
  assert_near_vec3(result, vec3<float>(1.5, 0, 0));
  result = apply_periodic_boundary_conditions({lbox.x() + 1, 0, 0}, box);
  assert_near_vec3(result, {1.0, 0, 0});
  //Test some aritrary distances multiple times the box size
  for(int i = 0; i < 10; i++){
    result = apply_periodic_boundary_conditions({i*lbox.x() + 1, 0, 0}, box);
    assert_near_vec3(result, {1.0, 0, 0});
  }
  //Test random distances
  for(int i = 0; i < 10; i++){
    auto pos = get_random_vec3(-10*lbox, 10*lbox);
    result = apply_periodic_boundary_conditions(pos, box);
    assert_near_vec3(result, apply_periodic_boundary_conditions_naive(pos, box));
  }
}


TEST(NeighborList, IsCorrectForNParticlesNBodyPeriodicBox){
  int num_particles = 100;
  float box_size = 128;
  nbody_test<float>(num_particles, vec3<float>(box_size), true, sqrt(3)*box_size*0.5);
}
