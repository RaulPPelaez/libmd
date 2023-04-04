/* Raul P. Pelaez 2023
   Tests for the libmd library
 */
#include <gtest/gtest.h>
#include "gmock/gmock.h"
#include <libmd.h>
#include <iostream>

using namespace cl;
using namespace md;

void assert_eq_vec3(vec3<float> a, vec3<float> b) {
  for (int i = 0; i < 3; i++) {
    ASSERT_EQ(a[i], b[i]);
  }
}

void assert_near_vec3(vec3<float> a, vec3<float> b, float tol = 1e-6) {
  for (int i = 0; i < 3; i++) {
    ASSERT_THAT(a[i], ::testing::FloatNear(b[i], tol));
  }
}

// Gives a different vec every time
auto get_random_vec3(vec3<float> min, vec3<float> max) {
  static std::mt19937 gen(0xBADA55D00D);
  static std::uniform_real_distribution<float> dis(0, 1);
  return vec3<float>(dis(gen), dis(gen), dis(gen)) * (max - min) + min;
}

TEST(Queue, DefaultIsCreatedCorrectly) {
  auto q = md::get_default_queue();
  // List properties of the queue with the log function
  log("Queue has device called %s",
      q.get_device().get_info<sycl::info::device::name>().c_str());
  log("Queue has %d compute units",
      q.get_device().get_info<sycl::info::device::max_compute_units>());
}

TEST(Queue, BufferCanBeCreated) {
  auto positions = sycl::buffer<vec3<float>>(2);
  // Get a host accessor for the property with write and sycl::no_init
  sycl::host_accessor positions_acc{positions, sycl::write_only, sycl::no_init};
  // Get a host pointer to the property
  auto positions_ptr = positions_acc.get_pointer();
  // Assert the pointer is not null
  ASSERT_NE(positions_ptr, nullptr);
}

TEST(Queue, CanUseAtomicRef) {
  // Create a queue with the default device
  sycl::queue q = md::get_default_queue();
  constexpr int N = 100;
  int v = 0;
  sycl::buffer<int> buf(&v, 1);
  q.submit([&](sycl::handler& h) {
     sycl::accessor acc{buf, h, sycl::read_write};
     h.parallel_for<class AtomicExample>(sycl::range<1>{N}, [=](sycl::id<1> i) {
       // Define an atomic reference to an integer in the buffer
       auto atom = sycl::atomic_ref<int, sycl::memory_order::relaxed,
                        sycl::memory_scope::device>(acc[0]);
       // Use the atomic reference to increment the value in the buffer
       // atomically
       atom.fetch_add(1);
     });
   }).wait_and_throw();
  sycl::host_accessor acc{buf, sycl::read_only};
  ASSERT_EQ(acc[0], N);
}

TEST(Queue, EquivalentDevicesAreEqual) {
  auto q1 = sycl::queue{sycl::default_selector{}};
  auto q2 = sycl::queue{sycl::default_selector{}};
  auto dev=  q2.get_device();
  ASSERT_TRUE(q1.get_device() == q2.get_device());
}


TEST(Queue, EquivalentContextsAreEqual) {
  auto q1 = get_default_queue();
  auto q2 = get_default_queue();
  ASSERT_TRUE(q1.get_context() == q2.get_context());
}


TEST(Allocator, CanAllocateWithUSMResource) {
  // Check that default queue has usm_shared_allocation support
  ASSERT_TRUE(md::get_default_queue().get_device().has(
      sycl::aspect::usm_shared_allocations));
  auto usm_res = md::get_default_usm_memory_resource();
  for (size_t i = 1; i < 1e7; i *= 10) {
    usm_res->do_allocate(i, 0);
  }
  usm_res->free_all();
}

TEST(NeighborList, IsCorrectForTwoParticlesOpenBox) {
  auto q = md::get_default_queue();
  auto positions = sycl::buffer<vec3<float>>(2);
  {
    sycl::host_accessor positions_acc{positions, sycl::write_only,
                                      sycl::no_init};
    positions_acc[0] = vec3<float>(0, 0, 0);
    positions_acc[1] = vec3<float>(1, 0, 0);
  }
  auto cutoff = 1.5f;
  auto [neighbors, neighbor_indices, max_num_neighbors] =
      md::computeNeighbors(positions, cutoff, empty_box<float>, 32, true);
  EXPECT_EQ(neighbors[0], 1);
  EXPECT_EQ(neighbors[1], 1);
  EXPECT_EQ(neighbor_indices[0], 1);
  EXPECT_EQ(neighbor_indices[1], 0);
}

template <std::floating_point T>
void nbody_test(int num_particles, vec3<T> box_size, bool periodic, T cutoff) {
  auto q = md::get_default_queue();
  auto positions = sycl::buffer<vec3<T>>(num_particles);
  {
    sycl::host_accessor positions_acc{positions, sycl::write_only,
                                      sycl::no_init};
    // Positions are placed randomly inside a cubic box
    std::mt19937 gen(0xBADA55D00D);
    std::uniform_real_distribution<T> dis(0, 1);
    for (int i = 0; i < num_particles; i++) {
      positions_acc[i] = vec3<T>(dis(gen), dis(gen), dis(gen)) * box_size;
    }
  }
  auto box = empty_box<T>;
  if (periodic)
    box = Box(box_size);
  int expected_neighs = num_particles - 1;
  auto [neighbors, neighbor_indices, max_num_neighbors] =
      md::computeNeighbors(positions, cutoff, box, expected_neighs, true);
  // Check that all particles have num_particles - 1 neighbors
  for (int i = 0; i < num_particles; i++) {
    ASSERT_EQ(neighbors[i], num_particles - 1)
        << "Particle " << i << " has wrong number of neighbors";
  }
  // Check that all pairs are included as neighbors
  for (int i = 0; i < num_particles; i++) {
    for (int j = 0; j < num_particles; j++) {
      if (i == j)
        continue;
      ASSERT_TRUE(
          std::find(neighbor_indices.data() + i * max_num_neighbors,
                    neighbor_indices.data() + (i + 1) * max_num_neighbors, j))
          << "Particle " << i << " does not have " << j << " as neighbor";
    }
  }
}

TEST(NeighborList, CorrectlyResizes) {
  auto q = md::get_default_queue();
  int num_particles = 1000;
  std::vector<vec3<float>> h_pos(num_particles, vec3<float>());
  sycl::buffer positions(h_pos.begin(), h_pos.end());
  auto cutoff = 1.5f;
  auto [neighbors, neighbor_indices, max_num_neighbors] =
      md::computeNeighbors(positions, cutoff, empty_box<float>, 32, true);
  ASSERT_GE(max_num_neighbors, num_particles - 1);
}

TEST(NeighborList, IsCorrectForNParticlesNBody) {
  int num_particles = 100;
  float box_size = 128;
  nbody_test<float>(num_particles, vec3<float>(box_size), false,
                    sqrt(3) * box_size);
}

template <std::floating_point T>
auto apply_periodic_boundary_conditions_naive(vec3<T> pos, Box<T> box) {
  vec3<T> result;
  for (int i = 0; i < 3; i++) {
    result[i] = pos[i];
    while (result[i] <= -box.size[i][i] * 0.5)
      result[i] += box.size[i][i];
    while (result[i] >= box.size[i][i] * 0.5)
      result[i] -= box.size[i][i];
  }
  return result;
}

TEST(Box, PeriodicBoundaryConditionsOrthorhombic) {
  vec3<float> lbox(10, 20, 30);
  auto box = Box(lbox);
  auto result = apply_periodic_boundary_conditions({1.5f, 0.f, 0.f}, box);
  assert_near_vec3(result, vec3<float>(1.5, 0, 0));
  result = apply_periodic_boundary_conditions({lbox.x() + 1, 0, 0}, box);
  assert_near_vec3(result, {1.0, 0, 0});
  // Test some aritrary distances multiple times the box size
  for (int i = 0; i < 10; i++) {
    result = apply_periodic_boundary_conditions({i * lbox.x() + 1, 0, 0}, box);
    assert_near_vec3(result, {1.0, 0, 0});
  }
  // Test random distances
  for (int i = 0; i < 10; i++) {
    auto pos = get_random_vec3(-10 * lbox, 10 * lbox);
    result = apply_periodic_boundary_conditions(pos, box);
    assert_near_vec3(result,
                     apply_periodic_boundary_conditions_naive(pos, box));
  }
}

TEST(NeighborList, IsCorrectForNParticlesNBodyPeriodicBox) {
  float box_size = 128;
  for (int exponent = 12; exponent > 0; exponent--) {
    int num_particles = 1 << exponent;
    nbody_test<float>(num_particles, vec3<float>(box_size), true,
                      sqrt(3) * box_size * 0.5);
  }
}



TEST(PairList, IsCorrectForTwoParticlesOpenBox) {
  auto q = md::get_default_queue();
  auto positions = sycl::buffer<vec3<float>>(2);
  {
    sycl::host_accessor positions_acc{positions, sycl::write_only,
                                      sycl::no_init};
    positions_acc[0] = vec3<float>(0, 0, 0);
    positions_acc[1] = vec3<float>(1, 0, 0);
  }
  auto cutoff = 1.5f;
  auto [neighbors, deltas, distances, num_pairs] =
      md::computeNeighborPairs(positions, cutoff, empty_box<float>, 32, true);
  q.wait_and_throw();
  EXPECT_EQ(neighbors[0], 1);
  EXPECT_EQ(neighbors[1], 0);
  EXPECT_NEAR(distances[0], 1, 1e-5);
  EXPECT_NEAR(deltas[0].x(), 1, 1e-5);
  EXPECT_NEAR(deltas[0].y(), 0, 1e-5);
  EXPECT_NEAR(deltas[0].z(), 0, 1e-5);
  EXPECT_EQ(num_pairs[0], 1);
}

template <std::floating_point T>
void nbody_test_pairs(int num_particles, vec3<T> box_size, bool periodic, T cutoff) {
  auto q = md::get_default_queue();
  auto positions = sycl::buffer<vec3<T>>(num_particles);
  {
    sycl::host_accessor positions_acc{positions, sycl::write_only,
                                      sycl::no_init};
    // Positions are placed randomly inside a cubic box
    std::mt19937 gen(0xBADA55D00D);
    std::uniform_real_distribution<T> dis(0, 1);
    for (int i = 0; i < num_particles; i++) {
      positions_acc[i] = vec3<T>(dis(gen), dis(gen), dis(gen)) * box_size;
    }
  }
  auto box = empty_box<T>;
  if (periodic)
    box = Box(box_size);
  int expected_pairs = num_particles*(num_particles - 1)/2;
  auto [neighbors, deltas, distances, num_pairs] =
    md::computeNeighborPairs(positions, cutoff, box, expected_pairs, true);
  q.wait_and_throw();
  ASSERT_EQ(num_pairs[0], expected_pairs) << "incorrect number of pairs";
  // Check that all pairs are included as neighbors
  std::vector<int> pairs(num_pairs[0]);
  {
    for(int i = 0; i < num_pairs[0]; i++) {
      int ii = neighbors[2*i];
      int jj = neighbors[2*i + 1];
      if (ii > jj)
	std::swap(ii, jj);
      pairs[i] = ii * num_particles + jj;
    }
  }
  std::sort(pairs.begin(), pairs.end());
  int c = 0;
  for(int i = 0; i < num_particles; i++) {
    for(int j = i + 1; j < num_particles; j++) {
      int pair = i * num_particles + j;
      ASSERT_TRUE(pairs[c] == pair) << "Pair (" << i << ", " << j << ") is missing";
      c++;
    }

  }
}

TEST(PairList, CorrectlyResizes) {
  auto q = md::get_default_queue();
  int num_particles = 1000;
  std::vector<vec3<float>> h_pos(num_particles, vec3<float>());
  sycl::buffer positions(h_pos.begin(), h_pos.end());
  auto cutoff = 1.5f;
  auto [neighbors, deltas, distances, num_pairs] =
      md::computeNeighborPairs(positions, cutoff, empty_box<float>, 32, true);
  q.wait_and_throw();
  ASSERT_GE(num_pairs[0], num_particles*(num_particles - 1)/2);
}

TEST(PairList, IsCorrectForNParticlesNBody) {
  int num_particles = 100;
  float box_size = 128;
  nbody_test_pairs<float>(num_particles, vec3<float>(box_size), false,
                    sqrt(3) * box_size);
}


TEST(PairList, IsCorrectForNParticlesNBodyPeriodicBox) {
  float box_size = 128;
  for (int exponent = 12; exponent > 0; exponent--) {
    int num_particles = 1 << exponent;
    nbody_test<float>(num_particles, vec3<float>(box_size), true,
                      sqrt(3) * box_size * 0.5);
  }
}


TEST(Clean, CleanUp) {
  cleanup();
}
