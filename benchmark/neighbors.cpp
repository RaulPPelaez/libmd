#include "common.hpp"
#include <hipSYCL/sycl/device_selector.hpp>
#include <libmd.h>

using namespace md;

// This function generates a random particle cloud inside a cubic box
// Returns a sycl buffer containing the positions of the particles
auto generateParticleCloud(int num_particles, float box_size) {
  auto positions = sycl::buffer<vec3<float>>(num_particles);
  sycl::host_accessor positions_acc{positions, sycl::write_only, sycl::no_init};
  // Positions are placed randomly inside a cubic box
  std::mt19937 gen(0xBADA55D00D);
  std::uniform_real_distribution<float> dis(0, 1);
  for (int i = 0; i < num_particles; i++) {
    positions_acc[i] = vec3<float>(dis(gen), dis(gen), dis(gen)) * box_size;
  }
  return positions;
}

auto run_benchmark(sycl::queue q, float density, int num_particles,
                   int expected_num_neighbors) {
  float lbox = cbrt(num_particles / density);
  Box<float> box(lbox);
  float cutoff = cbrt(3 * expected_num_neighbors / (4 * M_PI * density));
  int max_num_neighbors = 32;
  auto positions = generateParticleCloud(num_particles, lbox);
  int warmup = 3;
  int nprof = 1000;
  int ntest = nprof + warmup;
  auto start = std::chrono::high_resolution_clock::now();
  auto [num_neighbors, neighbor_indices, found_max_num_neighbors] =
    computeNeighbors(q, positions, cutoff, box, max_num_neighbors, true);
  max_num_neighbors = found_max_num_neighbors;
  for (int i = 0; i < ntest; i++) {
    // Measure time beyond the 3rd iteration
    if (i == warmup) {
      q.wait_and_throw();
      start = std::chrono::high_resolution_clock::now();
    }
    std::tie(num_neighbors, neighbor_indices, found_max_num_neighbors) =
      computeNeighbors(q, positions, cutoff, box, max_num_neighbors);
  }
  q.wait_and_throw();
  auto end = std::chrono::high_resolution_clock::now();
  int mean_num_neighbors;
  //Average num_neighbors over all particles
  sycl::host_accessor num_neighbors_acc{num_neighbors, sycl::read_only};
  mean_num_neighbors = std::accumulate(num_neighbors_acc.get_pointer(),
				       num_neighbors_acc.get_pointer() + num_particles,
				       0);
  mean_num_neighbors /= num_particles;
  log<MESSAGE>("num_particles: %d, max_num_neighbors: %d, mean_num_neighbors: %d",
	       num_particles, max_num_neighbors, mean_num_neighbors);
  auto elapsed =
      std::chrono::duration_cast<std::chrono::nanoseconds>(end - start);
  return elapsed.count() / nprof;
}

int main() {
  sycl::property_list prop_list{sycl::property::queue::enable_profiling()};
  auto selector = sycl::default_selector{};
  sycl::queue q(selector, prop_list);
  // Print queue information
  log<MESSAGE>("Running on %s",
               q.get_device().get_info<sycl::info::device::name>().c_str());
  log<MESSAGE>("Device vendor: %s",
               q.get_device().get_info<sycl::info::device::vendor>().c_str());
  log<MESSAGE>("Device version: %s",
               q.get_device().get_info<sycl::info::device::version>().c_str());
  log<MESSAGE>(
      "Device driver version: %s",
      q.get_device().get_info<sycl::info::device::driver_version>().c_str());
  float density = 0.5;
  printf("#%-10s\t%-10s\n", "num_particles", "time (ms)");

  for (int n = 1; n <= 14; n++) {
    int num_particles = 1 << n;
    int expected_num_neighbors = std::min(num_particles, 128);
    auto elapsed =
        run_benchmark(q, density, num_particles, expected_num_neighbors);
    printf("%-10d\t%-10.3f\n", num_particles, elapsed / 1e6);
  }

  return 0;
}
