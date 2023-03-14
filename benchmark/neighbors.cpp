#include <libmd.h>
using namespace md;

// This function generates a random particle cloud inside a cubic box
// Returns a sycl buffer containing the positions of the particles
auto generateParticleCloud(int num_particles, float box_size) {
  std::vector<vec3<float>> positions(num_particles);
  // Positions are placed randomly inside a cubic box
  std::mt19937 gen(0xBADA55D00D);
  std::uniform_real_distribution<float> dis(0, 1);
  for (int i = 0; i < num_particles; i++) {
    positions[i] = vec3<float>(dis(gen), dis(gen), dis(gen)) * box_size;
  }
  return positions;
}

auto run_benchmark(float density, int num_particles,
                   int expected_num_neighbors) {
  const float lbox = cbrt(num_particles / density);
  const Box<float> box(lbox);
  const float cutoff = cbrt(3 * expected_num_neighbors / (4 * M_PI * density));
  int max_num_neighbors = expected_num_neighbors;
  auto positions_v = generateParticleCloud(num_particles, lbox);
  sycl::buffer positions(positions_v.begin(), positions_v.end());
  int warmup = 10;
  int nprof = 1000;
  int ntest = nprof + warmup;
  auto q = md::get_default_queue();
  auto start = std::chrono::high_resolution_clock::now();
  auto [num_neighbors, neighbor_indices, found_max_num_neighbors] =
      computeNeighbors(positions, cutoff, box, max_num_neighbors, true);
  max_num_neighbors = found_max_num_neighbors;
  for (int i = 0; i < ntest; i++) {
    if (i == warmup) {
      q.wait_and_throw();
      start = std::chrono::high_resolution_clock::now();
    }
    std::tie(num_neighbors, neighbor_indices, found_max_num_neighbors) =
        computeNeighbors(positions, cutoff, box, max_num_neighbors);
  }
  q.wait_and_throw();
  auto end = std::chrono::high_resolution_clock::now();
  // Average num_neighbors over all particles
  // sycl::host_accessor num_neighbors_acc{num_neighbors, sycl::read_only};
  float mean_num_neighbors;

  mean_num_neighbors =
      std::accumulate(num_neighbors.begin(), num_neighbors.end(), 0.0f);
  mean_num_neighbors /= num_particles;
  log<MESSAGE>(
      "num_particles: %d, max_num_neighbors: %d, mean_num_neighbors: %g",
      num_particles, max_num_neighbors, mean_num_neighbors);
  auto elapsed =
      std::chrono::duration_cast<std::chrono::nanoseconds>(end - start);
  return elapsed.count() / nprof;
}

int main() {
  {
    sycl::queue q = md::get_default_queue();
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
    if (!q.get_device().has(sycl::aspect::usm_shared_allocations)) {
      log<ERROR>("Device does not support usm_shared_allocations");
      return 1;
    }

    float density = 0.5;
    printf("#%-10s\t%-10s\n", "num_particles", "time (ms)");

    for (int n = 16; n >= 1; n--) {
      int num_particles = 1 << n;
      int expected_num_neighbors = std::min(num_particles, 128);
      auto elapsed =
        run_benchmark(density, num_particles, expected_num_neighbors);
      printf("%-10d\t%-10.3f\n", num_particles, elapsed / 1e6);
    }
  }
  cleanup();
  return 0;
}
