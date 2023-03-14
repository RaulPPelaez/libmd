/* Raul P. Pelaez 2023.
   This example shows how to use the neighbor list function in libMD.
   SYCL is used to parallelize the computation and handle storage.
*/
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

int main() {
  int num_particles = 100;
  float box_size = 16.f;
  float cutoff = 4.0f;
  auto positions = generateParticleCloud(num_particles, box_size);
  auto [neighbors, neighbor_indices, max_num_neighbors] =
      md::computeNeighbors(positions, cutoff);
  // The format of the neighbor list is as follows:
  // neighbors[i]: The number of neighbors of particle i
  // neighbor_indices[i * max_num_neighbors + j]: The index of the jth neighbor
  // of particle i max_num_neighbors: The maximum number of neighbors of any
  // particle
  std::cout << "Number of neighbors of particle 0: " << neighbors[0]
            << std::endl;
  // Print every neighbor of particle 0 in the format "Neighbors of particle 0:
  // neighbor1 neighbor2 ..."
  std::cout << "Neighbors of particle 0: ";
  for (int i = 0; i < neighbors[0]; i++) {
    std::cout << neighbor_indices[i] << " ";
  }
  std::cout << std::endl;
  return 0;
}
