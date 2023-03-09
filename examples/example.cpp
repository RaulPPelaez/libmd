/* Raul P. Pelaez 2023.
   Basic interaction with libmd.
 */
#include <libmd.h>

using namespace cl;

int main() {

  sycl::queue q = md::get_default_queue();
  // Print some properties of the sycl queue
  md::log("Device: " + q.get_device().get_info<sycl::info::device::name>());
  md::log("Vendor: " + q.get_device().get_info<sycl::info::device::vendor>());
  md::log("Driver version: " +
          q.get_device().get_info<sycl::info::device::driver_version>());
  md::log("Max work group size: %d",
          q.get_device().get_info<sycl::info::device::max_work_group_size>());
  md::log(
      "Max work item dimensions: %d",
      q.get_device().get_info<sycl::info::device::max_work_item_dimensions>());
  md::log("Max work item sizes: %d %d %d",
					q.get_device().get_info<sycl::info::device::max_work_item_sizes>()[0],
					q.get_device().get_info<sycl::info::device::max_work_item_sizes>()[1],
					q.get_device().get_info<sycl::info::device::max_work_item_sizes>()[2]);
  md::log("Max compute units: %d",
	  q.get_device().get_info<sycl::info::device::max_compute_units>());
  return 0;
}
