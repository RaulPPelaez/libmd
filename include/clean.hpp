/* Raul P. Pelaez 2023. Clean up functions.

   Calling cleanup finishes every operation on the default queue and frees all memory used by libmd.
   Any container still existing after calling cleanup results in undefined behavior.
 */
#pragma once
#include"queue.hpp"
#include "allocator.hpp"

namespace md{
  /**
   * @brief Finish all remaining operations and clean all memory used by libmd
   *
   */
  static void cleanup(){
    auto q = get_default_queue();
    q.wait_and_throw();
    clear_memory_resources();

  }
}
