/* Raul P. Pelaez 2023. Queue utilities for libMD
 */
#pragma once
#include"common.hpp"
namespace md{

  using queue = sycl::queue;

  /**
   * @brief Returns the default queue used by libMD
   *
   */
  static inline queue& get_default_queue() {
    static sycl::default_selector selector;
    static queue q{selector};
    return q;
  }
}
