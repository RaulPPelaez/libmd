/* Raul P. Pelaez 2023. SYCL-enabled memory utilities.

   This file contains code for:
   - A memory resource that uses USM to allocate memory.
   - A memory resource that pools allocations.
   - A container that uses the pooled usm memory resource.
 */
#pragma once
#include "common.hpp"
#include "log.hpp"
#include "queue.hpp"
#include <memory_resource>
#include <map>
namespace md {

  class usm_memory_resource : public std::pmr::memory_resource {
    sycl::device m_device;
    sycl::context m_context;

  public:
    explicit usm_memory_resource(sycl::queue& queue)
        : m_device(queue.get_device()), m_context(queue.get_context()) {}

    usm_memory_resource(sycl::device device, sycl::context context)
        : m_device(device), m_context(context) {}

    ~usm_memory_resource() override = default;

    void* do_allocate(std::size_t bytes, std::size_t alignment) override {
      log<DEBUG6>("Allocating %zu bytes", bytes);
      return sycl::malloc_shared(bytes, m_device, m_context);
    }

    void do_deallocate(void* p, std::size_t bytes,
                       std::size_t alignment) override {
      log<DEBUG6>("Deallocating %zu bytes", bytes);
      sycl::free(p, m_context);
    }

    bool do_is_equal(
        const std::pmr::memory_resource& other) const noexcept override {
      return this == &other;
    }
  };

  template <class MR> auto* get_default_resource() {
    static MR resource;
    return &resource;
  }

  template <> auto* get_default_resource<usm_memory_resource>() {
    static usm_memory_resource resource(get_default_queue());
    return &resource;
  }

  // A pool device memory_resource, stores previously allocated blocks in a
  // cache
  //  and retrieves them fast when similar ones are allocated again (without
  //  calling malloc everytime).
  template <class MR>
  struct pool_memory_resource_adaptor : public std::pmr::memory_resource {
  private:
    using super = std::pmr::memory_resource;
    MR* res;

  public:
    using pointer = void*;

    ~pool_memory_resource_adaptor() {
      try {
        free_all();
      }
      catch (...) {
      }
    }

    explicit pool_memory_resource_adaptor(MR* resource) : res(resource) {}

    pool_memory_resource_adaptor() : res(get_default_resource<MR>()) {}

    using FreeBlocks = std::multimap<std::ptrdiff_t, void*>;
    using AllocatedBlocks = std::map<void*, std::ptrdiff_t>;
    FreeBlocks free_blocks;
    AllocatedBlocks allocated_blocks;

    pointer do_allocate(std::size_t bytes, std::size_t alignment) override {
      pointer result;
      std::ptrdiff_t blockSize = 0;
      auto available_blocks = free_blocks.equal_range(bytes);
      auto available_block = available_blocks.first;
      // Look for a block of the same size
      if (available_block == free_blocks.end()) {
        available_block = available_blocks.second;
      }
      // Try to find a block greater than requested size
      if (available_block != free_blocks.end()) {
        result = static_cast<pointer>(available_block->second);
        blockSize = available_block->first;
        free_blocks.erase(available_block);
      } else {
        result = res->do_allocate(bytes, alignment);
        blockSize = bytes;
      }
      allocated_blocks.insert(std::make_pair(result, blockSize));
      return result;
    }

    void do_deallocate(pointer p, std::size_t bytes,
                       std::size_t alignment) override {
      auto block = allocated_blocks.find(p);
      if (block == allocated_blocks.end()) {
        throw std::system_error(EFAULT, std::generic_category(),
                                "Address is not handled by this instance.");
      }
      std::ptrdiff_t num_bytes = block->second;
      allocated_blocks.erase(block);
      free_blocks.insert(std::make_pair(num_bytes, p));
    }

    bool do_is_equal(const super& other) const noexcept override {
      return res->do_is_equal(other);
    }

    void free_unused() {
      for (auto& i : free_blocks)
        res->do_deallocate(static_cast<pointer>(i.second), i.first, 0);
      free_blocks.clear();
    }

    void free_all() {
      free_unused();
      for (auto& i : allocated_blocks)
        res->do_deallocate(static_cast<pointer>(i.first), i.second, 0);
      allocated_blocks.clear();
    }
  };

  using pool_usm_resource = pool_memory_resource_adaptor<usm_memory_resource>;

  // This namespace holds a map mapping pairs of device context to
  // usm_memory_resource objects. This is necessary because the
  // usm_memory_resource constructor requires a device, but the device is not
  // known at the time the memory resource is constructed. The map and the
  // resources are static
  namespace pool_usm_memory_resource_factory {

    struct device_hash {

      std::size_t operator()(std::pair<sycl::device, sycl::context> in) const {
        return in.first.hipSYCL_hash_code() ^ in.second.hipSYCL_hash_code();
      }
    };

    static std::unordered_map<std::pair<sycl::device, sycl::context>,
                              std::pair<std::shared_ptr<pool_usm_resource>,
                                        std::shared_ptr<usm_memory_resource>>,
                              device_hash>
        resources;

    static pool_usm_resource* get(sycl::device&& device,
                                  sycl::context&& context) {
      auto key = std::make_pair(device, context);
      auto it = resources.find(key);
      if (it == resources.end()) {
        log<DEBUG4>("Creating new usm_memory_resource for device %s",
                    device.get_info<sycl::info::device::name>().c_str());
        auto usm = std::make_shared<usm_memory_resource>(device, context);
        auto res = std::make_shared<pool_usm_resource>(usm.get());
        resources.insert(std::make_pair(key, std::make_pair(res, usm)));
        it = resources.find(key);
      }
      return it->second.first.get();
    }

    static pool_usm_resource* get(sycl::queue q) {
      return get(q.get_device(), q.get_context());
    }

  } // namespace pool_usm_memory_resource_factory

  // Free all resources and clear the map
  static void clear_memory_resources() {
    for (auto& i : pool_usm_memory_resource_factory::resources) {
      i.second.first->free_all();
    }
    pool_usm_memory_resource_factory::resources.clear();
  }

  static auto*
  get_default_usm_memory_resource(sycl::queue& q = get_default_queue()) {
    auto usm_res = pool_usm_memory_resource_factory::get(q);
    // static pool_memory_resource_adaptor<usm_memory_resource> pool_mr;
    // return &pool_mr;
    return usm_res;
  }

  /**
   * @brief A vector class that uses USM memory.
   *
   * @tparam T The type of the elements in the vector.
   */
  template <class T> class usm_vector {
    using Container = std::shared_ptr<T>;
    using Ptr = T*;
    Container m_data;
    size_t m_size, capacity;
    sycl::queue m_queue;

    /**
     * @brief Creates a new vector with the given size.
     *
     * @param s The size of the vector.
     * @param q The queue to use for the allocation.
     * @return A new vector with the given size.
     */
    static Container create(size_t s, sycl::queue& q = get_default_queue()) {
      if (s > 0) {
        auto* res = get_default_usm_memory_resource(q);
        return Container(
            static_cast<T*>(res->allocate(s * sizeof(T), alignof(T))),
            [=](T* ptr) { res->do_deallocate(ptr, 0, 0); });
      }
      return Container();
    }

  public:
    using iterator = T*;

    /**
     * @brief Constructs a new vector with the given size.
     *
     * @param i_size The size of the vector.
     * @param q The queue to use for the allocation.
     */
    explicit usm_vector(size_t i_size = 0, queue q = get_default_queue())
        : m_size(0), capacity(0), m_data(), m_queue(q) {
      this->resize(i_size);
    }

    /**
     * @brief Constructs a new vector by copying the elements from the given STL
     * vector.
     *
     * @param other The vector to copy.
     * @param q The queue to use for the allocation.
     */
    explicit usm_vector(const std::vector<T>& other,
                        queue q = get_default_queue())
        : usm_vector(other.size(), q) {
      std::copy(other.begin(), other.end(), begin());
    }

    T& operator[](size_t i) const { return m_data.get()[i]; }

    iterator begin() const { return iterator(data()); }

    iterator end() const { return begin() + m_size; }

    [[nodiscard]] size_t size() const { return m_size; }

    void resize(size_t newsize) {
      if (newsize > capacity) {
        auto data2 = create(newsize, m_queue);
        if (size() > 0) {
          std::copy(begin(), end(), data2.get());
        }
        m_data.swap(data2);
        capacity = newsize;
        m_size = newsize;
      } else {
        m_size = newsize;
      }
    }

    T* data() const { return m_data.get(); }

    void clear() {
      resize(0);
      m_data = create(0);
      capacity = 0;
    }

    void swap(usm_vector<T>& another) { m_data.swap(another.m_data); }

    operator std::vector<T>() const {
      std::vector<T> hvec(size());
      std::copy(begin(), end(), hvec.begin());
      return hvec;
    }
  };

  /**
   * @brief A vector class that uses USM memory and caches the allocation.
   *
   * @tparam T The type of the elements in the vector.
   */
  template <class T> using cached_vector = usm_vector<T>;
} // namespace md
