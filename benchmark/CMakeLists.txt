
# Add an executable for every .cpp file in this folder
file(GLOB_RECURSE BENCHMARKS "*.cpp")
foreach(BENCHMARK ${BENCHMARKS})
  get_filename_component(BENCHMARK_NAME ${BENCHMARK} NAME_WE)
  string(APPEND BENCHMARK_NAME "_benchmark")
  add_executable(${BENCHMARK_NAME} ${BENCHMARK})
  add_sycl_to_target(TARGET ${BENCHMARK_NAME} SOURCES ${BENCHMARK})
  install(TARGETS ${BENCHMARK_NAME}
    RUNTIME DESTINATION benchmark/)
endforeach()
