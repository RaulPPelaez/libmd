include(FetchContent)
FetchContent_Declare(
  googletest
  GIT_REPOSITORY https://github.com/google/googletest.git
  GIT_TAG release-1.12.1
)
# For Windows: Prevent overriding the parent project's compiler/linker settings
set(gtest_force_shared_crt ON CACHE BOOL "" FORCE)
FetchContent_MakeAvailable(googletest)
enable_testing()
add_executable(libmd_tests  libmd_tests.cpp)

target_link_libraries(
  libmd_tests PRIVATE
  GTest::gtest_main
  GTest::gmock_main
)
add_sycl_to_target(TARGET libmd_tests)
include(GoogleTest)
gtest_discover_tests(libmd_tests)
