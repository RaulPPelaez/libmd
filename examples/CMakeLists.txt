add_executable(example example.cpp)
add_sycl_to_target(TARGET example SOURCES example.cpp)
install(TARGETS example
        RUNTIME DESTINATION examples/)
