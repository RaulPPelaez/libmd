cmake_minimum_required (VERSION 3.5)
project(libmd VERSION 0.1)
set(CMAKE_MODULE_PATH ${CMAKE_MODULE_PATH} ${PROJECT_SOURCE_DIR}/../cmake)
set (CMAKE_CXX_STANDARD 20)
find_package(hipSYCL CONFIG REQUIRED)

set(MD_LOG_LEVEL 15 CACHE STRING "Log level")
add_compile_definitions(MD_LOG_LEVEL=${MD_LOG_LEVEL})

set(THREADS_PREFER_PTHREAD_FLAG ON)
find_package(Threads REQUIRED)
link_libraries(Threads::Threads)
if(NOT CMAKE_BUILD_TYPE)
  set(CMAKE_BUILD_TYPE DEBUG)
endif()

# Add optimization flags in Release mode
if(CMAKE_BUILD_TYPE MATCHES Release)
  set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -O3")
elseif(CMAKE_BUILD_TYPE MATCHES Debug)
  set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -O0 -g")
endif()
if(WIN32)
  add_definitions(-D_USE_MATH_DEFINES)
endif()

include_directories(${PROJECT_BINARY_DIR} ${PROJECT_SOURCE_DIR} include)
enable_testing()
subdirs(examples tests benchmark)
