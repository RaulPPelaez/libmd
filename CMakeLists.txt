cmake_minimum_required (VERSION 3.5)
project(libmd VERSION 0.0)
set(CMAKE_MODULE_PATH ${CMAKE_MODULE_PATH} ${PROJECT_SOURCE_DIR}/../cmake)
set (CMAKE_CXX_STANDARD 20)
find_package(hipSYCL CONFIG REQUIRED)

if(NOT CMAKE_BUILD_TYPE)
  set(CMAKE_BUILD_TYPE Release)
endif()

if(WIN32)
  add_definitions(-D_USE_MATH_DEFINES)
endif()

include_directories(${PROJECT_BINARY_DIR} ${PROJECT_SOURCE_DIR} include)
subdirs(examples tests)
