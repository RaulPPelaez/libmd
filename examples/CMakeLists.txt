# add_executable(example example.cpp)
# add_sycl_to_target(TARGET example SOURCES example.cpp)
# install(TARGETS example
#         RUNTIME DESTINATION examples/)

# Add an executable for every .cpp file in this folder
file(GLOB_RECURSE EXAMPLES "*.cpp")
foreach(EXAMPLE ${EXAMPLES})
		get_filename_component(EXAMPLE_NAME ${EXAMPLE} NAME_WE)
		add_executable(${EXAMPLE_NAME} ${EXAMPLE})
		add_sycl_to_target(TARGET ${EXAMPLE_NAME} SOURCES ${EXAMPLE})
		install(TARGETS ${EXAMPLE_NAME}
		  RUNTIME DESTINATION examples/)
endforeach()
