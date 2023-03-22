Compiling
==========

**libMD is a header-only library**. Instructions below will help you compile all the examples and tests for libMD, you can then adapt this process to compile your code.

Compilation requires a working SYCL compiler. This project was developed using `OpenSYCL <https://github.com/OpenSYCL/OpenSYCL>`, but any other implementation (such as `ComputeCpp <https://developer.codeplay.com/products/computecpp/ce/home>`) should work with minimal changes to CMakeLists.txt.

.. sidebar::

   .. hint:: Changing the compiler vendor should amount to changing the :code:`find_package` line at the start of CMakeLists.txt for the one suggested by the vendor.


In order to target a certain platform (such as CUDA, ROCm or OpenMP) SYCL requires a working backend for it.
If you are able to compile CUDA code, your SYCL compiler should be able to just pick that up. Similarly for the rest of backends.
However, vendors usually provide instructions and/or packages to facilitate backend installation. See for instance `OpenSYCL <https://github.com/OpenSYCL/OpenSYCL/blob/develop/install/scripts/README.md>`.

Compilation is done via CMake:

.. code:: bash

	  git clone https://github.com/RaulPPelaez/libMD
	  cd libMD
	  mkdir build
	  cd build
	  #Compilation requires a list of targets
	  OPENSYCL_TARGETS="omp;cuda:sm_70;hip:gfx906" cmake ..
	  #Solve any issues pointed by cmake before continuing
	  make
	  make test #Optionally run the tests 
	  
.. hint::

   Possible compilation targets depend on the SYCL implementation in use, run :code:`syclcc --help` for a list.

This will compile all examples and tests and put them under the build directory.
Take :code:`examples/CMakeLists.txt` as a template on how to compile sources external to libMD.
