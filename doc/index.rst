.. libMD documentation master file, created by RaulPPelaez
   You can adapt this file completely to your liking, but it should at least
   contain the root `toctree` directive.

.. _index:

Welcome to the libMD v0.0 documentation!
=========================================
----------------------------------------------------
libMD
----------------------------------------------------


For information about a particular module, see its page. You will find the most up to date information about a module in its header file.

-----

About
======

libMD is a library for common operations in machine learning methods applied to MD. libMD is presented as a SYCL library, written in C++20.


.. important::

   **Compiling and running any code containing libMD requires a working SYCL compiler**. We recommend using `OpenSYCL <https://github.com/OpenSYCL/OpenSYCL>`

If this is the first time you encounter libMD and want a quick start guide check the :doc:`QuickStart`. The :doc:`FAQ` is also a good source of knowledge.

Look in :doc:`Compiling` if some Makefile gives you troubles. libMD is a header only framework and it is mostly self contained in terms of dependencies.



A quick example:
=================

A brief example of how a code using libMD typically looks like.

.. code:: cpp

  auto q = md::get_default_queue();
  int num_particles = 100;
  float box_size = 128.f;
  float cutoff = 1.5f;
  auto positions = sycl::buffer<vec3<float>>(num_particles);
  {
    sycl::host_accessor positions_acc{positions, sycl::write_only, sycl::no_init};
    //Positions are placed randomly inside a cubic box
    std::mt19937 gen(0xBADA55D00D);
    std::uniform_real_distribution<float> dis(0, 1);
    for(int i = 0; i < num_particles; i++){
      positions_acc[i] = vec3<float>(dis(gen), dis(gen), dis(gen))*box_size;
    }
  }
  auto [neighbors, neighbor_indices, max_num_neighbors] =
    md::computeNeighbors(q, positions, cutoff);
  //The format of the neighbor list is as follows:
  //neighbors[i]: The number of neighbors of particle i
  //neighbor_indices[i * max_num_neighbors + j]: The index of the jth neighbor of particle i
  //max_num_neighbors: The maximum number of neighbors of any particle
  sycl::host_accessor neighbors_acc{neighbors, sycl::read_only};
  std::cout<<"Number of neighbors of particle 0: "<<neighbors_acc[0]<<std::endl;
 	  


----------------------


.. toctree::
   :hidden:
   :maxdepth: 4
   :caption: First steps

   QuickStart
   Compiling
   FAQ
   Examples
   Tests



Indices and tables
==================

* :ref:`genindex`
* :ref:`modindex`
* :ref:`search`
