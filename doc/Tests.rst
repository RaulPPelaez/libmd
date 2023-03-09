Tests
======

Tests are written using `Google test <https://github.com/google/googletest>`. CMake will automatically clone and compile this testsuite.

To run the tests start by :ref:`compiling the project <Compiling>`, then you can run

.. code:: bash

   $ ctest

or

.. code:: bash

   $ make test


or

.. code:: bash

   $./tests/libmd_tests
