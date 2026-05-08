eddy - Extracting Disk Dynamics
===============================

*eddy* provides a suite of tools to extract kinematical information from
spatially and spectrally resolved line data.

The :class:`eddy.rotationmap.rotationmap` class enables the fitting of a Keplerian
rotation pattern to a rotation map to infer geometrical properties of the disk,
a dynamical mass of the star, or to infer the presence of a flared emission
surface or warp in the disk.


Installation
------------

Installation is as easy as,

.. code-block:: bash

    pip install astro-eddy

If you want to live on the edge, you can clone the repository and install that,

.. code-block:: bash

    pip install .

but I cannot promise everything will work as described here.


Contents
--------

.. toctree::
   :maxdepth: 2
   :caption: Tutorials

   tutorials/tutorial_0
   tutorials/tutorial_1
   tutorials/tutorial_2
   tutorials/tutorial_3
   tutorials/tutorial_4
   tutorials/tutorial_5
   tutorials/tutorial_6_numpyro


Support
-------

If you are having issues, please open a `issue`_ on the GitHub page.


Attribution
-----------

If you use `eddy` in any of your work, please cite the `JOSS article`_,

::

    @article{eddy,
        doi = {10.21105/joss.01220},
        url = {https://doi.org/10.21105/joss.01220},
        year = {2019},
        month = {feb},
        publisher = {The Open Journal},
        volume = {4},
        number = {34},
        pages = {1220},
        author = {Richard Teague},
        title = {eddy},
        journal = {The Journal of Open Source Software}
    }


License
-------

The project is licensed under the `MIT license`_.


.. _emcee: https://github.com/dfm/emcee
.. _issue: https://github.com/richteague/eddy/issues
.. _MIT license: https://github.com/richteague/eddy/blob/master/LICENSE.md
.. _JOSS article: http://joss.theoj.org/papers/10.21105/joss.01220
