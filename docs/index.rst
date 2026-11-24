eddy - Extracting Disk Dynamics
===============================

.. raw:: html

   <p class="eddy-lede">Fit Keplerian rotation patterns to spatially and
      spectrally resolved line data — and measure the gas motions that give
      away a forming planet.</p>
   <div class="eddy-actions">
     <a class="eddy-btn eddy-btn--primary" href="#installation">Install eddy</a>
     <a class="eddy-btn eddy-btn--ghost" href="tutorials/tutorial_1.html">Start the tutorials</a>
     <code class="literal">pip install astro-eddy</code>
   </div>

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
   :caption: Velocity Maps

   faq/velocity_maps
   tutorials/tutorial_1
   tutorials/tutorial_2
   tutorials/tutorial_5

.. toctree::
   :maxdepth: 2
   :caption: Annuli

   faq/annuli
   tutorials/tutorial_3
   tutorials/tutorial_4

.. toctree::
   :maxdepth: 2
   :caption: Structure Functions

   faq/structure_functions
   tutorials/structure_functions_coming_soon


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
