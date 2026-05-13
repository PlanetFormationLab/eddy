# eddy - Extracting Disk Dynamics

<p align='center'>
  <br/>
  <img src="https://github.com/PlanetFormationLab/eddy/blob/master/docs/_static/eddy_logo.png" width="500" height="500">
  <br/>
  <br/>
  <a href="https://pypi.org/project/astro-eddy/"><img src="https://img.shields.io/pypi/v/astro-eddy.svg" alt="PyPI version" />
  </a>
  <a href="https://pypi.org/project/astro-eddy/"><img src="https://img.shields.io/pypi/pyversions/astro-eddy.svg" alt="Python versions" />
  </a>
  <a href="http://joss.theoj.org/papers/2868c5ad4b6405eba1aaf1cd8ea53274"><img src="http://joss.theoj.org/papers/2868c5ad4b6405eba1aaf1cd8ea53274/status.svg">
  </a>
  <a href='https://eddy.readthedocs.io/en/latest/?badge=latest'><img src='https://readthedocs.org/projects/eddy/badge/?version=latest' alt='Documentation Status' />
  </a>
  <a href="https://doi.org/10.5281/zenodo.1440051"><img src="https://zenodo.org/badge/DOI/10.5281/zenodo.1440051.svg" alt="DOI" />
  </a>
  <a href="http://ascl.net/1901.010"><img src="https://img.shields.io/badge/ascl-1901.010-blue.svg?colorB=262255" alt="ascl:1901.010" />
  </a>
  <br/>
  <br/>
</p>

## What Is It?

`eddy` is a suite of Python tools to recover precise velocity profiles of protoplanetary disks from Doppler shifted line emission. `eddy` makes fitting of first moment maps and the inference of a rotation velocity from an annulus of spectra a breeze.

As of v3.0.0, the rotation-map model and likelihood are JAX-backed: JIT-compiled, autodifferentiable, and GPU-aware. Sampling can be done either with [`emcee`](https://github.com/dfm/emcee) (the default) or [`numpyro`](https://github.com/pyro-ppl/numpyro) NUTS via `mcmc='numpyro'`; the Gaussian-process annulus path uses [`tinygp`](https://github.com/dfm/tinygp).

## Installation

The most simple method is with `pip`,

```
pip install astro-eddy
```

Runtime dependencies are pulled in automatically: `numpy`, `scipy`, `matplotlib`, `jax`, `jaxlib`, `emcee`, `zeus-mcmc`, `numpyro`, `tinygp`, `corner`, and `bettermoments`. Python 3.10 or newer.

For an editable development install,

```
git clone https://github.com/PlanetFormationLab/eddy.git
cd eddy
pip install -e ".[test]"   # add test extras for running pytest
```

If things have installed correctly you should be able to run the [Jupyter Notebooks](https://github.com/PlanetFormationLab/eddy/tree/master/docs/tutorials) with no errors. If something goes wrong, please [open an issue](https://github.com/PlanetFormationLab/eddy/issues/new).

## Useage

For guides on how to use `eddy` you will find extensive examples in the [documents](https://github.com/PlanetFormationLab/eddy/tree/master/docs). We shamelessly recommend [bettermoments](https://github.com/richteague/bettermoments) to make the moment maps required for the fitting.

## Citations

If you use `eddy` as part of your research, please cite the [JOSS article](http://joss.theoj.org/papers/10.21105/joss.01220):

```latex
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
```

A full list of citations including dependencies can be found on the [citations](./docs/citations.md) page.

## Works Using `eddy`

A list of the works using (or at least citing) `eddy` can be found on [ADS](https://ui.adsabs.harvard.edu/abs/2019JOSS....4.1220T/citations).
