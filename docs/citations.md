# Citations

If you use `eddy`, please include the reference to the JOSS [article](http://joss.theoj.org/papers/10.21105/joss.01220) describing the code:

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

In addition, if you used any of the fitting routines then please also cite the following dependencies, without which `eddy` couldn't work.

```latex
@article{emcee, % for the MCMC sampler
    author = {{Foreman-Mackey}, D. and {Hogg}, D.~W. and {Lang}, D. and {Goodman}, J.},
    title = {emcee: The MCMC Hammer},
    journal = {PASP},
    year = 2013,
    volume = 125,
    pages = {306-312},
    eprint = {1202.3665},
    doi = {10.1086/670067}
}

% The GP backend is `tinygp`, the JAX-native successor to `celerite`. The
% original `celerite` paper introduces the quasi-separable Matern-3/2 kernel
% still in use, so it remains worth citing alongside the `tinygp` software
% reference.
@article{celerite,
    author = {{Foreman-Mackey}, D. and {Agol}, E. and {Angus}, R. and
              {Ambikasaran}, S.},
    title = {Fast and scalable Gaussian process modeling
             with applications to astronomical time series},
    year = {2017},
    journal = {AJ},
    volume = {154},
    pages = {220},
    doi = {10.3847/1538-3881/aa9332},
    url = {https://arxiv.org/abs/1703.09710}
}

@software{tinygp, % for the GP annulus path
    author = {Foreman-Mackey, Dan and others},
    title = {{tinygp: The tiniest of Gaussian Process libraries}},
    url = {https://github.com/dfm/tinygp},
    doi = {10.5281/zenodo.7269074}
}

@software{numpyro, % for mcmc='numpyro' (NUTS) in fit_map / get_vlos
    author = {Phan, Du and Pradhan, Neeraj and Jankowiak, Martin},
    title = {{Composable Effects for Flexible and Accelerated Probabilistic
              Programming in NumPyro}},
    year = {2019},
    eprint = {1912.11554}
}

@software{jax, % for the JAX backend
    author = {Bradbury, James and others},
    title = {{JAX: composable transformations of Python+NumPy programs}},
    url = {http://github.com/google/jax},
    year = {2018}
}

@article{corner, % for the covariance plots
    doi = {10.21105/joss.00024},
    url = {https://doi.org/10.21105/joss.00024},
    year  = {2016},
    month = {jun},
    publisher = {The Open Journal},
    volume = {1},
    number = {2},
    pages = {24},
    author = {Daniel Foreman-Mackey},
    title = {corner.py: Scatterplot matrices in Python},
    journal = {The Journal of Open Source Software}
}
```
