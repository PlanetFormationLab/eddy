"""Functions to easily build example spectra to test the approach."""

import numpy as np
from eddy.fit_annulus import annulus as annulus_class


def gaussian_ensemble(vrot, vrad=0.0, Tb=40., dV=350., tau=None, rms=0.0,
                      dV_chan=30., N=10, oversample=False, PAmin=-np.pi,
                      PAmax=np.pi, linear_sample=True, plot=True,
                      return_annulus=False):
    """
    Model an ensemble of optically thick or thin Gaussian line profiles.

    Args:
        vrot (float): Rotation velocity at the given radius [m/s].
        vrad (optional[float]): Radial velocity at the given radius [m/s].
        Tb (float): Peak brightness temperature for the lines [K].
        dV (float): Doppler linewidth in [m/s].
        tau (optional[float]): Optical depth. If no optical depth is given,
            assume a purely Gaussian profile.
        rms (optional[float]): RMS of the noise in [K].
        dV_chan (optional[float]): Channel width for the velocity axis [m/s].
        N (int): Number of spectra to generate in the ensemble.
        oversample (bool): If provided, oversample the spectral resolution by
            this amount when building the line profiles to account for
            low-resolution effects.
        PAmin (optional[float]): Minimum polar angle in [rad] to consider.
        PAmax (optional[float]): Maximum polar angle in [rad] to consider.
        linear_sample (optional[bool]): Linearlly sample the position angles or
            draw randomly from the PA range given.
        plot (optional[bool]): If true, plot the ensemble of spectra.
        return_annulus (optional[bool]): If true, return a pre-initialised
            annulus instance.

    Returns (if return_annulus is False):
        spectra (ndarray[float]): [theta.size, velax.size] shaped array of the
            spectra in [K].
        theta (ndarray[float]): Array of polar angles of spectra [rad].
        velax (ndarray[float]): Velocity axis in [m/s].

    Returns (if return_annulus is True):
        annulus (annulus instance): Pre-initialized annnulus instance.
    """

    # Take a velocity axis that is twice the shift due to the rotation and
    # include a random offset so lines are not always at the same velocity.

    velax = 2 * vrot
    velax = np.arange(-velax, velax + 1, dV_chan)
    velax += (2.0 * np.random.random() - 1.0) * np.diff(velax)[0]

    # Generate the theta sampling. Either linearly sample the
    # radian range or randomly sample between the bounds.

    if N < 2:
        raise ValueError("Must have at least two spectra.")
    if PAmin >= PAmax:
        raise ValueError("PAmin must be smaller than PAmax.")
    if linear_sample:
        theta = np.linspace(PAmin, PAmax, N + 1)[1:]
        theta += np.diff(theta)[-1] * np.random.random()
    else:
        theta = (PAmax - PAmin) * np.random.random(N) - PAmin
        theta = np.array(sorted(theta))

    # Build the line profiles.

    v_los = vrot * np.cos(theta) + vrad * np.sin(theta)
    if tau is not None:
        Tex = Tb / (1. - np.exp(-tau))
        spectra = [_thick_line(velax, v, dV, Tex, tau, N=N) for v in v_los]
        spectra = np.array(spectra)
    else:
        spectra = [_gaussian(velax, v, dV, Tb, N=N) for v in v_los]
        spectra = np.array(spectra)
    spectra += rms * np.random.randn(spectra.size).reshape(spectra.shape)

    # Plot the profiles to check they're OK and then return.

    annulus = annulus_class(spectra=spectra, theta=theta, velax=velax)
    if plot:
        annulus.plot_spectra()
    if return_annulus:
        return annulus
    return spectra, theta, velax


def flared_disk_ensemble(radius=1.0, inc=30., mstar=1.0, dist=100., Tb=40.,
                         Tb2=21., dV=350., dV2=110., rms=1., dV_chan=30., N=10,
                         PAmin=-np.pi, PAmax=np.pi, linear_sample=True,
                         aspect_ratio=0.3, flaring_angle=1.25, plot=True,
                         return_ensemble=False):
    """
    Create an ensemble of Gaussian lines with contamination from the far side
    of a flared disk (see Rosenfeld et al. 2013 for examples). Unlike
    :func:`gaussian_ensemble`, a rotation velocity is not specified directly;
    instead a disk location is defined by its radius, inclination, and stellar
    mass, and the Keplerian velocity is computed internally.

    Args:
        radius (float): Radius of the annulus in [arcsec].
        inc (float): Inclination of the disk in [degrees]. Must be non-zero.
        mstar (float): Mass of the central star in [Msun].
        dist (float): Distance to the source in [pc].
        Tb (float): Peak brightness temperature of the front-side component
            in [K].
        Tb2 (float): Peak brightness temperature of the far-side component in
            [K]. For CO this is typically the freeze-out temperature (~21 K);
            see Pinte et al. (2018a) for a discussion.
        dV (float): Doppler linewidth of the front-side component in [m/s].
        dV2 (float): Doppler linewidth of the far-side component in [m/s].
            The default corresponds to the thermal width at ~21 K.
        rms (float): RMS noise level in [K].
        dV_chan (float): Channel width of the velocity axis in [m/s].
        N (int): Number of spectra to generate in the ensemble.
        PAmin (Optional[float]): Minimum polar angle in [rad] to consider.
        PAmax (Optional[float]): Maximum polar angle in [rad] to consider.
        linear_sample (Optional[bool]): If ``True``, linearly sample polar
            angles; otherwise draw randomly from the given range.
        aspect_ratio (Optional[float]): Aspect ratio ``z/r`` of the emission
            surface at 1".
        flaring_angle (Optional[float]): Flaring exponent of the emission
            surface.
        plot (Optional[bool]): If ``True``, plot the ensemble of spectra.
        return_ensemble (Optional[bool]): If ``True``, return a
            pre-initialised annulus instance instead of raw arrays.

    Returns (if ``return_ensemble`` is ``False``):
        spectra (ndarray): Array of shape ``[N, velax.size]`` of spectra in
            [K].
        theta (ndarray): Polar angles of each spectrum in [rad].
        velax (ndarray): Velocity axis in [m/s].
        vrot (float): Keplerian rotation velocity of the front side in [m/s].

    Returns (if ``return_ensemble`` is ``True``):
        annulus (annulus instance): Pre-initialised annulus instance.
    """

    from scipy.interpolate import griddata

    # Check the inclination.
    if inc <= 0.0:
        raise ValueError("Inclination must be positive.")

    # Define the vertical height functions.

    def zfunc(r):
        return aspect_ratio * np.power(r, flaring_angle)

    # Cyclindrical coordinates of the disk midplane.

    if N < 2:
        raise ValueError("Must have at least two spectra.")
    if PAmin >= PAmax:
        raise ValueError("Must have PA_min < PA_max.")
    if linear_sample:
        tdisk = np.linspace(PAmin, PAmax, N + 1)[1:]
        tdisk += np.diff(tdisk)[-1] * np.random.random()
    else:
        tdisk = 2. * np.pi * np.random.random(N) - np.pi
        tdisk = np.array(sorted(tdisk))
    rdisk = radius * np.ones(N)
    zdisk = zfunc(rdisk)

    # Cartesian coordinates of the disk midplane.

    xdisk = rdisk * np.cos(tdisk)
    ydisk = rdisk * np.sin(tdisk)

    # Find the projected sky coordinates of these locations.

    xsky, ysky = _disk_to_sky(xdisk, ydisk, zdisk, inc)

    # Find where these sky coordinates intersect the rear side of the disk.
    # First make a grid of the cylindrical coordinates of the far side
    # of the disk then find their sky coordinates. We then interpolate this
    # using the sky coordinates of the front side to recover the midplane
    # coordaintes which are interesected. There is probably a way to do
    # this analytically but I haven't worked that out yet...

    xgrid = np.linspace(1.2 * xdisk.min(), 1.2 * xdisk.max(), 50)
    ygrid = np.linspace(5.0 * ydisk.min(), 1.2 * ydisk.max(), 50)
    xgrid, ygrid = np.meshgrid(xgrid, ygrid)
    xgrid, ygrid = xgrid.flatten(), ygrid.flatten()
    zgrid = -zfunc(np.hypot(xgrid, ygrid))
    xproj, yproj = _disk_to_sky(xgrid, ygrid, zgrid, inc)

    # Intersected cylindrical coordinates.

    xback = griddata((xproj, yproj), xgrid, (xsky, ysky))
    yback = griddata((xproj, yproj), ygrid, (xsky, ysky))
    rback, tback = np.hypot(xback, yback), np.arctan2(yback, xback)
    zback = zfunc(rback)

    # Calculate the projected velocities for the line centres.

    vrot = _keplerian_velocity(radius, zfunc(radius), 0.0, inc, dist, mstar)
    vfront = _keplerian_velocity(rdisk, zdisk, tdisk, inc, dist, mstar)
    vback = _keplerian_velocity(rback, zback, tback, inc, dist, mstar)

    print("Rotation velocity: %d m/s." % vrot)

    # Generate the line ensemble, similar to `gaussian_ensemble()`.

    velax = 2 * vrot
    velax = np.arange(-velax, velax + 1, dV_chan)
    velax += (2.0 * np.random.random() - 1.0) * np.diff(velax)[0]

    # Include the emission from the far side of the disk. We take the maximum
    # value to mimic the optically thick front side blocking the rear.

    spectra_f = _gaussian(velax[None, :], vfront[:, None], dV, Tb)
    spectra_b = _gaussian(velax[None, :], vback[:, None], dV2, Tb2)
    spectra = np.max([spectra_f, spectra_b], axis=0)
    spectra += rms * np.random.randn(spectra.size).reshape(spectra.shape)

    # Plot the profiles to check they're OK and then return.

    annulus = annulus_class(spectra=spectra, theta=tdisk, velax=velax)
    if plot:
        annulus.plot_spectra()
    if return_ensemble:
        return annulus
    return spectra, tdisk, velax, vrot


def _gaussian(x, x0, dx, A, N=False):
    """Simple Gaussian function with oversampling."""
    if not N:
        return A * np.exp(-np.power((x - x0) / dx, 2))
    xx = np.linspace(x[0], x[-1], x.size * int(N))
    yy = A * np.exp(-np.power((xx - x0) / dx, 2))
    return np.array([np.average(yy[i*N:(i+1)*N]) for i in range(x.size)])


def _thick_line(x, x0, dx, Tex, tau, N=False):
    """Optically thick line profile."""
    if not N:
        return Tex * (1. - np.exp(-_gaussian(x, x0, dx, tau, N=1)))
    xx = np.linspace(x[0], x[-1], x.size * int(N))
    yy = Tex * (1. - np.exp(-_gaussian(xx, x0, dx, tau, N=False)))
    return np.array([np.average(yy[i*N:(i+1)*N]) for i in range(x.size)])


def _disk_to_sky(xdisk, ydisk, zdisk, inc):
    """Convert disk coordinates to sky coordinates."""
    t = zdisk / np.sin(np.radians(inc)) if abs(inc) > 0.0 else 0.0
    ysky = ydisk - t * np.sin(np.radians(inc))
    return xdisk, ysky * np.cos(np.radians(inc))


def _keplerian_velocity(rdisk, zdisk, tdisk, inc, dist, mstar):
    """Projected Keplerian velocity in [m/s]."""
    import scipy.constants as sc
    vkep = sc.G * mstar * 1.988e30 * np.power(rdisk * dist * sc.au, 2.0)
    vkep /= np.power(np.hypot(rdisk, zdisk) * dist * sc.au, 3.0)
    return np.sqrt(vkep) * np.sin(np.radians(inc)) * np.cos(tdisk)
