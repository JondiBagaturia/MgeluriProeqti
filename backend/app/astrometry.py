"""Parallax-aware astrometric fit.

Model, per coordinate, in the tangent plane (arcsec):

    east(t)  = east0  + mu_e * (t - t0) + plx * f_e(t)
    north(t) = north0 + mu_n * (t - t0) + plx * f_n(t)

f_e, f_n are the parallax factors: the displacement, for a parallax of 1",
caused by the observer's position relative to the solar-system barycentre
at time t. With (X, Y, Z) the observer's barycentric ICRS position in au:

    f_e =  X sin(ra) - Y cos(ra)
    f_n =  X cos(ra) sin(dec) + Y sin(ra) sin(dec) - Z cos(dec)

Observer: the Earth's centre (astropy's built-in ephemeris). SPHEREx is in
low Earth orbit, ~7000 km from it: 5e-5 au, i.e. a 0.005% error on the
factors, irrelevant here.

The fit is linear in its five parameters and solved by weighted least
squares. Whether to BELIEVE the parallax is a separate decision, made by
`parallax_is_constrained`:

  * at least 4 epochs (5 parameters, 2 measurements per epoch, so 3 epochs
    leave one degree of freedom and cannot expose a bad epoch);
  * the epochs must sample the parallax ellipse: the parallax factors must
    not be reproducible by a straight line in time (otherwise parallax and
    proper motion are degenerate). Measured by how much of the factors'
    variance survives after removing a linear trend;
  * parallax at least 3 times its uncertainty.

If any of these fails the caller should keep the plain proper-motion fit
(app/discovery.py: fit_motion) and say why. SPHEREx revisits a position
every ~6 months near quadrature, where the factors are large and alternate
in sign, so the geometry is favourable; what limits it is the positional
error per epoch (see docs/PHASE2.md for the numbers on real data).
"""
from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Optional

import numpy as np
from astropy.coordinates import get_body_barycentric
from astropy.time import Time

DAYS_PER_YEAR = 365.25


def parallax_factors(times_mjd, ra_deg: float, dec_deg: float) -> tuple[np.ndarray, np.ndarray]:
    t = Time(np.asarray(times_mjd, float), format="mjd", scale="tdb")
    pos = get_body_barycentric("earth", t)
    x, y, z = pos.x.to_value("au"), pos.y.to_value("au"), pos.z.to_value("au")
    a, d = math.radians(ra_deg), math.radians(dec_deg)
    f_e = x * math.sin(a) - y * math.cos(a)
    f_n = x * math.cos(a) * math.sin(d) + y * math.sin(a) * math.sin(d) - z * math.cos(d)
    return np.atleast_1d(f_e), np.atleast_1d(f_n)


@dataclass
class ParallaxFit:
    n_epochs: int
    t0_mjd: float
    east0_arcsec: float
    north0_arcsec: float
    pmra_mas_yr: float
    pmdec_mas_yr: float
    parallax_mas: float
    pmra_err_mas_yr: float
    pmdec_err_mas_yr: float
    parallax_err_mas: float
    chi2: float
    dof: int
    factor_independence: float      # 0 = parallax fully degenerate with linear motion, 1 = fully independent
    errors_rescaled: bool = False
    constrained: bool = False
    reason: str = ""


def fit_motion_parallax(times_mjd, east, north, sigma, ra_deg: float, dec_deg: float) -> Optional[ParallaxFit]:
    t = np.asarray(times_mjd, float)
    e, n, s = np.asarray(east, float), np.asarray(north, float), np.asarray(sigma, float)
    if len(t) < 3:
        return None
    w = 1.0 / s ** 2
    t0 = float(np.sum(w * t) / np.sum(w))
    tau = (t - t0) / DAYS_PER_YEAR
    f_e, f_n = parallax_factors(t, ra_deg, dec_deg)

    # unknowns: east0, north0, mu_e, mu_n, plx
    k = len(t)
    A = np.zeros((2 * k, 5))
    A[:k, 0] = 1; A[:k, 2] = tau; A[:k, 4] = f_e
    A[k:, 1] = 1; A[k:, 3] = tau; A[k:, 4] = f_n
    y = np.concatenate([e, n])
    sw = np.sqrt(np.concatenate([w, w]))
    Aw, yw = A * sw[:, None], y * sw
    try:
        cov = np.linalg.inv(Aw.T @ Aw)
    except np.linalg.LinAlgError:
        return None
    p = cov @ (Aw.T @ yw)
    resid = yw - Aw @ p
    chi2 = float(resid @ resid)
    dof = 2 * k - 5
    rescaled = False
    if dof > 0 and chi2 / dof > 1.0:
        cov = cov * (chi2 / dof)
        rescaled = True
    err = np.sqrt(np.diag(cov))

    # How much of the parallax signal is NOT absorbable by offset + linear motion.
    def unexplained(f):
        B = np.vstack([np.ones(k), tau]).T * np.sqrt(w)[:, None]
        coef, *_ = np.linalg.lstsq(B, f * np.sqrt(w), rcond=None)
        r = f * np.sqrt(w) - B @ coef
        return float(r @ r), float(np.sum(w * f ** 2))
    ue, te = unexplained(f_e)
    un, tn = unexplained(f_n)
    independence = (ue + un) / (te + tn) if (te + tn) > 0 else 0.0

    return ParallaxFit(
        n_epochs=k, t0_mjd=t0, east0_arcsec=float(p[0]), north0_arcsec=float(p[1]),
        pmra_mas_yr=float(p[2] * 1000), pmdec_mas_yr=float(p[3] * 1000), parallax_mas=float(p[4] * 1000),
        pmra_err_mas_yr=float(err[2] * 1000), pmdec_err_mas_yr=float(err[3] * 1000), parallax_err_mas=float(err[4] * 1000),
        chi2=chi2, dof=dof, factor_independence=float(independence), errors_rescaled=rescaled,
    )


def parallax_is_constrained(fit: Optional[ParallaxFit], min_epochs: int = 4, min_independence: float = 0.2,
                            min_snr: float = 3.0) -> tuple[bool, str]:
    if fit is None:
        return False, "fewer than 3 epochs: parallax cannot be fitted at all"
    if fit.n_epochs < min_epochs:
        return False, f"only {fit.n_epochs} epochs: a 5-parameter fit needs at least {min_epochs} to check itself"
    if fit.factor_independence < min_independence:
        return False, (f"the epochs do not sample the parallax ellipse (only {100 * fit.factor_independence:.0f}% of the "
                       "parallax signal is distinguishable from straight-line motion)")
    if fit.parallax_err_mas <= 0 or abs(fit.parallax_mas) / fit.parallax_err_mas < min_snr:
        return False, (f"parallax {fit.parallax_mas:.0f} +- {fit.parallax_err_mas:.0f} mas is less than "
                       f"{min_snr:.0f} sigma from zero")
    if fit.parallax_mas < 0:
        return False, "the fitted parallax is negative, which is unphysical"
    return True, "parallax is constrained"


def expected_parallax_error_mas(times_mjd, sigma_arcsec: float, ra_deg: float, dec_deg: float) -> Optional[float]:
    """Formal parallax uncertainty for given epochs and per-epoch positional
    error: what the geometry alone allows, before looking at any data."""
    t = np.asarray(times_mjd, float)
    fit = fit_motion_parallax(t, np.zeros(len(t)), np.zeros(len(t)), np.full(len(t), sigma_arcsec), ra_deg, dec_deg)
    return None if fit is None else fit.parallax_err_mas
