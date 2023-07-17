import numpy as np
from scipy.interpolate import interp1d
from astropy import units as U
from astropy.cosmology import WMAP7


def z_to_t(z_in, CP=WMAP7):
    theta = np.sqrt(1 - CP.Om0) \
        * np.power(CP.Om0 * np.power(1.0 + z_in, 3.0)
                   + (1 - CP.Om0), -0.5)
    return np.power(CP.H0, -1).to(U.Gyr) \
        * np.power(3.0 * np.sqrt(1 - CP.Om0), -1) \
        * np.log((1 + theta) / (1 - theta))


def t_to_z(t_in, CP=WMAP7, z_max=25.0, z_min=0.0):
    n_table = (z_max - z_min) * 100.
    z_table = np.linspace(z_max, z_min, int(n_table))
    t_table = z_to_t(z_table, CP=CP)
    if np.min(t_table) > np.min(t_in):
        return t_to_z(t_in, CP=CP, z_max=z_max * 10., z_min=z_min)
    if np.max(t_table) < np.max(t_in):
        return t_to_z(t_in, CP=CP, z_max=z_max, z_min=(z_min - 1) * 10)
    return interp1d(
        t_table,
        z_table,
        assume_sorted=True,
        bounds_error=False,
        fill_value=np.nan,
        copy=False
    )(t_in.to(U.Gyr))


def z_to_a(z_in, CP=WMAP7):
    return 1.0 / (1.0 + z_in)


def a_to_z(a_in, CP=WMAP7):
    return 1.0 / a_in - 1.0


def lb_to_t(lb_in, CP=WMAP7):
    age = z_to_t(0.0, CP=CP)
    return age - lb_in


def t_to_lb(t_in, CP=WMAP7):
    age = z_to_t(0.0, CP=CP)
    return age - t_in


def t_to_a(t_in, CP=WMAP7):
    retval = z_to_a(t_to_z(t_in, CP=CP), CP=CP)
    retval[t_in == 0.0] = 0.0
    return retval


def a_to_t(a_in, CP=WMAP7):
    return z_to_t(a_to_z(a_in, CP=CP), CP=CP)


def lb_to_a(lb_in, CP=WMAP7):
    return t_to_a(lb_to_t(lb_in, CP=CP), CP=CP)


def a_to_lb(a_in, CP=WMAP7):
    return t_to_lb(a_to_t(a_in, CP=CP), CP=CP)


def lb_to_z(lb_in, CP=WMAP7):
    return t_to_z(lb_to_t(lb_in, CP=CP), CP=CP)


def z_to_lb(z_in, CP=WMAP7):
    return t_to_lb(z_to_t(z_in, CP=CP), CP=CP)
