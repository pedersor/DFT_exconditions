import sys
import functools

sys.path.append('../../')
sys.path.append('../../../')

import pylibxc
import matplotlib.pyplot as plt
import numpy as np
from pyscf import gto, dft, lib, cc, scf
from pyscf.dft import numint

from exact_conds import CondChecker
import utils

pi = np.pi


class GedankenDensity():

  def radial_reduced_grad_dist(
      grids,
      n,
      n_grad,
      s_grids=np.linspace(0, 3, num=1000),
      fermi_temp=0.05,
      density_tol=1e-6,
  ):
    """ Obtain distribution of the reduced gradient, g(s) as defined in:
      
      Zupan, Ales, et al. "Density-gradient analysis for density functional 
      theory: Application to atoms." International journal of quantum chemistry 
      61.5 (1997): 835-845.

      https://doi.org/10.1002/(SICI)1097-461X(1997)61:5<835::AID-QUA9>3.0.CO;2-X
      
      """

    dx = (grids[-1] - grids[0]) / (len(grids) - 1)
    s_grids = np.expand_dims(s_grids, axis=1)
    s = utils.get_s(n, n_grad)

    # avoid numerical problems
    mask = n > density_tol
    n = n[mask]
    s = s[mask]
    grids = grids[mask]

    fermi_dist = 1 / (np.exp(-(s_grids - s) / fermi_temp) + 1)

    integrand = 4 * np.pi * grids**2 * np.nan_to_num(n * fermi_dist) * dx
    int_g_s = np.sum(integrand, axis=1)

    s_grids = np.squeeze(s_grids, axis=1)
    g_s = GedankenDensity.num_deriv_fn(int_g_s, s_grids)

    return s_grids, g_s

  def num_deriv_fn(arr, grids):
    """ Numerical 1st derivative of arr on grids."""
    dx = (grids[-1] - grids[0]) / (len(grids) - 1)
    deriv = np.gradient(arr, dx, edge_order=2, axis=0)
    return deriv

  def gedanken_density(
      gamma,
      r_s_min,
      r_s_max,
      s_target,
      num_peaks,
      smoothing_factor,
      base_grid_pts=1000,
      num_elec=1,
  ):

    n_max = utils.get_density(r_s_min)
    n_min = utils.get_density(r_s_max)
    grad_n = utils.get_grad_n(s_target, (n_max + n_min) / 2)

    # parameterized density
    amp = n_max - n_min
    offset = n_max
    a = smoothing_factor
    period = 2 * amp * (1 - a) / grad_n

    # grids used in oscillatory region
    osc_len = (num_peaks - 3 / 4) * period
    grids_1 = np.linspace(0, osc_len, base_grid_pts)

    def osc_density(grids, offset, amp, a, period):
      """ Oscillatory region of the density. """

      theta = (1 / period) * 2 * pi * (grids + (period / 4))

      osc_density = offset - amp * np.arccos((1 - a) * np.sin(theta)) / pi
      deriv_osc_density_1 = 2 * amp * (1 - a) * np.cos(theta)
      deriv_osc_density_2 = period * np.sqrt(1 - (1 - a)**2 * np.sin(theta)**2)
      deriv_osc_density = deriv_osc_density_1 / deriv_osc_density_2

      return osc_density, deriv_osc_density

    n_osc, grad_n_osc = osc_density(grids_1, offset, amp, a, period)

    # grids used in decay region
    # (use 2x the length of osc. region)
    grids_2 = np.linspace(osc_len, 3 * osc_len, base_grid_pts * 2)

    def decay_tail(grids, b, x, y, y_prime):
      """ decay tail f(x) = A x^b e^(-kx) with f(x) = y and f'(x) = y' """

      k = ((-y_prime * x / y) + b) / x
      A = y / (x**b * np.exp(-k * x))

      decay_tail = A * grids**b * np.exp(-k * grids)
      decay_tail_deriv = A * np.exp(
          -k * grids) * grids**(-1 + b) * (b - k * grids)

      return decay_tail, decay_tail_deriv

    tail, tail_deriv = decay_tail(grids_2,
                                  b=-2,
                                  x=grids_1[-1],
                                  y=n_osc[-1],
                                  y_prime=-grad_n)

    n_g = np.concatenate((n_osc, tail[1:]), axis=0)
    n_g_grad = np.concatenate((grad_n_osc, tail_deriv[1:]), axis=0)

    grids = np.concatenate((grids_1, grids_2[1:]), axis=0)

    # easy rescaling
    grids /= gamma
    n_g *= gamma**3
    n_g_grad *= gamma**4

    # normalize to num_elec
    norm = 4 * pi * np.trapz(n_g * (grids**2), grids)
    n_g *= num_elec / norm
    n_g_grad *= num_elec / norm

    return grids, n_g, n_g_grad

  def default_gedanken_density():
    """ Default gedanken density parameters. 
  
    Returns: callable gedanken density (gamma).
    """
    density = functools.partial(
        GedankenDensity.gedanken_density,
        r_s_min=1.5,
        r_s_max=2,
        s_target=2,
        num_peaks=5,
        smoothing_factor=0.05,
    )
    return density

  def get_e_xc(xc, gamma):
    """ Return E_xc[n^gedanken_\gamma] for a given XC functional. """

    gdn_density = GedankenDensity.default_gedanken_density()
    grids, n_g, n_g_grad = gdn_density(gamma=gamma)

    func = pylibxc.LibXCFunctional(xc, "unpolarized")
    inp = {}
    inp["rho"] = n_g
    inp["sigma"] = n_g_grad**2
    eps_xc = func.compute(inp)
    eps_xc = np.squeeze(eps_xc['zk'])

    # check normalization
    int_check = 4 * pi * np.trapz(n_g * (grids**2), grids)
    e_xc = 4 * pi * np.trapz(eps_xc * n_g * (grids**2), grids)

    return e_xc

  def gedanken_g_s():
    density = functools.partial(GedankenDensity.gedanken_density,
                                r_s_min=1.5,
                                r_s_max=2,
                                s_target=2,
                                num_peaks=5,
                                smoothing_factor=0.05)

    grids, n_g, n_g_grad = density(gamma=1)

    s_grids, g_s = GedankenDensity.radial_reduced_grad_dist(
        grids,
        n_g,
        n_g_grad,
        s_grids=np.linspace(0, 5, num=1000),
    )

    return s_grids, g_s

  def plot_gedanken_density():

    density = functools.partial(GedankenDensity.gedanken_density,
                                r_s_min=1.5,
                                r_s_max=2,
                                s_target=2,
                                num_peaks=5,
                                smoothing_factor=0.05)

    grids, n_g, n_g_grad = density(gamma=1)

    utils.use_standard_plotting_params()
    plt.plot(grids, n_g, label='gedanken density')
    plt.ylabel('$n(r)$')
    plt.xlabel('$r$')
    plt.ylim(0, .1)
    plt.xlim(left=0, right=3)
    plt.legend(loc='upper right')
    plt.savefig('gedanken_density.pdf', bbox_inches='tight')
    plt.close()


class Examples():

  s_grids = np.linspace(0, 5, num=1000)

  def li_atom_g_s():

    li_atom = gto.M(
        atom='Li 0 0 0',
        basis='aug-pcseg-4',
        spin=1,
        verbose=4,
    )

    mf = scf.UHF(li_atom)
    mf.conv_tol_grad = 1e-9
    mf.kernel()

    checker = CondChecker(mf, xc='HF')

    s_grids, g_s = checker.reduced_grad_dist(s_grids=Examples.s_grids)

    return s_grids, g_s

  def n_atom_g_s():

    n_atom = gto.M(
        atom='N 0 0 0',
        basis='aug-pcseg-4',
        spin=3,
        verbose=4,
    )

    mf = scf.UHF(n_atom)
    mf.conv_tol_grad = 1e-9
    mf.kernel()

    checker = CondChecker(mf, xc='HF')

    s_grids, g_s = checker.reduced_grad_dist(s_grids=Examples.s_grids)

    return s_grids, g_s

  def ar_atom_g_s():
    ar_atom = gto.M(
        atom='Ar 0 0 0',
        basis='aug-pcseg-4',
        verbose=4,
    )

    mf = scf.RHF(ar_atom)
    mf.conv_tol_grad = 1e-9
    mf.kernel()

    checker = CondChecker(mf, xc='HF')

    s_grids, g_s = checker.reduced_grad_dist(s_grids=Examples.s_grids)

    return s_grids, g_s

  def combined_examples():

    n_out = Examples.n_atom_g_s()
    ar_out = Examples.ar_atom_g_s()
    gedanken_out = GedankenDensity.gedanken_g_s()

    # normalize g_s across different systems
    n_out = (n_out[0], n_out[1] / 7)
    ar_out = (ar_out[0], ar_out[1] / 18)
    gedanken_out = (gedanken_out[0], gedanken_out[1] / 1)

    s_min = np.min([n_out[0], ar_out[0], gedanken_out[0]])
    s_max = np.max([n_out[0], ar_out[0], gedanken_out[0]])

    plt.plot(*gedanken_out, label='gedanken')
    plt.plot(*n_out, label='N atom')
    plt.plot(*ar_out, label='Ar atom')

    utils.use_standard_plotting_params()
    plt.legend(loc='upper right')
    plt.xlim(left=s_min, right=s_max)
    plt.ylim(bottom=0)
    plt.ylabel('$g(s) / N$')
    plt.xlabel('$s$')
    plt.savefig('g_s_combined.pdf', bbox_inches='tight')
    plt.close()


if __name__ == '__main__':
  """ Obtain plots and results for the paper. """

  e_c_gdn_density_lyp = GedankenDensity.get_e_xc('gga_c_lyp', gamma=1)
  print(f'E_c[n^gedanken] = {e_c_gdn_density_lyp}')
  GedankenDensity.plot_gedanken_density()
  Examples.combined_examples()
