import sys
import functools

sys.path.append('../../')
sys.path.append('../../../')

import matplotlib.pyplot as plt
import numpy as np
from pyscf import gto, dft, lib, cc, scf
from pyscf.dft import numint

from exact_conds import CondChecker

import utils

pi = np.pi


def radial_reduced_grad_dist(
    grids,
    n,
    n_grad,
    s_grids=np.linspace(0, 3, num=1000),
    fermi_temp=0.05,
    density_tol=1e-6,
):
  """ Obtain distribution of the reduced gradient, g_3(s) as defined in:
    
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
  int_g3_s = np.sum(integrand, axis=1)

  s_grids = np.squeeze(s_grids, axis=1)
  g3_s = num_deriv_fn(int_g3_s, s_grids)

  return s_grids, g3_s


def num_deriv_fn(arr, grids):
  """ Numerical 1st derivative of arr on grids."""
  dx = (grids[-1] - grids[0]) / (len(grids) - 1)
  deriv = np.gradient(arr, dx, edge_order=2, axis=0)
  return deriv


class GedankenDensity():

  def decay_tail(grids, b, x, y, y_prime):
    """ decay tail f(x) = A x^b e^(-kx) with f(x) = y and f'(x) = y' """

    k = ((-y_prime * x / y) + b) / x
    A = y / (x**b * np.exp(-k * x))

    return A * grids**b * np.exp(-k * grids)

  @staticmethod
  def monster(
      gamma,
      r_s_min,
      r_s_max,
      s_target,
      num_peaks,
      smoothing_factor,
      base_grid_pts=1000,
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

    n_m = np.concatenate((n_osc, tail[1:]), axis=0)
    n_m_grad = np.concatenate((grad_n_osc, tail_deriv[1:]), axis=0)

    grids = np.concatenate((grids_1, grids_2[1:]), axis=0)

    # easy rescaling
    grids /= gamma
    n_m *= gamma**3
    n_m_grad *= gamma**4

    return grids, n_m, n_m_grad

  def get_eps_c(func_c, gamma, density):

    grids, n_m, n_m_grad = density(gamma=gamma)

    inp = {}
    inp["rho"] = n_m
    inp["sigma"] = n_m_grad**2

    func_c_res = func_c.compute(inp)
    eps_c = np.squeeze(func_c_res['zk'])

    return eps_c

  def get_E_c_gam(func_c, gamma, density):

    grids, n_m, n_m_grad = density(gamma=gamma)

    inp = {}
    inp["rho"] = n_m
    inp["sigma"] = n_m_grad**2

    func_c_res = func_c.compute(inp)
    eps_c = np.squeeze(func_c_res['zk'])

    int_check = 4 * pi * np.trapz(n_m * (grids**2), grids)
    E_c_gam = 4 * pi * np.trapz(eps_c * n_m * (grids**2), grids)

    return E_c_gam, int_check

  def run_example():
    density = functools.partial(GedankenDensity.monster,
                                r_s_min=1.5,
                                r_s_max=2,
                                s_target=2,
                                num_peaks=5,
                                smoothing_factor=0.05)

    grids, n_m, n_m_grad = density(gamma=1)

    s_grids, g3_s = radial_reduced_grad_dist(
        grids,
        n_m,
        n_m_grad,
        s_grids=np.linspace(0, 5, num=1000),
    )

    plt.plot(s_grids, g3_s, label='$g_3(s)$')

    plt.xlabel('$s$')
    plt.legend(loc='lower right')
    plt.savefig(f'g3_s_gedanken.pdf', bbox_inches='tight')


class OtherExamples():

  def li_atom_g3_s():

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

    s_grids, g3_s = checker.reduced_grad_dist()

    plt.plot(s_grids, g3_s)
    plt.savefig('g3_s_li_atom.pdf', bbox_inches='tight')

  def ar_atom_g3_s():
    ar_atom = gto.M(
        atom='Ar 0 0 0',
        basis='aug-pcseg-4',
        verbose=4,
    )

    mf = scf.RHF(ar_atom)
    mf.conv_tol_grad = 1e-9
    mf.kernel()

    checker = CondChecker(mf, xc='HF')

    s_grids, g3_s = checker.reduced_grad_dist()

    plt.plot(s_grids, g3_s)
    plt.savefig('g3_s_ar_atom.pdf', bbox_inches='tight')


if __name__ == '__main__':
  #GedankenDensity.run_example()
  OtherExamples.ar_atom_g3_s()