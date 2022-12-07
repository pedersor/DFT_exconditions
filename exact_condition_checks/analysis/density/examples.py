import sys
import functools

sys.path.append('../../')
sys.path.append('../../../')

import pylibxc
import matplotlib.pyplot as plt
import numpy as np
from pyscf import gto, dft, lib, cc, scf
from pyscf.dft import numint
from scipy.optimize import fsolve

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

  def num_deriv2_fn(arr, grids):
    """ Numerical 2nd derivative of arr on grids."""
    dx = (grids[-1] - grids[0]) / (len(grids) - 1)
    deriv2 = np.diff(arr, 2, axis=0) / (dx**2)
    return deriv2

  def gedanken_density(
      gamma,
      r_s_min,
      r_s_max,
      s_target,
      num_peaks,
      smoothing_factor,
      base_grid_pts=1000,
      num_elec=2,
  ):

    # desired max and min density values and max \grad n value
    # in oscillating region.
    n_max = utils.get_density(r_s_min)
    n_min = utils.get_density(r_s_max)
    grad_n = utils.get_grad_n(s_target, (n_max + n_min) / 2)

    # derived density parameters
    amp = n_max - n_min
    offset = n_max
    eta = smoothing_factor
    period = 2 * amp * (1 - eta) / grad_n

    # grids used in oscillatory region
    osc_len = (num_peaks - 3 / 4) * period
    grids = np.linspace(0, 3 * osc_len, num=3 * base_grid_pts)
    grids_1, grids_2 = np.split(grids, [base_grid_pts])

    def osc_density(grids, offset, amp, eta, period):
      """ Oscillatory region of the density. """

      theta = (1 / period) * 2 * pi * (grids + (period / 4))

      osc_density = offset - amp * np.arccos((1 - eta) * np.sin(theta)) / pi
      deriv_osc_density_1 = 2 * amp * (1 - eta) * np.cos(theta)
      deriv_osc_density_2 = period * np.sqrt(1 -
                                             (1 - eta)**2 * np.sin(theta)**2)
      deriv_osc_density = deriv_osc_density_1 / deriv_osc_density_2

      return osc_density, deriv_osc_density

    n_osc, grad_n_osc = osc_density(grids_1, offset, amp, eta, period)

    def decay_tail(grids, x, y, y_prime, y_2prime):
      """ decay tail f(x) = c e^(a x^2 + b x) with f(x) = y, f'(x) = y', 
      and f''(x) = y'' """

      def non_linear_eqs(inp):
        """ Solve set of non-linear equations to obtain decay tail parameters 
        by ensuring continuity of 0th, 1st, and 2nd derivatives. """

        a, b, c = inp
        gauss = c * np.exp(a * x**2 + b * x)
        gauss_prime = (b + 2 * a * x) * gauss
        gauss_2prime = (2 * a + (b + 2 * a * x)**2) * gauss

        # match continuities and derivative cont. (up to 2nd order)
        cont = y - gauss
        cont_prime = y_prime - gauss_prime
        cont_2prime = y_2prime - gauss_2prime
        return (cont, cont_prime, cont_2prime)

      def multiple_guess_fsolve(eqs, guess_ranges):
        # fsolve accuracy very sensitive to initial guess. Try many and refine.

        x0_mesh = np.array(np.meshgrid(*guess_ranges))
        x0_mesh = x0_mesh.T.reshape(-1, len(guess_ranges))

        fits = []
        scores = []
        for x0 in x0_mesh:
          res = fsolve(eqs, x0, full_output=True)
          errs = res[1]['fvec']
          fits.append(res[0])
          scores.append(np.sum(errs**2))

        fits = np.array(fits)
        best_fits = scores < sorted(scores)[5]
        best_fits = fits[best_fits]
        top_fit = fits[np.argmin(scores)]

        return top_fit, best_fits

      # initial guess ranges. Try many and refine.
      a = np.linspace(0, -10, 10)
      b = np.linspace(10, -10, 10)
      c = np.linspace(0, 1, 10)
      _, best_fits = multiple_guess_fsolve(non_linear_eqs, (a, b, c))

      # refined guess ranges.
      a_min = np.min(best_fits[:, 0])
      a_max = np.max(best_fits[:, 0])
      b_min = np.min(best_fits[:, 1])
      b_max = np.max(best_fits[:, 1])
      c_min = np.min(best_fits[:, 2])
      c_max = np.max(best_fits[:, 2])

      a = np.linspace(a_min, a_max, 10)
      b = np.linspace(b_min, b_max, 10)
      c = np.linspace(c_min, c_max, 10)
      top_fit, _ = multiple_guess_fsolve(non_linear_eqs, (a, b, c))

      a, b, c = top_fit
      decay_tail = c * np.exp(a * grids**2 + b * grids)
      decay_tail_deriv = (b + 2 * a * grids) * decay_tail

      return decay_tail, decay_tail_deriv

    tail, tail_deriv = decay_tail(
        grids_2,
        x=grids_1[-1],
        y=n_osc[-1],
        y_prime=-grad_n,
        y_2prime=(grad_n_osc[-1] - grad_n_osc[-2]) /
        (grids_1[-1] - grids_1[-2]),
    )

    # append tail to oscillatory region
    n_g = np.concatenate((n_osc, tail), axis=0)
    n_g_grad = np.concatenate((grad_n_osc, tail_deriv), axis=0)

    # easy rescaling
    grids /= gamma
    n_g *= gamma**3
    n_g_grad *= gamma**4

    # normalize to num_elec
    norm = 4 * pi * np.trapz(n_g * (grids**2), grids)
    n_g *= num_elec / norm
    n_g_grad *= num_elec / norm

    # TODO? report parameters used

    return grids, n_g, n_g_grad

  def default_gedanken_density():
    """ Default gedanken density parameters. 
  
    Returns: callable gedanken density (gamma).
    """
    density = functools.partial(
        GedankenDensity.gedanken_density,
        r_s_min=1,
        r_s_max=1.5,
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
    density = GedankenDensity.default_gedanken_density()

    grids, n_g, n_g_grad = density(gamma=1)

    s_grids, g_s = GedankenDensity.radial_reduced_grad_dist(
        grids,
        n_g,
        n_g_grad,
        s_grids=np.linspace(0, 5, num=1000),
    )

    return s_grids, g_s

  def plot_gedanken_density():

    # plot gedanken density
    density = GedankenDensity.default_gedanken_density()
    grids, n_g, n_g_grad = density(gamma=1)

    # plot HF-calculated He density for reference
    he_grids, he_density = Examples.he_atom_radial_density()

    plt.plot(grids, n_g, label='gedanken density', zorder=2)
    plt.plot(he_grids, he_density / 7, label='He density / 7')
    plt.ylabel('$n(r)$')
    plt.xlabel('$r$')
    plt.ylim(bottom=0, top=0.3)
    plt.xlim(left=0, right=2.5)
    plt.legend(loc='upper right')
    plt.savefig('gedanken_density.pdf', bbox_inches='tight')
    plt.close()

  def plot_gedanken_ks_potential():
    """ Calculate the KS potential for a one- or two-electron (singlet) 
    radial gedanken density. 
    
    Obtained from direct inversion of the radial KS equation:

    v_s(r) = 1/2 (d^2/dr^2 (r n^0.5)) / (r n^0.5) 

    """

    density = GedankenDensity.default_gedanken_density()
    grids, ged_density, _ = density(gamma=1)

    mask = ged_density > 1e-6
    mask[0] = mask[-1] = False

    v_s = 0.5 * GedankenDensity.num_deriv2_fn(grids * ged_density**0.5, grids)
    v_s = v_s[mask[1:-1]] / (grids[mask] * ged_density[mask]**0.5)

    plt.plot(grids, ged_density, label='gedanken density', zorder=2)
    plt.plot(grids[mask], v_s / 800, label='KS potential / 800', zorder=1)
    plt.xlabel('$r$')
    plt.xlim(left=0, right=1.8)
    plt.ylim(top=0.4)
    plt.legend(loc='upper right')
    plt.savefig('gedanken_density_w_potential.pdf', bbox_inches='tight')
    plt.close()


class Examples():

  s_grids = np.linspace(0, 5, num=1000)

  def he_atom_radial_density():
    atom = gto.M(
        atom='He 0 0 0',
        basis='aug-pcseg-4',
        spin=0,
        verbose=4,
    )

    mf = scf.RHF(atom)
    mf.conv_tol_grad = 1e-9
    mf.kernel()

    grids = dft.gen_grid.Grids(atom)
    grids.level = 3
    grids.prune = None
    grids.build()

    num_radial_pts = 500
    radial_grids = np.linspace(1e-5, 2.5, num_radial_pts)
    radi_coords = np.expand_dims(radial_grids, axis=1)
    radi_coords = np.append(radi_coords, np.zeros((num_radial_pts, 2)), axis=1)

    ao_value = numint.eval_ao(atom, radi_coords, deriv=0)
    dm = mf.make_rdm1()
    radial_density = numint.eval_rho(atom, ao_value, dm, xctype='HF')

    return radial_grids, radial_density

  def he_atom_g_s():

    atom = gto.M(
        atom='He 0 0 0',
        basis='aug-pcseg-4',
        spin=0,
        verbose=4,
    )

    mf = scf.RHF(atom)
    mf.conv_tol_grad = 1e-9
    mf.kernel()

    n_ang_grids = dft.gen_grid._default_ang(nuc=2, level=9)
    n_rad_grids = 300

    grids = dft.gen_grid.Grids(atom)
    grids.atom_grid = {'He': (n_rad_grids, n_ang_grids)}
    grids.prune = None
    grids.build()

    mf.grids = grids

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

    grids = dft.gen_grid.Grids(n_atom)
    grids.level = 9
    grids.prune = None
    grids.build()

    mf.grids = grids

    checker = CondChecker(mf, xc='HF')

    s_grids, g_s = checker.reduced_grad_dist(s_grids=Examples.s_grids)

    return s_grids, g_s

  def n2_mol_g_s():

    n2_mol = gto.M(
        atom='N 0 0 0;N 0 0 1.09',
        basis='aug-pcseg-4',
        verbose=4,
    )

    mf = scf.RHF(n2_mol)
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
    he_out = Examples.he_atom_g_s()
    n2_out = Examples.n2_mol_g_s()
    gedanken_out = GedankenDensity.gedanken_g_s()

    # normalize g_s across different systems
    n_out = (n_out[0], n_out[1] / 7)
    n2_out = (n2_out[0], n2_out[1] / 14)
    he_out = (he_out[0], he_out[1] / 2)
    gedanken_out = (gedanken_out[0], gedanken_out[1] / 1)

    s_min = np.min([n_out[0], he_out[0], gedanken_out[0]])
    s_max = np.max([n_out[0], he_out[0], gedanken_out[0]])

    plt.plot(*gedanken_out, label='gedanken')
    plt.plot(*he_out, label='He atom')
    plt.plot(*n_out, label='N atom')
    plt.plot(*n2_out, label='N$_2$ molecule')

    plt.legend(loc='upper right')
    plt.xlim(left=s_min, right=s_max)
    plt.ylim(bottom=0)
    plt.ylabel('$g(s) / N$')
    plt.xlabel('$s$')
    plt.savefig('g_s_combined.pdf', bbox_inches='tight')
    plt.close()


if __name__ == '__main__':
  """ Obtain plots and results for the paper. """

  utils.use_standard_plotting_params()

  # Fig. 1
  e_c_gdn_density_lyp = GedankenDensity.get_e_xc('gga_c_lyp', gamma=1)
  print(f'E^LYP_c[n^gedanken] = {e_c_gdn_density_lyp}')
  e_c_gdn_density_pbe = GedankenDensity.get_e_xc('gga_c_pbe', gamma=1)
  print(f'E^PBE_c[n^gedanken] = {e_c_gdn_density_pbe}')
  GedankenDensity.plot_gedanken_density()
  # Fig. 2
  Examples.combined_examples()

  # Supp. Material Fig. 1
  GedankenDensity.plot_gedanken_ks_potential()
