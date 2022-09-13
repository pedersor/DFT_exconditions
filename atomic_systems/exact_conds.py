import copy

import numpy as np

from pyscf import gto, dft, lib, cc, scf
from pyscf.dft import numint
from scipy.stats import linregress


class CondChecker():
  """ check conditions on self-consistent densities. """

  def __init__(self, mf, gams=np.linspace(0.01, 2)):
    self.mf = mf
    self.gams = gams
    self.mol = mf.mol
    self.xc = mf.xc
    self.xctype = mf._numint._xc_type(mf.xc)
    self.weights = mf.grids.weights
    self.nelec = np.array(self.mol.nelec, dtype=np.float64)
    self.spin = self.mol.spin

    # setup density (rho)
    ao_value = numint.eval_ao(self.mol, mf.grids.coords, deriv=2)
    if self.spin == 0:
      dm = mf.make_rdm1()
      self.rho = numint.eval_rho(self.mol, ao_value, dm, xctype=self.xctype)
    else:
      dm_up, dm_dn = mf.make_rdm1()
      rho_up = numint.eval_rho(self.mol, ao_value, dm_up, xctype=self.xctype)
      rho_dn = numint.eval_rho(self.mol, ao_value, dm_dn, xctype=self.xctype)
      self.rho = (rho_up, rho_dn)

  def get_scaled_sys(self, gam):

    if self.spin == 0:
      scaled_rho = self.get_scaled_rho(self.rho, gam)
    else:
      rho_up, rho_dn = self.rho
      scaled_rho_up = self.get_scaled_rho(rho_up, gam)
      scaled_rho_dn = self.get_scaled_rho(rho_dn, gam)
      scaled_rho = (scaled_rho_up, scaled_rho_dn)

    # scale integral quadrature weights
    scaled_weights = self.weights / gam**3

    return scaled_rho, scaled_weights

  @staticmethod
  def get_scaled_rho(rho, gam):
    """ Scale density: \gamma^3 n(\gamma \br) ."""

    scaled_rho = copy.deepcopy(rho)
    scaled_rho[0] = (gam**3) * scaled_rho[0]
    scaled_rho[1:4] = (gam**4) * scaled_rho[1:4]
    if scaled_rho.shape[0] > 4:
      scaled_rho[4:] = (gam**5) * scaled_rho[4:]

    return scaled_rho

  def get_Ec_gam(self, gam, gam_inf=5000):

    # E_x = \lim_{gamma \to \infty} E_xc[n_gamma] / gamma
    ex = self.get_Exc_gam(gam=gam * gam_inf) / gam_inf

    ec = self.get_Exc_gam(gam) - ex
    return ec

  def get_Ec_gams(self, gams):
    ec_gams = np.array([self.get_Ec_gam(gam) for gam in gams])
    return ec_gams

  def get_Exc_gam(self, gam):

    scaled_rho, scaled_weights = self.get_scaled_sys(gam)
    eps_xc = dft.libxc.eval_xc(self.xc, scaled_rho, spin=self.spin)[0]

    if self.spin == 0:
      rho = scaled_rho[0]
      int_nelec = np.einsum('i,i->', rho, scaled_weights)
      nelec = np.sum(self.nelec)
      np.testing.assert_allclose(int_nelec, nelec, rtol=1e-03)

      exc = np.einsum('i,i,i->', eps_xc, rho, scaled_weights)

    else:
      rho_up = scaled_rho[0][0]
      rho_dn = scaled_rho[1][0]

      int_nelec_up = np.einsum('i,i->', rho_up, scaled_weights)
      np.testing.assert_allclose(int_nelec_up, self.nelec[0], rtol=1e-03)
      int_nelec_dn = np.einsum('i,i->', rho_dn, scaled_weights)
      np.testing.assert_allclose(int_nelec_dn, self.nelec[1], rtol=1e-03)

      exc = np.einsum('i,i,i->', eps_xc, rho_up + rho_dn, scaled_weights)

    return exc

  def get_Exc_gams(self, gams):
    exc_gams = np.array([self.get_Exc_gam(gam) for gam in gams])
    return exc_gams

  @staticmethod
  def grid_spacing(arr):
    """ Get uniform spacing of gammas. """
    dx = arr[1] - arr[0]
    dx_alt = (arr[-1] - arr[0]) / (len(arr) - 1)
    np.testing.assert_allclose(
        dx,
        dx_alt,
        err_msg='values need to be uniformly spaced.',
    )
    return dx

  def deriv_fn(self, arr, grids):
    """ Numerical 1st derivative of arr on grids."""
    dx = self.grid_spacing(grids)
    deriv = np.gradient(arr, dx, edge_order=2, axis=0)
    return deriv

  def deriv2_fn(self, arr, grids):
    """ Numerical 2nd derivative of arr on grids."""
    dx = self.grid_spacing(grids)
    deriv2 = np.diff(arr, 2, axis=0) / (dx**2)
    return deriv2

  ## Exact conditions

  def ec_non_positivity(self, tol=5e-6):

    ec_gams = self.get_Ec_gams(self.gams)

    cond = ec_gams <= tol
    cond = np.all(cond)

    return cond

  def ec_scaling_check(self, tol=5e-6):

    ec = self.get_Ec_gam(1)

    gams_s = self.gams[np.where(self.gams < 1, True, False)]
    gams_l = self.gams[np.where(self.gams > 1, True, False)]

    # small (s) gam < 1
    ec_gams_s = self.get_Ec_gams(gams_s)
    cond_s = -tol < gams_s * ec - ec_gams_s
    cond_s = np.all(cond_s)

    # large (l) gam > 1
    ec_gams_l = self.get_Ec_gams(gams_l)
    cond_l = tol > gams_l * ec - ec_gams_l
    cond_l = np.all(cond_l)

    return cond_s and cond_l

  def tc_non_negativity(self, tol=5e-6, end_pt_skip=3):

    ec_gams = self.get_Ec_gams(self.gams)
    ec_deriv = self.deriv_fn(ec_gams, self.gams)
    tc_gams = self.gams * ec_deriv - ec_gams
    # skip end points (inaccurate deriv.)
    cond = tc_gams[end_pt_skip:-end_pt_skip] >= -tol
    cond = np.all(cond)

    return cond

  def tc_upper_bound(
      self,
      tol=1e-6,
      end_pt_skip=3,
      zero_gams=None,
  ):

    ec_gams = self.get_Ec_gams(self.gams)
    ec_deriv = self.deriv_fn(ec_gams, self.gams)
    tc_gams = self.gams * ec_deriv - ec_gams

    # Ec(\gamma -> 0) from linear extrapolation
    if zero_gams is None:
      center = 1e-2
      step = 1e-4
      zero_gams = np.linspace(center - 5 * step, center + 5 * step, num=10)
    ec_gams0 = self.get_Ec_gams(zero_gams) / zero_gams

    _, ec_extrap_gam0, r_val, _, _ = linregress(zero_gams, ec_gams0)
    if r_val**2 < 0.7:
      raise ValueError(
          "Issue with gam->0 extrapolation. Adjust zero_gams parameter.")

    cond = tc_gams + (self.gams * ec_extrap_gam0) - ec_gams
    cond = cond[end_pt_skip:-end_pt_skip] <= tol
    cond = np.all(cond)

    return cond

  def adiabatic_ec_concavity(self, tol=5e-6, end_pt_skip=3):

    ec_invgams = self.get_Ec_gams(1 / self.gams)
    cond = self.deriv2_fn((self.gams**2) * ec_invgams, self.gams)
    cond = cond[end_pt_skip:-end_pt_skip] <= tol
    cond = np.all(cond)

    return cond

  def check_conditions(self, conds_str_list=None):
    """ Check several different exact conditions."""

    # all exact conditions
    conds = {
        'ec_non_positivity': self.ec_non_positivity,
        'ec_scaling_check': self.ec_scaling_check,
        'tc_non_negativity': self.tc_non_negativity,
        'tc_upper_bound': self.tc_upper_bound,
        'adiabatic_ec_concavity': self.adiabatic_ec_concavity,
    }

    # run all checks
    if conds_str_list is None:
      conds_str_list = conds.keys()

    res = {}
    # note: default settings only across all condition checks
    for cond in conds_str_list:
      check = conds[cond]()
      res[cond] = check

    return res
