import copy

import numpy as np

from pyscf import gto, dft, lib, cc, scf
from pyscf.dft import numint, libxc
from scipy.stats import linregress


class CondChecker():
  """ check conditions on self-consistent densities. """

  def __init__(self, mf, xc=None, gams=np.linspace(0.01, 2)):
    self.mf = mf
    self.gams = gams
    self.mol = mf.mol
    if xc is None:
      self.xc = mf.xc
    else:
      self.xc = xc
    self.xctype = libxc.xc_type(self.xc)
    if self.xctype == 'HF':
      # force calculation of higher derivatives on rho (for analysis purposes)
      self.xctype = 'MGGA'

    if getattr(mf, 'grids', False):
      mol_grids = mf.grids
    else:
      mol_grids = dft.gen_grid.Grids(self.mol)
      mol_grids.level = 6
      mol_grids.prune = None
      mol_grids.build()

    self.weights = mol_grids.weights
    self.nelec = np.array(self.mol.nelec, dtype=np.float64)

    if self.mol.spin == 0 and self.mol.charge != -1:
      self.unrestricted = 0
    else:
      self.unrestricted = 1

    # setup density (rho)
    ao_value = numint.eval_ao(self.mol, mol_grids.coords, deriv=2)
    if not self.unrestricted:
      dm = mf.make_rdm1()
      self.rho = numint.eval_rho(self.mol, ao_value, dm, xctype=self.xctype)
    else:
      dm_up, dm_dn = mf.make_rdm1()
      rho_up = numint.eval_rho(self.mol, ao_value, dm_up, xctype=self.xctype)
      rho_dn = numint.eval_rho(self.mol, ao_value, dm_dn, xctype=self.xctype)
      self.rho = (rho_up, rho_dn)

    # caches
    self._caches = {}

  def get_cache(self, s: str):
    return self._caches.get(s, None)

  def set_cache(self, s: str, obj) -> None:
    self._caches[s] = obj

  def get_reduced_grad(self):
    """ Obtain reduced gradient: s(r). """

    rho = self.rho
    if self.unrestricted:
      # get total density
      rho = sum(rho)

    n = rho[0]
    n_grad = rho[1:4]
    abs_n_grad = np.sum(n_grad**2, axis=0)**(1 / 2)

    s = abs_n_grad / (2 * ((3 * (np.pi**2) * n)**(1 / 3)) * n)
    return s

  def reduced_grad_dist(
      self,
      s_grids=np.linspace(0, 3, num=1000),
      fermi_temp=0.05,
      density_tol=1e-9,
  ):
    """ Obtain distribution of the reduced gradient, g_3(s) as defined in:
    
    Zupan, Ales, et al. "Density‐gradient analysis for density functional 
    theory: Application to atoms." International journal of quantum chemistry 
    61.5 (1997): 835-845.

    https://doi.org/10.1002/(SICI)1097-461X(1997)61:5<835::AID-QUA9>3.0.CO;2-X
    
    """

    rho = self.rho
    if self.unrestricted:
      # get total density
      rho = sum(rho)

    n = rho[0]
    s_grids = np.expand_dims(s_grids, axis=1)
    s = self.get_reduced_grad()

    # avoid numerical problems
    mask = n > density_tol
    n = n[mask]
    s = s[mask]
    weights = self.weights[mask]

    fermi_dist = 1 / (np.exp(-(s_grids - s) / fermi_temp) + 1)

    integrand = np.nan_to_num(n * fermi_dist * weights)
    int_g3_s = np.sum(integrand, axis=1)

    s_grids = np.squeeze(s_grids, axis=1)
    g3_s = self.deriv_fn(int_g3_s, s_grids)

    return s_grids, g3_s

  def get_scaled_sys(self, gam):

    if not self.unrestricted:
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

    # create a copy to prevent modifying the original density
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

    if self.get_cache('Ec_gams') is not None and np.array_equal(
        gams, self.gams):
      return self.get_cache('Ec_gams')

    ec_gams = np.array([self.get_Ec_gam(gam) for gam in gams])

    # cache Ec_gams
    if np.array_equal(gams, self.gams):
      self.set_cache('Ec_gams', ec_gams)
    return ec_gams

  def get_Ex_lda(self, gam):
    """ Obtain unpolarized LDA exchange energy. """

    scaled_rho, scaled_weights = self.get_scaled_sys(gam)
    if self.unrestricted:
      # run unpolarized LDA calculation
      scaled_rho = sum(scaled_rho)
    eps_x = dft.libxc.eval_xc('LDA_X', scaled_rho, spin=0)[0]

    rho = scaled_rho[0]
    int_nelec = np.einsum('i,i->', rho, scaled_weights)
    nelec = np.sum(self.nelec)
    np.testing.assert_allclose(int_nelec, nelec, rtol=1e-03)

    ex = np.einsum('i,i,i->', eps_x, rho, scaled_weights)

    return ex

  def get_Ex_lda_gams(self, gams):

    if self.get_cache('Ex_lda_gams') is not None and np.array_equal(
        gams, self.gams):
      return self.get_cache('Ex_lda_gams')

    ex_gams = np.array([self.get_Ex_lda(gam) for gam in gams])

    # cache Ec_gams
    if np.array_equal(gams, self.gams):
      self.set_cache('Ex_lda_gams', ex_gams)
    return ex_gams

  def get_Exc_gam(self, gam):

    scaled_rho, scaled_weights = self.get_scaled_sys(gam)
    eps_xc = dft.libxc.eval_xc(self.xc, scaled_rho, spin=self.unrestricted)[0]

    if not self.unrestricted:
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

    gams_l = self.gams[np.where(self.gams > 1, True, False)]

    # small (s) gam < 1
    mask_s = np.where(self.gams < 1, True, False)
    gams_s = self.gams[mask_s]
    ec_gams_s = self.get_Ec_gams(self.gams)[mask_s]
    cond_s = -tol < gams_s * ec - ec_gams_s
    cond_s = np.all(cond_s)

    # large (l) gam > 1
    mask_l = np.where(self.gams > 1, True, False)
    gams_l = self.gams[mask_l]
    ec_gams_l = self.get_Ec_gams(self.gams)[mask_l]
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
      tol=5e-4,
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

    ec_zero_gams = self.get_Ec_gams(zero_gams)
    deriv_ec_zero_gams = self.deriv_fn(ec_zero_gams, zero_gams)
    deriv_ec_zero = np.median(deriv_ec_zero_gams)

    # Use \lim_{gam -> 0} dEc[n_gam]/dgam = \lim_{gam -> 0} Ec[n_gam] / gam
    # for more numerically accurate limit calculation.
    lim_ec_zero_gams = ec_zero_gams / zero_gams

    _, lim_ec_zero_extrap, r_val, _, _ = linregress(zero_gams, lim_ec_zero_gams)
    if r_val**2 < 0.7:
      raise ValueError(
          "Issue with gam->0 extrapolation. Adjust zero_gams parameter.")

    # Give the benefit of the doubt. Sometimes the extrapolation is unstable,
    # e.g. in some 1 electron cases where Ec[n_gam] ~ 0.
    cond_deriv_ec_zero = min(lim_ec_zero_extrap, deriv_ec_zero)

    cond = tc_gams + (self.gams * cond_deriv_ec_zero) - ec_gams
    cond = cond[end_pt_skip:-end_pt_skip] <= tol
    cond = np.all(cond)

    return cond

  def adiabatic_ec_concavity(self, tol=5e-6, end_pt_skip=3):

    ec_invgams = self.get_Ec_gams(1 / self.gams)
    cond = self.deriv2_fn((self.gams**2) * ec_invgams, self.gams)
    cond = cond[end_pt_skip:-end_pt_skip] <= tol
    cond = np.all(cond)

    return cond

  def lieb_oxford_bound_uxc(self, tol=5e-6, lob_coeff=2.27, end_pt_skip=3):

    ec_gams = self.get_Ec_gams(self.gams)
    ec_deriv = self.deriv_fn(ec_gams, self.gams)
    tc_gams = self.gams * ec_deriv - ec_gams
    exc_gams = self.get_Exc_gams(self.gams)

    # LDA exchange
    ex_lda = self.get_Ex_lda_gams(self.gams)
    cond = exc_gams - tc_gams - lob_coeff * ex_lda
    cond = cond[end_pt_skip:-end_pt_skip] >= -tol
    cond = np.all(cond)

    return cond

  def lieb_oxford_bound_exc(self, tol=5e-6, lob_coeff=2.27):

    # LDA exchange
    ex_lda = self.get_Ex_lda_gams(self.gams)

    exc = self.get_Exc_gams(self.gams)
    cond = (exc - lob_coeff * ex_lda) >= -tol
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
        'lieb_oxford_bound_exc': self.lieb_oxford_bound_exc,
        'lieb_oxford_bound_uxc': self.lieb_oxford_bound_uxc,
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
