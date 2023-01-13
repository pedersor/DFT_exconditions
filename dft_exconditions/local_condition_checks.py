from collections.abc import Callable
from typing import List, Optional, Tuple, Dict, Union

import numpy as np
import pandas as pd

import pylibxc


def get_density(r_s: np.ndarray) -> np.ndarray:
  """Obtains densities, n, from Wigner-Seitz radii, r_s.
  
  Args:
    r_s: Wigner-Seitz radii on a grid with shape (num_r_s_points,).
  
  Returns:
    n: densities on a grid with shape (num_r_s_points,).
  """

  return 3 / (4 * np.pi * (r_s**3))


def hartree_to_mRy(energy: float) -> float:
  """Hartree to mRy units conversion."""

  return energy * 2 * 1000


def get_eps_x_unif(n: np.ndarray) -> np.ndarray:
  """Uniform gas exchange energy per particle. 
  
  Args:
    n: densities on a grid with shape (num_density_points,).

  Returns:
    eps_x_unif: uniform gas exchange energy per particle on a 
      grid with shape (num_density_points,).
  """

  return -(3 / (4 * np.pi)) * ((n * 3 * np.pi**2)**(1 / 3))


def get_grad_n(s: np.ndarray, n: np.ndarray) -> np.ndarray:
  """Obtain |\nabla n| from reduced gradient, s. 
  
  Args:
    s: reduced density gradient on a grid with shape (num_density_points,).
    n: densities on a grid with shape (num_density_points,).

  Returns:
    grad_n: |\nabla n| on a grid with shape (num_density_points,).
  """

  return s * (2 * ((3 * np.pi**2)**(1 / 3)) * (n**(4 / 3)))


def get_up_dn_density(n: np.ndarray, zeta: np.ndarray) -> np.ndarray:
  """Obtains up- and down-spin densities

  Params:
    n: real-space density on a grid with shape (num_density_points,).
    zeta: relative spin polarization on a grid with shape 
      (num_density_points,).
  
  Returns:
    up_dn_density: up and down density on a grid with shape 
      (num_density_points, 2)
  """

  n = np.expand_dims(n, axis=1)
  zeta = np.expand_dims(zeta, axis=1)

  up_coeff, dn_ceoff = zeta_coeffs(zeta)
  up_density = up_coeff * n
  dn_density = dn_ceoff * n

  return np.concatenate((up_density, dn_density), axis=1)


def zeta_coeffs(zeta: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
  """Coefficients relating n_sigma to zeta."""

  return (1 + zeta) / 2, (1 - zeta) / 2


def get_tau(
    alpha: np.ndarray,
    grad_n: np.ndarray,
    n: np.ndarray,
    zeta: np.ndarray,
) -> np.ndarray:
  """Obtains up- and down- kinetic energy densities, tau 

  Kinetic energy density:
  tau_sigma = 1/2 \sum_i^occ |\nabla \phi_{sigma i}|^2 .
  
  For the uniform gas:
  tau_unif = 3/10 * (3 pi^2)^2/3 * n^{5/3} .

  The von Weizsacker kinetic energy density is given by:
  tau_vw = 1/8 * |\nabla n|^2 / n .

  Args:
    alpha: (tau - tau_vw) / tau_unif. On a grid with shape (num_density_points,).
    grad_n: |\nabla n| on a grid with shape (num_density_points,).
    n: densities on a grid with shape (num_density_points,).
    zeta: relative spin polarization on a grid with shape (num_density_poitns,).

  Returns:
    tau: up- and down- kinetic energy densities on a grid with shape 
      (num_density_points, 2)
  """

  tau_vw = (grad_n**2) / (8 * n)
  tau_unif = (3 / 10) * ((3 * np.pi**2)**(2 / 3)) * (n**(5 / 3))
  d_s = ((1 + zeta)**(5 / 3) + (1 - zeta)**(5 / 3)) / 2
  tau_unif *= d_s
  tau = (alpha * tau_unif) + tau_vw

  up_coeff, dn_coeff = (
      np.expand_dims(coeff, axis=1) for coeff in zeta_coeffs(zeta))
  tau = np.expand_dims(tau, axis=1)
  tau = np.concatenate((up_coeff * tau, dn_coeff * tau), axis=1)

  return tau


def get_sigma(grad_n: np.ndarray, zeta: np.ndarray) -> np.ndarray:
  """Obtains contracted density gradients.

  Args: 
    grad_n: |\nabla n| on a grid with shape (num_density_points,).
    zeta: relative spin polarization on a grid with shape 
      (num_density_points,).
  
  Returns:
    sigma: density gradients, [[\nabla n_up * \nabla n_up, 
      \nabla n_up * \nabla n_dn, \nabla n_dn * \nabla n_dn], ...] with 
      shape (num_density_points, 3).
  """

  sigma = np.expand_dims(grad_n**2, axis=1)

  up_coeff, dn_coeff = (
      np.expand_dims(coeff, axis=1) for coeff in zeta_coeffs(zeta))

  sigma = np.concatenate(
      (up_coeff**2 * sigma, up_coeff * dn_coeff * sigma, dn_coeff**2 * sigma),
      axis=1)

  return sigma


def get_lapl(q: np.ndarray, n: np.ndarray, zeta: np.ndarray) -> np.ndarray:
  """Obtains density laplacians.

  Args:
    q: reduced density Laplacian (Eq. 14 in PhysRevA.96.052512) with shape 
      (num_density_points,).
    n: densities on a grid with shape (num_density_points,).
    zeta: relative spin polarization on a grid with shape (num_density_points,).

  Returns:
    lapl: up- and down- density laplacians with shape (num_density_points, 2). 
  """

  n = np.expand_dims(n, axis=1)
  q = np.expand_dims(q, axis=1)
  zeta = np.expand_dims(zeta, axis=1)
  lapl = q * 4 * (3 * np.pi**2)**(2 / 3) * (n**(5 / 3))

  up_coeff, dn_ceoff = zeta_coeffs(zeta)
  up_lapl = up_coeff * lapl
  dn_lapl = dn_ceoff * lapl

  return np.concatenate((up_lapl, dn_lapl), axis=1)


def lda_xc(
    libxc_fun: pylibxc.LibXCFunctional,
    r_s: np.ndarray,
    zeta: np.ndarray,
) -> np.ndarray:
  """Obtains (exchange)-correlation (X)C energy per particle for 
  local density approximation (LDA) functionals.

  \epsilon_(x)c^{LDA}(r_s, \zeta)
  
  Args:
    libxc_fun: Libxc functional object.
    r_s: Wigner-Seitz radii on a grid with shape (mesh_shape,).
    zeta: relative spin polarization on a grid with shape (mesh_shape,).
  
  Returns:
    eps_xc: (X)C energy per particle on a grid with shape
      (mesh_shape,).
  """

  mesh_shape = r_s.shape
  inp = (r_s, zeta)
  inp = (feature.flatten() for feature in inp)
  r_s, zeta = inp

  # obtain Libxc inps
  n = get_density(r_s)
  rho = get_up_dn_density(n, zeta)

  inp = {}
  inp["rho"] = rho

  libxc_fun_res = libxc_fun.compute(inp)
  eps_xc = np.squeeze(libxc_fun_res['zk'])

  return eps_xc.reshape(mesh_shape)


def gga_xc(
    libxc_fun: pylibxc.LibXCFunctional,
    r_s: np.ndarray,
    s: np.ndarray,
    zeta: np.ndarray,
) -> np.ndarray:
  """Obtains (X)C energy per particle for generalized gradient approximation
  (GGA) functionals.

  \epsilon_(x)c^{GGA}(r_s, s, \zeta, \alpha)

  Args:
    libxc_fun: Libxc functional object.
    r_s: Wigner-Seitz radii on a grid with shape (mesh_shape,).
    s: reduced density gradients on a grid with shape (mesh_shape,).
    zeta: relative spin polarization on a grid with shape (mesh_shape,).

  Returns:
    eps_xc: (X)C energy per particle on a grid with 
      shape (mesh_shape,).
  """

  mesh_shape = r_s.shape
  inp = (r_s, s, zeta)
  inp = (feature.flatten() for feature in inp)
  r_s, s, zeta = inp

  # obtain Libxc inps
  n = get_density(r_s)
  grad_n = get_grad_n(s, n)
  rho = get_up_dn_density(n, zeta)
  sigma = get_sigma(grad_n, zeta)

  inp = {}
  inp["rho"] = rho
  inp["sigma"] = sigma

  libxc_fun_res = libxc_fun.compute(inp)
  eps_xc = np.squeeze(libxc_fun_res['zk'])

  return eps_xc.reshape(mesh_shape)


def mgga_xc(
    libxc_fun: pylibxc.LibXCFunctional,
    r_s: np.ndarray,
    s: np.ndarray,
    zeta: np.ndarray,
    alpha: np.ndarray,
) -> np.ndarray:
  """Obtains (X)C energy per particle for meta-GGA (MGGA) functionals 
  (without laplacian).
  
  \epsilon_c^{MGGA}(r_s, s, \zeta, \alpha) .
  
  Args:
    libxc_fun: Libxc functional object.
    r_s: Wigner-Seitz radii on a grid with shape (mesh_shape,).
    s: reduced density gradients on a grid with shape (mesh_shape,).
    zeta: relative spin polarization on a grid with shape (mesh_shape,).
    alpha: (tau - tau_vw) / tau_unif. On a grid with shape (mesh_shape,). 

  Returns:
    eps_xc: (X)C energy per particle on a grid with 
      shape (mesh_shape,).
  """

  mesh_shape = r_s.shape
  inp = (r_s, s, zeta, alpha)
  inp = (feature.flatten() for feature in inp)
  r_s, s, zeta, alpha = inp

  # obtain Libxc inps
  n = get_density(r_s)
  grad_n = get_grad_n(s, n)
  rho = get_up_dn_density(n, zeta)
  tau = get_tau(alpha, grad_n, n, zeta)
  sigma = get_sigma(grad_n, zeta)

  inp = {}
  inp["rho"] = rho
  inp["sigma"] = sigma
  inp["tau"] = tau

  libxc_fun_res = libxc_fun.compute(inp)
  eps_xc = np.squeeze(libxc_fun_res['zk'])

  return eps_xc.reshape(mesh_shape)


def mgga_xc_lapl(
    libxc_fun: pylibxc.LibXCFunctional,
    r_s: np.ndarray,
    s: np.ndarray,
    zeta: np.ndarray,
    alpha: np.ndarray,
    q: np.ndarray,
) -> np.ndarray:
  """Obtains (X)C energy per particle for meta-GGA (MGGA) functionals 
  with laplacian.

  \epsilon_c^{MGGA}(r_s, s, \zeta, \alpha, q) .
  
  Args:
    libxc_fun: Libxc functional object.
    r_s: Wigner-Seitz radii on a grid with shape (mesh_shape,).
    s: reduced density gradients on a grid with shape (mesh_shape,).
    zeta: relative spin polarization on a grid with shape (mesh_shape,).
    alpha: (tau - tau_vw) / tau_unif. On a grid with shape (mesh_shape,). 
    q: reduced density Laplacian (Eq. 14 in PhysRevA.96.052512) with shape 
      (mesh_shape,).

  Returns:
    eps_xc: (X)C energy per particle on a grid with 
      shape (mesh_shape,).
  """

  mesh_shape = r_s.shape
  inp = (r_s, s, zeta, alpha, q)
  inp = (feature.flatten() for feature in inp)
  r_s, s, zeta, alpha, q = inp

  # obtain Libxc inputs
  n = get_density(r_s)
  grad_n = get_grad_n(s, n)
  rho = get_up_dn_density(n, zeta)
  tau = get_tau(alpha, grad_n, n, zeta)
  sigma = get_sigma(grad_n, zeta)
  lapl = get_lapl(q, n, zeta)

  inp = {}
  inp["rho"] = rho
  inp["sigma"] = sigma
  inp["tau"] = tau
  inp["lapl"] = lapl

  libxc_fun_res = libxc_fun.compute(inp)
  eps_xc = np.squeeze(libxc_fun_res['zk'])

  return eps_xc.reshape(mesh_shape)


def eps_to_enh_factor(
    inp_mesh: List[np.ndarray],
    eps_x_c: np.ndarray,
) -> np.ndarray:
  """Convert (X)C energy per particle, \epsilon_(X)C, to enhancement factor, 
  F_(X)C.
  
  Args:
    inp_mesh: list of input mesh features each with shape (mesh_shape,).
    eps_x_c: (X)C energy per particle on a grid with shape (mesh_shape,).

  Returns:
    enh_factor: enhancement factor on a grid with shape (mesh_shape,).
  """

  r_s_mesh = inp_mesh[0]
  n = get_density(r_s_mesh)
  eps_x_unif = get_eps_x_unif(n)

  return eps_x_c / eps_x_unif


def get_dfa_rung(func_id: str) -> Callable:
  """Returns the density functional approximation (DFA) type of a 
  given libxc functional id string. 

  Args:
    func_id: Libxc functional identifier string.

  Returns:
    fun: callable dfa_xc function.
  """

  dfa_rungs = {
      "lda_": lda_xc,
      "mgga_": mgga_xc,
      "gga_": gga_xc,
      "hyb_mgga_": mgga_xc,
      "hyb_gga_": gga_xc,
  }

  for dfa, dfa_fun in dfa_rungs.items():
    if dfa in func_id:
      return dfa_fun

  raise NotImplementedError(f"functional {func_id} not supported.")


def is_xc_func(func_id: str) -> bool:
  """Check if a given functional is a XC functional than cannot be 
  separated in Libxc. 
  """

  return '_xc_' in func_id


def get_enh_factor_x_c(
    func_id: str,
    std_inp: List[np.ndarray],
    xc_func: Optional[bool] = False,
) -> np.ndarray:
  """Obtains (X)C enhancement factor for a given functional and input (inp). 
  
  If the correlation enhancement factor is requested and the C functional is
  not available in Libxc, the enhancement factor is obtained from the following
  "conventional" partitioning:

  \epsilon_c(r_s, ...) = \epsilon_xc(r_s, ...) 
    - \lim_{\gamma \to \infty} \epsilon_xc(r_s / \gamma, ...) / \gamma 

  Args:
    func_id: Libxc functional identifier string.
    std_inp: list of input mesh features each with shape (mesh_shape,).
    xc_func: whether the functional is a XC functional than cannot be separated 
      in Libxc.

  Returns:
    enh_factor: (X)C enhancement factor on a grid with shape (mesh_shape,).
  """

  inp_mesh = np.meshgrid(*std_inp, indexing='ij')
  dfa = get_dfa_rung(func_id)

  if xc_func and '_c_' in func_id:
    func_id = func_id.replace('_c_', '_xc_')
    libxc_fun = pylibxc.LibXCFunctional(func_id, "polarized")
    if libxc_fun._needs_laplacian:
      dfa = mgga_xc_lapl

    eps_xc = dfa(libxc_fun, *inp_mesh)
    f_xc = eps_to_enh_factor(inp_mesh, eps_xc)

    # obtain "conventional" partitioning by taking an approximate limit
    # using \gamma = 5000.
    inf_gam = 5000
    scaled_r_s = std_inp[0] / inf_gam
    scaled_inp_mesh = np.meshgrid(scaled_r_s, *std_inp[1:], indexing='ij')
    eps_x = dfa(libxc_fun, *scaled_inp_mesh) / inf_gam
    f_x = eps_to_enh_factor(inp_mesh, eps_x)

    f_c = f_xc - f_x
    return f_c
  elif xc_func and '_x_' in func_id:
    func_id = func_id.replace('_x_', '_xc_')
    libxc_fun = pylibxc.LibXCFunctional(func_id, "polarized")
    if libxc_fun._needs_laplacian:
      dfa = mgga_xc_lapl

    # obtain "conventional" partitioning by taking an approximate limit
    # using \gamma = 5000.
    inf_gam = 5000
    scaled_r_s = std_inp[0] / inf_gam
    scaled_inp_mesh = np.meshgrid(scaled_r_s, *std_inp[1:], indexing='ij')
    eps_x = dfa(libxc_fun, *scaled_inp_mesh) / inf_gam
    f_x = eps_to_enh_factor(inp_mesh, eps_x)
    return f_x
  else:
    libxc_fun = pylibxc.LibXCFunctional(func_id, "polarized")
    if libxc_fun._needs_laplacian:
      dfa = mgga_xc_lapl

    eps_x_c = dfa(libxc_fun, *inp_mesh)
    f_x_c = eps_to_enh_factor(inp_mesh, eps_x_c)
    return f_x_c


class Functional():

  def __init__(self, func_id: str) -> None:
    """Initialize a functional object.

    Args:
      func_id: Libxc functional identifier string.
    """

    func_id = func_id.lower()
    self.is_combined_xc = '_xc_' in func_id

    if self.is_combined_xc:
      self.libxc_fun = self._get_libxc_fun(func_id)
      self.name = self.get_name(self.libxc_fun._xc_func_name)
      self.family = self.libxc_fun._family
      self.needs_laplacian = self.libxc_fun._needs_laplacian
      # callable density functional approximation (DFA) function to use
      self.dfa_fun = self._get_dfa_fun(self.libxc_fun)
      self._get_hybrid_variables(self.libxc_fun)

    else:
      # try to get corresponding x or c functional ids (if possible)
      if '_x_' in func_id:
        libxc_fun_x = self._get_libxc_fun(func_id)
        libxc_fun_c = self._get_corresponding_x_c(func_id)
      elif '_c_' in func_id:
        libxc_fun_c = self._get_libxc_fun(func_id)
        libxc_fun_x = self._get_corresponding_x_c(func_id)
      else:
        raise ValueError(f"functional {func_id} not supported.")

      self.libxc_fun = (libxc_fun_x, libxc_fun_c)

      if libxc_fun_x and libxc_fun_c and libxc_fun_x._family != libxc_fun_c._family:
        raise ValueError(
            f"Only exchange and correlation functionals that are in the same \
            family are supported.")

      self.name = self.get_name(libxc_fun_c._xc_func_name)
      self.family = libxc_fun_x._family
      self.needs_laplacian = (
          (libxc_fun_c is not None and libxc_fun_c._needs_laplacian) or
          (libxc_fun_x is not None and libxc_fun_x._needs_laplacian))
      self._get_hybrid_variables(libxc_fun_x)
      self.dfa_fun = self._get_dfa_fun()

  def get_name(self, func_id: str) -> str:

    func_id = func_id.lower()
    if '_xc_' in func_id:
      xc_label = func_id.split('_xc_')[-1]
    elif '_c_' in func_id:
      xc_label = func_id.split('_c_')[-1]
    else:
      raise ValueError(f"functional {func_id} not supported.")
    xc_label = xc_label.replace('_', '-').upper()
    return xc_label

  def _get_libxc_fun(self, func_id: str) -> pylibxc.LibXCFunctional:
    """Get the libxc functional object for a given functional id."""

    libxc_fun = pylibxc.LibXCFunctional(func_id, "polarized")

    if libxc_fun._nlc_C or libxc_fun._nlc_b:
      raise ValueError(f"Non-local correlation functionals are not supported.")

    return libxc_fun

  def _get_hybrid_variables(self, libxc_fun) -> None:
    """Get the associated hybrid variables for a given libxc functional object."""

    self.range_sep_hyb = False

    if libxc_fun is None:
      self.hyb_exx_coef = 0
      return

    hyb_type = libxc_fun._hyb_type
    if hyb_type == pylibxc.flags.XC_HYB_SEMILOCAL:
      self.hyb_exx_coef = 0
    elif hyb_type == pylibxc.flags.XC_HYB_HYBRID:
      self.hyb_exx_coef = libxc_fun.get_hyb_exx_coef()
    elif (hyb_type == pylibxc.flags.XC_HYB_CAM or
          hyb_type == pylibxc.flags.XC_HYB_CAMY or
          hyb_type == pylibxc.flags.XC_HYB_CAMG):
      self.range_sep_hyb = True
    else:
      raise ValueError(f"Functional {libxc_fun._xc_func_name} not supported. \
          It may be a double hybrid or some other unsupported functional.")

  def _get_corresponding_x_c(self, func_id_x_c: str) -> str:
    """Get the corresponding exchange (correlation) functional id for a given
    correlation (exchange) functional id."""

    if '_c_' in func_id_x_c:
      corresponding_ids = [
          func_id_x_c.replace('_c_', '_x_'),
          'hyb_' + func_id_x_c.replace('_c_', '_x_'),
      ]
    elif '_x_' in func_id_x_c:
      corresponding_ids = [
          func_id_x_c.replace('_x_', '_c_'),
          func_id_x_c.replace('_x_', '_c_').replace('hyb_', ''),
      ]
    else:
      raise ValueError(f"functional {func_id_x_c} not supported.")

    id_match = None
    for id in corresponding_ids:
      try:
        id_match = self._get_libxc_fun(id)
      except:
        pass

    return id_match

  def _get_dfa_fun(self) -> Callable:
    """Get the callable function for the corresponding density functional 
    approximation in libxc."""

    if self.needs_laplacian and self.family == pylibxc.flags.XC_FAMILY_MGGA:
      return mgga_xc_lapl
    elif self.family == pylibxc.flags.XC_FAMILY_LDA:
      return lda_xc
    elif self.family == pylibxc.flags.XC_FAMILY_GGA:
      return gga_xc
    elif self.family == pylibxc.flags.XC_FAMILY_MGGA:
      return mgga_xc
    else:
      raise ValueError(f"functional not supported.")


# TODO: omit range-separated hybrids from LO analysis and omit all NLC functionals
class LocalCondChecker():

  def __init__(
      self,
      functional: Functional,
      conditions_to_check: List[str] = None,
      vars_to_check: Dict[str, np.ndarray] = None,
  ):
    self.functional = functional

    if conditions_to_check is None:
      # by default, get all available conditions to check
      self.conditions_to_check = self.get_avail_conds_to_check()
    else:
      self.conditions_to_check = conditions_to_check

    if vars_to_check is None:
      # use default grid search variables
      self.vars_to_check = default_search_variables(self.functional)
    else:
      self.vars_to_check = vars_to_check

  def possible_search_vars(self) -> List[str]:
    """Get the possible search variables."""

    possible_vars = ['r_s', 's', 'zeta', 'alpha', 'q']

    return possible_vars

  def get_avail_conds_to_check(self) -> List[str]:
    """Get the available conditions to check."""

    # correlation energy conditions
    c_conds = [
        "negativity_check",
        "deriv_lower_bd_check",
        "deriv_upper_bd_check_1",
        "deriv_upper_bd_check_2",
        "second_deriv_check",
    ]

    # XC energy conditions
    xc_conds = [
        "lieb_oxford_bd_check_Uxc",
        "lieb_oxford_bd_check_Exc",
    ]

    avail_conds = []
    if self.functional.is_combined_xc and self.functional.range_sep_hyb:
      avail_conds.extend(c_conds)
    elif self.functional.is_combined_xc:
      avail_conds.extend(c_conds)
      avail_conds.extend(xc_conds)
    elif self.functional.libxc_fun[0] is None or self.functional.range_sep_hyb:
      # correlation-only functional or range-separated hybrid
      avail_conds.extend(c_conds)
    elif self.functional.libxc_fun[0] and self.functional.libxc_fun[1]:
      # exchange + correlation functional
      avail_conds.extend(xc_conds)
      avail_conds.extend(c_conds)
    elif self.functional.libxc_fun[1]:
      # exchange-only functional
      raise NotImplementedError("Exchange-only functionals not supported yet.")
    else:
      raise ValueError("Functional not supported.")

    return avail_conds

  def get_enh_factor_x_c(
      self,
      functional: Functional,
      std_inp: List[np.ndarray],
  ) -> np.ndarray:
    """Obtains (X)C enhancement factor for a given functional and input (inp). 
    
    If the correlation enhancement factor is requested and the C functional is
    not available in Libxc, the enhancement factor is obtained from the following
    "conventional" partitioning:

    \epsilon_c(r_s, ...) = \epsilon_xc(r_s, ...) 
      - \lim_{\gamma \to \infty} \epsilon_xc(r_s / \gamma, ...) / \gamma 

    Args:
      func_id: Libxc functional identifier string.
      std_inp: list of input mesh features each with shape (mesh_shape,).
      xc_func: whether the functional is a XC functional than cannot be separated 
        in Libxc.

    Returns:
      enh_factor: (X)C enhancement factor on a grid with shape (mesh_shape,).
    """

    inp_mesh = np.meshgrid(*std_inp, indexing='ij')
    dfa = functional.dfa_fun

    if False and xc_func and '_c_' in func_id:
      func_id = func_id.replace('_c_', '_xc_')
      libxc_fun = pylibxc.LibXCFunctional(func_id, "polarized")
      if libxc_fun._needs_laplacian:
        dfa = mgga_xc_lapl

      eps_xc = dfa(libxc_fun, *inp_mesh)
      f_xc = eps_to_enh_factor(inp_mesh, eps_xc)

      # obtain "conventional" partitioning by taking an approximate limit
      # using \gamma = 5000.
      inf_gam = 5000
      scaled_r_s = std_inp[0] / inf_gam
      scaled_inp_mesh = np.meshgrid(scaled_r_s, *std_inp[1:], indexing='ij')
      eps_x = dfa(libxc_fun, *scaled_inp_mesh) / inf_gam
      f_x = eps_to_enh_factor(inp_mesh, eps_x)

      f_c = f_xc - f_x
      return f_c
    elif False and xc_func and '_x_' in func_id:
      func_id = func_id.replace('_x_', '_xc_')
      libxc_fun = pylibxc.LibXCFunctional(func_id, "polarized")
      if libxc_fun._needs_laplacian:
        dfa = mgga_xc_lapl

      # obtain "conventional" partitioning by taking an approximate limit
      # using \gamma = 5000.
      inf_gam = 5000
      scaled_r_s = std_inp[0] / inf_gam
      scaled_inp_mesh = np.meshgrid(scaled_r_s, *std_inp[1:], indexing='ij')
      eps_x = dfa(libxc_fun, *scaled_inp_mesh) / inf_gam
      f_x = eps_to_enh_factor(inp_mesh, eps_x)
      return f_x
    else:

      libxc_fun_x, libxc_fun_c = functional.libxc_fun

      eps_x_c = dfa(libxc_fun_c, *inp_mesh)
      f_x_c = eps_to_enh_factor(inp_mesh, eps_x_c)
      return f_x_c

  def check_condition_work(
      self,
      functional: Functional,
      condition: str,
      std_inp: List[np.ndarray],
      tol: Optional[float] = None,
  ) -> Tuple:

    r_s = std_inp[0]

    if condition == 'deriv_upper_bd_check_1':
      # add r_s = 100 (to approximate r_s -> \infty)
      r_s = np.append(r_s, 100)

    r_s_dx = r_s[1] - r_s[0]

    if 'lieb_oxford_bd_check' in condition:
      f_c = get_enh_factor_x_c(func_id_c, std_inp, xc_func=xc_func)

      # get exchange
      try:
        func_id_x = func_id_c.replace('_c_', '_x_')
        f_x = get_enh_factor_x_c(func_id_x, std_inp, xc_func=xc_func)
      except:
        func_id_x = 'hyb_' + func_id_c.replace('_c_', '_x_')
        f_x = get_enh_factor_x_c(func_id_x, std_inp, xc_func=xc_func)

      f_x_c = (f_x, f_c)
    else:
      f_x_c = self.get_enh_factor_x_c(functional, std_inp)

    inp_mesh = np.meshgrid(*std_inp, indexing='ij')

    condition_fun = globals()[condition]
    if tol is None:
      result = condition_fun(inp_mesh, f_x_c, r_s_dx)
    else:
      result = condition_fun(inp_mesh, f_x_c, r_s_dx, tol)

    return result

  def check_conditions(
      self,
      num_blocks: Optional[int] = 100,
  ) -> pd.DataFrame:

    func_label = self.functional.name
    num_conditions = len(self.conditions_to_check)
    df = {
        'xc': [func_label] * num_conditions,
        'condition': [None] * num_conditions,
        'satisfied': [None] * num_conditions,
        'percent_violated': [None] * num_conditions,
    }
    range_labels = [key + '_range' for key in self.vars_to_check.keys()]
    for label in range_labels:
      df[label] = [[]] * num_conditions

    s = self.vars_to_check.get('s', None)
    std_inp = get_standard_input(self.vars_to_check)

    # total number of condition checks. Keep track of number of violations
    num_checks = np.prod(np.array([var.shape for var in std_inp]))
    num_violated = 0
    cond_satisfied = True

    if s is None:
      s_splits = [None]
    else:
      s_splits = np.split(s, num_blocks)

    for s_split in s_splits:
      if s_split is not None:
        std_inp[1] = s_split

      # todo: for i, condition in enumerate(self.conditions_to_check):

      i = 0
      condition = 'negativity_check'

      split_cond_satisfied, split_num_violated, ranges = self.check_condition_work(
          self.functional,
          condition,
          std_inp,
      )

      num_violated += split_num_violated
      if not split_cond_satisfied:
        cond_satisfied = False
        for j, r in enumerate(ranges):
          df[range_labels[j]][i].append(r)

    if cond_satisfied:
      for label in range_labels:
        df[label] = ['---']
    else:
      for label in range_labels:
        min_range = np.amin(df[label])
        max_range = np.amax(df[label])
        df[label] = [[min_range, max_range]]

    df['satisfied'] = [cond_satisfied]
    df['percent_violated'] = [num_violated / num_checks]
    df = pd.DataFrame.from_dict(df)
    return df


def check_condition_work(
    func_id: str,
    condition: Callable,
    std_inp: List[np.ndarray],
    tol: Optional[float] = None,
) -> Tuple:

  r_s = std_inp[0]
  r_s_dx = r_s[1] - r_s[0]
  xc_func = is_xc_func(func_id)
  if xc_func:
    func_id_c = func_id.replace('_xc_', '_c_')
  else:
    func_id_c = func_id

  if 'lieb_oxford_bd_check' in condition.__name__:
    f_c = get_enh_factor_x_c(func_id_c, std_inp, xc_func=xc_func)

    # get exchange
    try:
      func_id_x = func_id_c.replace('_c_', '_x_')
      f_x = get_enh_factor_x_c(func_id_x, std_inp, xc_func=xc_func)
    except:
      func_id_x = 'hyb_' + func_id_c.replace('_c_', '_x_')
      f_x = get_enh_factor_x_c(func_id_x, std_inp, xc_func=xc_func)

    f_x_c = (f_x, f_c)
  else:
    f_x_c = get_enh_factor_x_c(func_id_c, std_inp, xc_func=xc_func)

  inp_mesh = np.meshgrid(*std_inp, indexing='ij')

  if tol is None:
    result = condition(inp_mesh, f_x_c, r_s_dx)
  else:
    result = condition(inp_mesh, f_x_c, r_s_dx, tol)

  return result


def get_standard_input(inp: Dict[str, np.ndarray]) -> np.ndarray:
  """Get standard variable input."""

  r_s = inp['r_s']
  zeta = inp['zeta']
  if len(inp) == 2:
    # LDA
    std_inp = [r_s, zeta]
  elif len(inp) == 3:
    # GGA
    s = inp['s']
    std_inp = [r_s, s, zeta]
  elif len(inp) == 4:
    # MGGA w/o laplacian
    s = inp['s']
    alpha = inp['alpha']
    std_inp = [r_s, s, zeta, alpha]
  elif len(inp) == 5:
    # MGGA w/ laplacian
    s = inp['s']
    alpha = inp['alpha']
    q = inp['q']
    std_inp = [r_s, s, zeta, alpha, q]

  return std_inp


def check_condition(
    func_id: str,
    condition_string: str,
    inp: Dict[str, np.ndarray],
    num_blocks: Optional[int] = 100,
    tol: Optional[float] = None,
) -> pd.DataFrame:
  """Assessment of a given local condition for a given functional and input. 
  
  If the condition is not met, return min. and max. variables where the 
  condition is not met. The percentage of violations is also returned.

  Args:
    func_id: XC functional identifier.
    condition_string: string identifying the condition function to check.
    inp: dictionary of input variables. Expected keys are 'r_s', 'zeta', 
      's', 'alpha', or 'q'.
    num_blocks: split the data into num_blocks blocks (based on memory 
      constraints) to check the condition.
    tol: tolerance for the condition. If None, use default tolerance.

  Returns:
    df: pandas dataframe with the following columns: "xc" [str], 
      "condition" [str], "satisfied" [Bool], "percent_violated" [float], 
      "r_s_range" [[float, float]], "zeta_range" [[float, float]], 
      "s_range" [[float, float]], "alpha_range" [[float, float]], 
      "q_range": [[float, float]].
  """

  condition = condition_string_to_fun(condition_string)
  func_id = func_id.lower()
  df = {
      'xc': [func_id],
      'condition': [condition_string],
      'satisfied': [],
      'percent_violated': [],
  }
  range_labels = [key + '_range' for key in inp]
  for label in range_labels:
    df[label] = []

  r_s = inp['r_s']
  zeta = inp['zeta']

  if condition_string == 'deriv_upper_bd_check_1':
    # add r_s = 100 (to approximate r_s -> \infty)
    r_s = np.append(r_s, 100)

  if len(inp) == 2:
    # LDA
    std_inp = [r_s, zeta]
    s = None
  elif len(inp) == 3:
    # GGA
    s = inp['s']
    std_inp = [r_s, s, zeta]
  elif len(inp) == 4:
    # MGGA w/o laplacian
    s = inp['s']
    alpha = inp['alpha']
    df['q_range'] = ['---']
    std_inp = [r_s, s, zeta, alpha]
  elif len(inp) == 5:
    # MGGA w/ laplacian
    s = inp['s']
    alpha = inp['alpha']
    q = inp['q']
    std_inp = [r_s, s, zeta, alpha, q]

  # total number of condition checks. Keep track of number of violations
  num_checks = np.prod(np.array([var.shape for var in std_inp]))
  num_violated = 0
  cond_satisfied = True

  if s is None:
    s_splits = [None]
  else:
    s_splits = np.split(s, num_blocks)

  for s_split in s_splits:
    if s_split is not None:
      std_inp[1] = s_split

    split_cond_satisfied, split_num_violated, ranges = check_condition_work(
        func_id,
        condition,
        std_inp,
        tol=tol,
    )

    num_violated += split_num_violated
    if not split_cond_satisfied:
      cond_satisfied = False
      for i, r in enumerate(ranges):
        df[range_labels[i]].append(r)

  if cond_satisfied:
    for label in range_labels:
      df[label] = ['---']
  else:
    for label in range_labels:
      min_range = np.amin(df[label])
      max_range = np.amax(df[label])
      df[label] = [[min_range, max_range]]

  df['satisfied'] = [cond_satisfied]
  df['percent_violated'] = [num_violated / num_checks]
  df = pd.DataFrame.from_dict(df)
  return df


def default_search_variables(func: Functional) -> Dict[str, np.ndarray]:
  """Default input for the grid search over a given functional.
  
  Args:
    func_id: LibXC functional identifier.

  Returns:
    inp: dictionary of input variables. Expected keys are 'r_s', 'zeta', ... 
      Each key corresponds to a 1D numpy array.
  """

  family = func.family
  if family == pylibxc.flags.XC_FAMILY_LDA:
    inp = {
        'r_s': np.linspace(0.0001, 5, 10000),
        'zeta': np.linspace(0, 1, 100),
    }
  elif family == pylibxc.flags.XC_FAMILY_GGA:
    inp = {
        'r_s': np.linspace(0.0001, 5, 10000),
        's': np.linspace(0, 5, 500),
        'zeta': np.linspace(0, 1, 100),
    }
  elif family == pylibxc.flags.XC_FAMILY_MGGA and not func.needs_laplacian:
    inp = {
        'r_s': np.linspace(0.0001, 5, 5000),
        's': np.linspace(0, 5, 100),
        'zeta': np.linspace(0, 1, 20),
        'alpha': np.linspace(0, 5, 100),
    }
  elif family == pylibxc.flags.XC_FAMILY_MGGA and func.needs_laplacian:
    inp = {
        'r_s': np.linspace(0.0001, 5, 3000),
        's': np.linspace(0, 5, 100),
        'zeta': np.linspace(0, 1, 20),
        'alpha': np.linspace(0, 5, 10),
        'q': np.linspace(0, 5, 50),
    }
  else:
    raise ValueError(f"functional not supported.")

  return inp


def default_input_grid_search(func_id: str) -> Dict[str, np.ndarray]:
  """Default input for the grid search over a given functional.
  
  Args:
    func_id: LibXC functional identifier.

  Returns:
    inp: dictionary of input variables. Expected keys are 'r_s', 'zeta', ... 
      Each key corresponds to a 1D numpy array.
  """

  libxc_fun = pylibxc.LibXCFunctional(func_id, "polarized")

  if 'mgga_c_' in func_id or 'mgga_xc_' in func_id:
    if libxc_fun._needs_laplacian:
      inp = {
          'r_s': np.linspace(0.0001, 5, 3000),
          's': np.linspace(0, 5, 100),
          'zeta': np.linspace(0, 1, 20),
          'alpha': np.linspace(0, 5, 10),
          'q': np.linspace(0, 5, 50),
      }
    else:
      inp = {
          'r_s': np.linspace(0.0001, 5, 5000),
          's': np.linspace(0, 5, 100),
          'zeta': np.linspace(0, 1, 20),
          'alpha': np.linspace(0, 5, 100),
      }
  elif 'gga_c_' in func_id or 'gga_xc_' in func_id:
    inp = {
        'r_s': np.linspace(0.0001, 5, 500),
        's': np.linspace(0, 5, 500),
        'zeta': np.linspace(0, 1, 100),
    }

  return inp


def available_conditions() -> Dict[str, Callable]:
  """Get available conditions dictionary. """

  conditions = {
      "negativity_check": negativity_check,
      "deriv_lower_bd_check": deriv_lower_bd_check,
      "deriv_upper_bd_check_1": deriv_upper_bd_check_1,
      "deriv_upper_bd_check_2": deriv_upper_bd_check_2,
      "second_deriv_check": second_deriv_check,
      "lieb_oxford_bd_check_Uxc": lieb_oxford_bd_check_Uxc,
      "lieb_oxford_bd_check_Exc": lieb_oxford_bd_check_Exc,
  }

  return conditions


def condition_string_to_fun(condition_string: str) -> Callable:
  """Get condition function (callable) from identifying string. """

  conditions = available_conditions()

  try:
    cond_fun = conditions[condition_string]
  except KeyError:
    lined_conds = '\n'.join(list(conditions.keys()))
    raise NotImplementedError(
        f'Condition not implemented: {condition_string} \n'
        'Available conditions: \n'
        f'{lined_conds}')

  return cond_fun


def lieb_oxford_bd_check_Uxc(
    std_inp: List[np.ndarray],
    f_x_c: Tuple[np.ndarray, np.ndarray],
    r_s_dx: float,
    tol: Optional[float] = 1e-3,
    lieb_oxford_bd_const: Optional[float] = 2.27,
) -> Tuple[bool, int, Union[None, Tuple[List[float]]]]:
  """Local condition for Lieb-Oxford (LO) bound on U_xc.

  F_xc +  r_s (d F_c / dr_s) <= lieb_oxford_bd_const
  """

  r_s_mesh = std_inp[0]
  f_x, f_c = f_x_c
  f_c_deriv = np.gradient(f_c, r_s_dx, edge_order=2, axis=0)
  f_xc = f_c + f_x

  regions = np.where(
      (r_s_mesh * f_c_deriv) + f_xc > lieb_oxford_bd_const + tol,
      True,
      False,
  )

  # finite differences at end points may be inaccurate
  cond_satisfied = not np.any(regions[3:-3])
  num_violated = np.sum(regions[3:-3])

  if not cond_satisfied:
    ranges = ([np.amin(feature[regions]),
               np.amax(feature[regions])] for feature in std_inp)
  else:
    ranges = None

  return cond_satisfied, num_violated, ranges


def lieb_oxford_bd_check_Exc(
    std_inp: List[np.ndarray],
    f_x_c: Tuple[np.ndarray, np.ndarray],
    r_s_dx: float,
    tol: Optional[float] = 1e-3,
    lieb_oxford_bd_const: Optional[float] = 2.27,
) -> Tuple[bool, int, Union[None, Tuple[List[float]]]]:
  """Local condition for Lieb-Oxford bound on E_xc:

  F_xc <= C 
  """

  # r_s_dx not used in this condition, but included for consistency with other
  # conditions.
  del r_s_dx

  f_x, f_c = f_x_c
  f_xc = f_c + f_x

  regions = np.where(
      f_xc > lieb_oxford_bd_const + tol,
      True,
      False,
  )

  # finite differences at end points may be inaccurate
  cond_satisfied = not np.any(regions[3:-3])
  num_violated = np.sum(regions[3:-3])

  if not cond_satisfied:
    ranges = ([np.amin(feature[regions]),
               np.amax(feature[regions])] for feature in std_inp)
  else:
    ranges = None

  return cond_satisfied, num_violated, ranges


def deriv_lower_bd_check(
    std_inp: List[np.ndarray],
    f_c: Tuple[np.ndarray, np.ndarray],
    r_s_dx: float,
    tol: Optional[float] = 1e-5,
) -> Tuple[bool, int, Union[None, Tuple[List[float]]]]:
  """Local condition for E_c[n_\gamma] scaling inequalities 
  (and T_c[n] non-negativity).

  0 <= d F_c / dr_s
  """
  # r_s_dx not used in this condition, but included for consistency with other
  # conditions.
  del r_s_dx

  regions = np.diff(f_c, axis=0)
  regions = np.where(regions < -tol, True, False)
  regions = regions.flatten()

  cond_satisfied = not np.any(regions)
  num_violated = np.sum(regions)

  if not cond_satisfied:
    # remove first entry
    std_inp = (feature[:-1].flatten() for feature in std_inp)
    ranges = ([np.amin(feature[regions]),
               np.amax(feature[regions])] for feature in std_inp)
  else:
    ranges = None

  return cond_satisfied, num_violated, ranges


def deriv_upper_bd_check_1(
    std_inp: List[np.ndarray],
    f_c: Tuple[np.ndarray, np.ndarray],
    r_s_dx: float,
    tol: Optional[float] = 1e-3,
) -> Tuple[bool, int, Union[None, Tuple[List[float]]]]:
  """Local condition for the T_c[n] upper bound exact condition.

  d F_c / dr_s <= (F_c[r_s->\infty, ...] - F_c[r_s, ...]) / r_s 
  """

  r_s_mesh = std_inp[0]
  f_c_inf = f_c[-1]
  f_c = f_c[:-1]
  r_s_mesh = r_s_mesh[:-1]

  f_c_deriv = np.gradient(f_c, r_s_dx, edge_order=2, axis=0)
  regions = np.where(
      (r_s_mesh * f_c_deriv) - (f_c_inf - f_c) > tol,
      True,
      False,
  )

  # finite differences at end points may be inaccurate
  cond_satisfied = not np.any(regions[3:-3])
  num_violated = np.sum(regions[3:-3])

  if not cond_satisfied:
    ranges = ([np.amin(feature[:-1][regions]),
               np.amax(feature[:-1][regions])] for feature in std_inp)
  else:
    ranges = None

  return cond_satisfied, num_violated, ranges


def deriv_upper_bd_check_2(
    std_inp: List[np.ndarray],
    f_c: Tuple[np.ndarray, np.ndarray],
    r_s_dx: float,
    tol: Optional[float] = 5e-4,
) -> Tuple[bool, int, Union[None, Tuple[List[float]]]]:
  """Local condition for the unproven conjectured inequality T_c[n] <= -E_c[n]. 

  d F_c / dr_s <= F_c / r_s .
  """

  r_s_mesh = std_inp[0]

  regions_grad = np.gradient(f_c, r_s_dx, edge_order=2, axis=0)
  regions = np.where(
      (r_s_mesh * regions_grad) - f_c > tol,
      True,
      False,
  )

  # finite differences at end points may be inaccurate
  cond_satisfied = not np.any(regions[3:-3])
  num_violated = np.sum(regions[3:-3])

  if not cond_satisfied:
    ranges = ([np.amin(feature[regions]),
               np.amax(feature[regions])] for feature in std_inp)
  else:
    ranges = None

  return cond_satisfied, num_violated, ranges


def second_deriv_check(
    std_inp: List[np.ndarray],
    f_c: Tuple[np.ndarray, np.ndarray],
    r_s_dx: float,
    tol: Optional[float] = 1e-3,
) -> Tuple[bool, int, Union[None, Tuple[List[float]]]]:
  """Local condition for the concavity exact condition for correlation energy
  adiabatic connection curves.
  
  d^2 F_c / dr_s^2 >= (-2/r_s) d F_c / dr_s  
  """

  r_s_mesh = std_inp[0]

  f_c_grad = np.gradient(f_c, r_s_dx, edge_order=2, axis=0)
  f_c_2grad = np.diff(f_c, 2, axis=0) / (r_s_dx**2)

  r_s_mesh = r_s_mesh[1:-1]
  f_c_grad = f_c_grad[1:-1]
  regions = np.where(
      (r_s_mesh * f_c_2grad) + (2 * f_c_grad) < -tol,
      True,
      False,
  )

  # finite differences at end points may be inaccurate
  cond_satisfied = not np.any(regions[3:-3])
  num_violated = np.sum(regions[3:-3])

  if not cond_satisfied:
    ranges = ([
        np.amin(feature[1:-1][regions]),
        np.amax(feature[1:-1][regions])
    ] for feature in std_inp)
  else:
    ranges = None

  return cond_satisfied, num_violated, ranges


def negativity_check(
    std_inp: List[np.ndarray],
    f_c: Tuple[np.ndarray, np.ndarray],
    r_s_dx: float,
    tol: Optional[float] = 1e-5,
) -> Tuple[bool, int, Union[None, Tuple[List[float]]]]:
  """Local condition for the correlation energy negativity exact condition.
  
  F_c >= 0 
  """

  # r_s_dx not used in this condition, but included for consistency with other
  # conditions.
  del r_s_dx

  regions = np.where(
      f_c < -tol,
      True,
      False,
  )

  cond_satisfied = not np.any(regions)
  num_violated = np.sum(regions)

  if not cond_satisfied:
    ranges = ([np.amin(feature[regions]),
               np.amax(feature[regions])] for feature in std_inp)
  else:
    ranges = None

  return cond_satisfied, num_violated, ranges
