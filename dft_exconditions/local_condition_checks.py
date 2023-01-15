from collections.abc import Callable
from typing import List, Optional, Tuple, Dict, Union

import numpy as np
import pandas as pd

import pylibxc

R_S_INF = 100
INF_GAM = 5000


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


class Functional():

  def __init__(self, func_id: str) -> None:
    """Initialize a functional object.

    Args:
      func_id: Libxc functional identifier string.
    """

    func_id = func_id.lower()
    self.func_id = func_id
    self.is_combined_xc = '_xc_' in func_id

    if self.is_combined_xc:
      self.libxc_fun = self._get_libxc_fun(func_id)
      self.name = self.get_name(self.libxc_fun._xc_func_name)
      self.family = self.libxc_fun._family
      self.needs_laplacian = self.libxc_fun._needs_laplacian
      self._get_hybrid_variables(self.libxc_fun)
      self.dfa_type = self._get_dfa_type()
      # callable density functional approximation (DFA) function to use
      self.dfa_fun = self._get_dfa_fun()

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
      self.family = libxc_fun_c._family
      self.needs_laplacian = (
          (libxc_fun_c is not None and libxc_fun_c._needs_laplacian) or
          (libxc_fun_x is not None and libxc_fun_x._needs_laplacian))
      self._get_hybrid_variables(libxc_fun_x)
      self.dfa_type = self._get_dfa_type()
      # callable density functional approximation (DFA) function to use
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

  def _get_dfa_type(self):

    flag_to_dfa_type = {
        pylibxc.flags.XC_FAMILY_LDA: 'LDA',
        pylibxc.flags.XC_FAMILY_GGA: 'GGA',
        pylibxc.flags.XC_FAMILY_MGGA: 'MGGA',
    }

    hyb_str = 'HYB_' if self.hyb_exx_coef > 0 or self.range_sep_hyb else ''
    dfa_type = hyb_str + flag_to_dfa_type[self.family]
    return dfa_type

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


class LocalCondChecker():

  def __init__(
      self,
      functional: Functional,
      conditions_to_check: List[str] = None,
      vars_to_check: Dict[str, np.ndarray] = None,
  ):
    self.functional = functional
    # Reuse enhancement factors and input meshses for evaluating
    # different conditions.
    self.curr_f_x_c = None
    self.curr_inp_mesh = None
    self.curr_f_c_inf = None

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
  ) -> Tuple[np.ndarray, np.ndarray]:
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
      mesh_grid with shape (mesh_shape,) and (X)C enhancement factor on the 
      grid with shape (mesh_shape,).
    """

    dfa = functional.dfa_fun

    if self.curr_f_x_c is not None:
      # reuse enhancement factors for evaluating different conditions
      return self.curr_inp_mesh, self.curr_f_x_c, self.curr_f_c_inf
    elif functional.is_combined_xc:
      # add a large r_s value to approximate r_s -> \infty)
      r_s_inp = np.append(std_inp[0], R_S_INF)
      inp_mesh = np.array(np.meshgrid(r_s_inp, *std_inp[1:], indexing='ij'))

      # xc functional without explicit partitioning -> use conventional
      # partitioning.
      libxc_fun_xc = functional.libxc_fun
      eps_xc = dfa(libxc_fun_xc, *inp_mesh)
      f_xc = eps_to_enh_factor(inp_mesh, eps_xc)

      # obtain "conventional" partitioning by taking an approximate limit
      # using a large \gamma value.
      scaled_r_s_inp = r_s_inp / INF_GAM
      scaled_inp_mesh = np.meshgrid(scaled_r_s_inp, *std_inp[1:], indexing='ij')
      eps_x = dfa(libxc_fun_xc, *scaled_inp_mesh) / INF_GAM
      f_x = eps_to_enh_factor(inp_mesh, eps_x)
      f_c = f_xc - f_x

      # remove r_s -> \infty value from inp_mesh
      inp_mesh, _ = np.split(inp_mesh, [-1], axis=1)
      f_c, f_c_inf = np.split(f_c, [-1])
      self.curr_f_x_c = f_x[:-1], f_c
      self.curr_f_c_inf = f_c_inf
      self.curr_inp_mesh = inp_mesh
      return self.curr_inp_mesh, self.curr_f_x_c, self.curr_f_c_inf
    else:
      # add a large r_s value to approximate r_s -> \infty)
      r_s_inp = np.append(std_inp[0], R_S_INF)
      inp_mesh = np.array(np.meshgrid(r_s_inp, *std_inp[1:], indexing='ij'))

      # obtain enhancement factors for X and C separately
      libxc_fun_x, libxc_fun_c = functional.libxc_fun

      if libxc_fun_x:
        eps_x = dfa(libxc_fun_x, *inp_mesh)
        f_x = eps_to_enh_factor(inp_mesh, eps_x)
        # remove r_s -> \infty value from f_x
        f_x = f_x[:-1]
      else:
        f_x = None
      if libxc_fun_c:
        eps_c = dfa(libxc_fun_c, *inp_mesh)
        f_c = eps_to_enh_factor(inp_mesh, eps_c)
        # separate r_s -> \infty value from f_c
        f_c, f_c_inf = np.split(f_c, [-1])

      else:
        f_c = None
        f_c_inf = None

      # remove r_s -> \infty value from inp_mesh
      inp_mesh, _ = np.split(inp_mesh, [-1], axis=1)

      self.curr_f_x_c = f_x, f_c
      self.curr_inp_mesh = inp_mesh
      self.curr_f_c_inf = f_c_inf
      return self.curr_inp_mesh, self.curr_f_x_c, self.curr_f_c_inf

  def check_condition_work(
      self,
      functional: Functional,
      condition: str,
      std_inp: List[np.ndarray],
      tol: Optional[float] = None,
  ) -> Tuple:

    r_s = std_inp[0]
    r_s_dx = r_s[1] - r_s[0]

    inp_mesh, f_x_c, f_c_inf = self.get_enh_factor_x_c(functional, std_inp)

    condition_fun = condition_str_to_fun(condition)

    std_kwargs = {
        'std_inp': inp_mesh,
        'f_x_c': f_x_c,
        'r_s_dx': r_s_dx,
        'f_c_inf': f_c_inf,
        'xc_lieb_oxford_coef': 2.27,
        'x_lieb_oxford_coef': 1.174,
        'hyb_exx_coef': functional.hyb_exx_coef,
    }
    if tol:
      std_kwargs['tol'] = tol

    result = condition_fun(**std_kwargs)
    return result

  def check_conditions(
      self,
      num_blocks: Optional[int] = 100,
  ) -> pd.DataFrame:

    num_conditions = len(self.conditions_to_check)
    df = {
        'xc id': [self.functional.func_id] * num_conditions,
        'xc label': [self.functional.name] * num_conditions,
        'dfa type': [self.functional.dfa_type] * num_conditions,
        'condition': [None] * num_conditions,
        'satisfied': [None] * num_conditions,
        'fraction violated': [0] * num_conditions,
    }

    std_var_input = ['r_s', 's', 'zeta', 'alpha', 'q']
    std_var_input = [var + '_range' for var in std_var_input]
    for label in std_var_input:
      # note: each of the lists is unique
      # (i.e. not a reference to the same list)
      df[label] = [[] for _ in range(num_conditions)]

    s = self.vars_to_check.get('s', None)
    std_inp = get_standard_input(self.vars_to_check)

    # total number of condition checks. Keep track of number of violations
    num_checks = np.prod(np.array([var.shape for var in std_inp]))

    if s is None:
      s_splits = [None]
    else:
      s_splits = np.split(s, num_blocks)

    for s_split in s_splits:
      if s_split is not None:
        std_inp[1] = s_split

      # reset evaluated enhancement factor values for new s_split
      self.curr_f_x_c = None
      self.curr_inp_mesh = None
      self.curr_f_c_inf = None
      for i, condition in enumerate(self.conditions_to_check):
        df['condition'][i] = condition

        split_cond_satisfied, split_num_violated, ranges = self.check_condition_work(
            self.functional,
            condition,
            std_inp,
        )

        df['fraction violated'][i] += split_num_violated

        if not split_cond_satisfied:
          for j, r in enumerate(ranges):
            df[std_var_input[j]][i].append(r)

    for i in range(num_conditions):

      for label in std_var_input:
        if len(df[label][i]) == 0:
          df[label][i] = None
          continue
        min_range = np.amin(df[label][i])
        max_range = np.amax(df[label][i])
        df[label][i] = [min_range, max_range]

      df['satisfied'][i] = df['fraction violated'][i] == 0
      df['fraction violated'][i] /= num_checks

    df = pd.DataFrame.from_dict(df)
    return df


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


def condition_str_to_fun(condition_str: str) -> Callable:
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

  cond_fun = conditions.get(condition_str, None)
  if cond_fun is None:
    lined_conds = '\n'.join(list(conditions.keys()))
    raise NotImplementedError(f'Condition not implemented: {condition_str} \n'
                              'Available conditions: \n'
                              f'{lined_conds}')

  return cond_fun


def lieb_oxford_bd_check_Uxc(
    std_inp: List[np.ndarray],
    f_x_c: Tuple[np.ndarray, np.ndarray],
    r_s_dx: float,
    hyb_exx_coef: float,
    tol: Optional[float] = 1e-3,
    xc_lieb_oxford_coef: Optional[float] = 2.27,
    x_lieb_oxford_coef: Optional[float] = 1.174,
    **kwargs,
) -> Tuple[bool, int, Union[None, Tuple[List[float]]]]:
  """Local condition for Lieb-Oxford (LO) bound on U_xc.

  F_xc +  r_s (d F_c / dr_s) <= xc_lieb_oxford_coef
  """
  del kwargs
  r_s_mesh = std_inp[0]
  f_x, f_c = f_x_c
  f_c_deriv = np.gradient(f_c, r_s_dx, edge_order=2, axis=0)
  f_xc = f_c + f_x

  regions = np.where(
      r_s_mesh * f_c_deriv + f_xc >
      (xc_lieb_oxford_coef - hyb_exx_coef * x_lieb_oxford_coef) + tol,
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
    hyb_exx_coef: float,
    tol: Optional[float] = 1e-3,
    xc_lieb_oxford_coef: Optional[float] = 2.27,
    x_lieb_oxford_coef: Optional[float] = 1.174,
    **kwargs,
) -> Tuple[bool, int, Union[None, Tuple[List[float]]]]:
  """Local condition for Lieb-Oxford bound on E_xc:

  F_xc <= C 
  """

  del kwargs
  f_x, f_c = f_x_c
  f_xc = f_c + f_x

  regions = np.where(
      f_xc > (xc_lieb_oxford_coef - hyb_exx_coef * x_lieb_oxford_coef) + tol,
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
    f_x_c: Tuple[np.ndarray, np.ndarray],
    tol: Optional[float] = 1e-5,
    **kwargs,
) -> Tuple[bool, int, Union[None, Tuple[List[float]]]]:
  """Local condition for E_c[n_\gamma] scaling inequalities 
  (and T_c[n] non-negativity).

  0 <= d F_c / dr_s
  """

  del kwargs
  f_c = f_x_c[1]

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
    f_x_c: Tuple[np.ndarray, np.ndarray],
    f_c_inf: np.ndarray,
    r_s_dx: float,
    tol: Optional[float] = 1e-3,
    **kwargs,
) -> Tuple[bool, int, Union[None, Tuple[List[float]]]]:
  """Local condition for the T_c[n] upper bound exact condition.

  d F_c / dr_s <= (F_c[r_s->\infty, ...] - F_c[r_s, ...]) / r_s 
  """
  del kwargs
  f_c = f_x_c[1]

  r_s_mesh = std_inp[0]

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
    ranges = ([np.amin(feature[regions]),
               np.amax(feature[regions])] for feature in std_inp)
  else:
    ranges = None

  return cond_satisfied, num_violated, ranges


def deriv_upper_bd_check_2(
    std_inp: List[np.ndarray],
    f_x_c: Tuple[np.ndarray, np.ndarray],
    r_s_dx: float,
    tol: Optional[float] = 5e-4,
    **kwargs,
) -> Tuple[bool, int, Union[None, Tuple[List[float]]]]:
  """Local condition for the unproven conjectured inequality T_c[n] <= -E_c[n]. 

  d F_c / dr_s <= F_c / r_s .
  """
  del kwargs
  f_c = f_x_c[1]
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
    f_x_c: Tuple[np.ndarray, np.ndarray],
    r_s_dx: float,
    tol: Optional[float] = 1e-3,
    **kwargs,
) -> Tuple[bool, int, Union[None, Tuple[List[float]]]]:
  """Local condition for the concavity exact condition for correlation energy
  adiabatic connection curves.
  
  d^2 F_c / dr_s^2 >= (-2/r_s) d F_c / dr_s  
  """

  del kwargs
  f_c = f_x_c[1]
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
    f_x_c: Tuple[np.ndarray, np.ndarray],
    tol: Optional[float] = 1e-5,
    **kwargs,
) -> Tuple[bool, int, Union[None, Tuple[List[float]]]]:
  """Local condition for the correlation energy negativity exact condition.
  
  F_c >= 0 
  """

  del kwargs
  f_c = f_x_c[1]

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
