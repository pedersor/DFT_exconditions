import pylibxc
import numpy as np
import matplotlib.pyplot as plt

# start defintions ====


def get_density(r_s):
  """ Obtains density n from r_s."""
  return 3 / (4 * np.pi * (r_s**3))


def hartree_to_mRy(energy):
  """ Hartree to mRy units conversion."""

  return energy * 2 * 1000


def get_eps_x_unif(n):
  """ Uniform gas exchange energy per particle. """
  return -(3 / (4 * np.pi)) * ((n * 3 * np.pi**2)**(1 / 3))


def get_grad_n(s, n):
  """ Obtain |\nabla n| from reduced gradient, s. """
  return s * (2 * ((3 * np.pi**2)**(1 / 3)) * (n**(4 / 3)))


def get_up_dn_density(n, zeta):
  """Obtains [n_up, n_dn]"""

  n = np.expand_dims(n, axis=1)
  zeta = np.expand_dims(zeta, axis=1)

  up_coeff, dn_ceoff = zeta_coeffs(zeta)
  up_density = up_coeff * n
  dn_density = dn_ceoff * n

  return np.concatenate((up_density, dn_density), axis=1)


def zeta_coeffs(zeta):
  """ Coefficients relating n_sigma to zeta."""

  return (1 + zeta) / 2, (1 - zeta) / 2


def get_tau(alpha, grad_n, n, zeta):
  """ Obtains [tau_up, tau_dn] where 
  tau_sigma = 1/2 \sum_i^occ |\nabla \phi_{sigma i}|^2 ."""

  tau_w = (grad_n**2) / (8 * n)
  tau_unif = (3 / 10) * ((3 * np.pi**2)**(2 / 3)) * (n**(5 / 3))
  d_s = ((1 + zeta)**(5 / 3) + (1 - zeta)**(5 / 3)) / 2
  tau_unif *= d_s
  tau = (alpha * tau_unif) + tau_w

  up_coeff, dn_coeff = (
      np.expand_dims(coeff, axis=1) for coeff in zeta_coeffs(zeta))
  tau = np.expand_dims(tau, axis=1)
  tau = np.concatenate((up_coeff * tau, dn_coeff * tau), axis=1)

  return tau


def get_sigma(grad_n, zeta):
  """ Obtains [\nabla n_up * \nabla n_up, \nabla n_up * \nabla n_dn, \nabla n_dn * \nabla n_dn]"""

  sigma = np.expand_dims(grad_n**2, axis=1)

  up_coeff, dn_coeff = (
      np.expand_dims(coeff, axis=1) for coeff in zeta_coeffs(zeta))

  sigma = np.concatenate(
      (up_coeff**2 * sigma, up_coeff * dn_coeff * sigma, dn_coeff**2 * sigma),
      axis=1)

  return sigma


def get_lapl(q, n, zeta):
  """ Obtains laplacian [\nabla^2 n_up, \nabla^2 n_dn]. 
  q is reduced density Laplacian. Eq. 14 in PhysRevA.96.052512"""

  n = np.expand_dims(n, axis=1)
  q = np.expand_dims(q, axis=1)
  zeta = np.expand_dims(zeta, axis=1)
  lapl = q * 4 * (3 * np.pi**2)**(2 / 3) * (n**(5 / 3))

  up_coeff, dn_ceoff = zeta_coeffs(zeta)
  up_lapl = up_coeff * lapl
  dn_lapl = dn_ceoff * lapl

  return np.concatenate((up_lapl, dn_lapl), axis=1)


# end defintions ====


def lda_c(func_c, r_s, zeta):
  """ Obtains correlation energy per particle for LDA-type functionals: 

  \epsilon_c^{LDA}(r_s, \zeta) .

  """

  input = (r_s, zeta)
  input = (feature.flatten() for feature in input)
  r_s, zeta = input

  # obtain libxc inputs
  n = get_density(r_s)
  rho = get_up_dn_density(n, zeta)

  inp = {}
  inp["rho"] = rho

  func_c_res = func_c.compute(inp)
  eps_c = np.squeeze(func_c_res['zk'])

  return eps_c


def gga_c(func_c, r_s, s, zeta):
  """ Obtains correlation energy per particle for GGA-type functionals:

  \epsilon_c^{GGA}(r_s, s, \zeta, \alpha) .

  """

  input = (r_s, s, zeta)
  input = (feature.flatten() for feature in input)
  r_s, s, zeta = input

  # obtain libxc inputs
  n = get_density(r_s)
  grad_n = get_grad_n(s, n)
  rho = get_up_dn_density(n, zeta)
  sigma = get_sigma(grad_n, zeta)

  inp = {}
  inp["rho"] = rho
  inp["sigma"] = sigma

  func_c_res = func_c.compute(inp)
  eps_c = np.squeeze(func_c_res['zk'])

  return eps_c


def mgga_c(func_c, r_s, s, zeta, alpha, q=None):
  """ Obtains correlation energy per particle for MGGA-type functionals 
  (without laplacian):
  
  \epsilon_c^{MGGA}(r_s, s, \zeta, \alpha) .
  
  """

  input = (r_s, s, zeta, alpha)
  input = (feature.flatten() for feature in input)
  r_s, s, zeta, alpha = input

  # obtain libxc inputs
  n = get_density(r_s)
  grad_n = get_grad_n(s, n)
  rho = get_up_dn_density(n, zeta)
  tau = get_tau(alpha, grad_n, n, zeta)
  sigma = get_sigma(grad_n, zeta)

  inp = {}
  inp["rho"] = rho
  inp["sigma"] = sigma
  inp["tau"] = tau

  func_c_res = func_c.compute(inp)
  eps_c = np.squeeze(func_c_res['zk'])

  return eps_c


def mgga_c_lapl(func_c, r_s, s, zeta, alpha, q):
  """ Obtains correlation energy per particle for MGGA-type functionals 
  with laplacian:

  \epsilon_c^{MGGA}(r_s, s, \zeta, \alpha, q) .

  """

  input = (r_s, s, zeta, alpha, q)
  input = (feature.flatten() for feature in input)
  r_s, s, zeta, alpha, q = input

  # obtain libxc inputs
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

  func_c_res = func_c.compute(inp)
  eps_c = np.squeeze(func_c_res['zk'])

  return eps_c


def get_eps_c(func_id, input):

  func_c = pylibxc.LibXCFunctional(func_id, "polarized")

  if 'mgga_c_' in func_id:
    if func_c._needs_laplacian:
      return mgga_c_lapl(func_c, *input)
    else:
      return mgga_c(func_c, *input)
  elif 'gga_c_' in func_id:
    return gga_c(func_c, *input)
  elif 'lda_c_' in func_id:
    return lda_c(func_c, *input)

  return NotImplementedError(f"functional {func_id} not supported.")


def check_condition(func_id, condition, input, tol=None):

  r_s = input[0]
  if condition.__name__ == 'deriv_upper_bd_check_1':
    # add r_s = 100 (to approximate r_s -> \infty)
    input[0] = np.append(r_s, 100)

  r_s_dx = r_s[1] - r_s[0]
  input = np.meshgrid(*input, indexing='ij')
  eps_c = get_eps_c(func_id, input)

  if tol:
    result = condition(input, eps_c, r_s_dx, tol)
  else:
    # use default tol for condition
    result = condition(input, eps_c, r_s_dx)

  return result


def deriv_lower_bd_check(input, eps_c, r_s_dx, tol=1e-5):
  """
  0 <= d F_c / dr_s
  """

  n = get_density(input[0])
  eps_x_unif = get_eps_x_unif(n)

  f_c = eps_c.reshape(input[0].shape) / eps_x_unif
  regions = np.diff(f_c, axis=0)

  regions = np.where(regions < -tol, True, False)
  regions = regions.flatten()

  cond_satisfied = not np.any(regions)

  if not cond_satisfied:
    # remove first entry
    input = (feature[:-1].flatten() for feature in input)
    ranges = ([np.amin(feature[regions]),
               np.amax(feature[regions])] for feature in input)
  else:
    ranges = None

  return cond_satisfied, ranges


def deriv_upper_bd_check_1(input, eps_c, r_s_dx, tol=1e-3):
  """ 
  d F_c / dr_s <= (F_c[r_s->\infty, ...] - F_c[r_s, ...]) / r_s 
  """
  r_s_mesh = input[0]
  n = get_density(r_s_mesh)
  eps_x_unif = get_eps_x_unif(n)

  f_c = eps_c.reshape(r_s_mesh.shape) / eps_x_unif
  f_c_inf = f_c[-1]
  f_c = f_c[:-1]
  r_s_mesh = r_s_mesh[:-1]

  f_c_deriv = np.gradient(f_c, r_s_dx, edge_order=2, axis=0)
  up_bd_regions = np.where(
      (r_s_mesh * f_c_deriv) - (f_c_inf - f_c) > tol,
      True,
      False,
  )

  # finite differences at end points may be inaccurate
  cond_satisfied = not np.any(up_bd_regions[3:-3])

  if not cond_satisfied:
    ranges = ([
        np.amin(feature[:-1][up_bd_regions]),
        np.amax(feature[:-1][up_bd_regions])
    ] for feature in input)
  else:
    ranges = None

  return cond_satisfied, ranges


def deriv_upper_bd_check_2(input, eps_c, r_s_dx, tol=1e-3):
  """ 
  d F_c / dr_s <= F_c / r_s .
  
  Note: sufficient condition to satisfy the unproven result T_c[n] <= -E_c[n].
  """

  r_s_mesh = input[0]
  n = get_density(r_s_mesh)
  eps_x_unif = get_eps_x_unif(n)

  f_c = eps_c.reshape(r_s_mesh.shape) / eps_x_unif

  regions_grad = np.gradient(f_c, r_s_dx, edge_order=2, axis=0)
  up_bd_regions = np.where(
      (r_s_mesh * regions_grad) - f_c > tol,
      True,
      False,
  )

  # finite differences at end points may be inaccurate
  cond_satisfied = not np.any(up_bd_regions[3:-3])

  if not cond_satisfied:
    ranges = ([
        np.amin(feature[up_bd_regions]),
        np.amax(feature[up_bd_regions])
    ] for feature in input)
  else:
    ranges = None

  return cond_satisfied, ranges


def second_deriv_check(input, eps_c, r_s_dx, tol=1e-3):
  """ d^2 F_c / dr_s^2 >= (-2/r_s) d F_c / dr_s . """

  r_s_mesh = input[0]
  n = get_density(r_s_mesh)
  eps_x_unif = get_eps_x_unif(n)

  f_c = eps_c.reshape(r_s_mesh.shape) / eps_x_unif

  f_c_grad = np.gradient(f_c, r_s_dx, edge_order=2, axis=0)
  f_c_2grad = np.diff(f_c, 2, axis=0) / (r_s_dx**2)

  r_s_mesh = r_s_mesh[1:-1]
  f_c_grad = f_c_grad[1:-1]
  up_bd_regions = np.where(
      (r_s_mesh * f_c_2grad) + (2 * f_c_grad) < -tol,
      True,
      False,
  )

  # finite differences at end points may be inaccurate
  cond_satisfied = not np.any(up_bd_regions[3:-3])

  if not cond_satisfied:
    ranges = ([
        np.amin(feature[1:-1][up_bd_regions]),
        np.amax(feature[1:-1][up_bd_regions])
    ] for feature in input)
  else:
    ranges = None

  return cond_satisfied, ranges


def negativity_check(input, eps_c, r_s_dx, tol=1e-5):
  """ F_c >= 0 ."""

  r_s_mesh = input[0]
  eps_c = eps_c.reshape(r_s_mesh.shape)

  regions = np.where(
      eps_c > tol,
      True,
      False,
  )

  cond_satisfied = not np.any(regions)

  if not cond_satisfied:
    ranges = ([np.amin(feature[regions]),
               np.amax(feature[regions])] for feature in input)
  else:
    ranges = None

  return cond_satisfied, ranges


if __name__ == '__main__':

  r_s = np.linspace(0.0001, 2, 1000)
  s = np.linspace(0, 5, 50)
  zeta = np.linspace(0, 1, 50)

  # note that order must be in the form
  input = [r_s, s, zeta]
  cond_satisfied, ranges = check_condition(
      "gga_c_pbe",
      deriv_upper_bd_check_1,
      input,
  )

  print(f"Condition satisified: {cond_satisfied}")
  if ranges is not None:
    for r in ranges:
      print(r)
