import pylibxc
import numpy as np
import pandas as pd

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


def lda_xc(func_c, r_s, zeta):
  """ Obtains correlation energy per particle for LDA-type functionals: 

  \epsilon_c^{LDA}(r_s, \zeta) .

  """
  mesh_shape = r_s.shape
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

  return eps_c.reshape(mesh_shape)


def gga_xc(func_c, r_s, s, zeta):
  """ Obtains correlation energy per particle for GGA-type functionals:

  \epsilon_c^{GGA}(r_s, s, \zeta, \alpha) .

  """
  mesh_shape = r_s.shape
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

  return eps_c.reshape(mesh_shape)


def mgga_xc(func_c, r_s, s, zeta, alpha, q=None):
  """ Obtains correlation energy per particle for MGGA-type functionals 
  (without laplacian):
  
  \epsilon_c^{MGGA}(r_s, s, \zeta, \alpha) .
  
  """
  mesh_shape = r_s.shape
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

  return eps_c.reshape(mesh_shape)


def mgga_xc_lapl(func_c, r_s, s, zeta, alpha, q):
  """ Obtains correlation energy per particle for MGGA-type functionals 
  with laplacian:

  \epsilon_c^{MGGA}(r_s, s, \zeta, \alpha, q) .

  """

  mesh_shape = r_s.shape
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

  return eps_c.reshape(mesh_shape)


def eps_to_enh_factor(input_mesh, eps_x_c):
  """ converts \epsilon_(x)c to F_(x)c ."""

  r_s_mesh = input_mesh[0]
  n = get_density(r_s_mesh)
  eps_x_unif = get_eps_x_unif(n)

  return eps_x_c / eps_x_unif


def get_dfa_rung(func_id):
  dfa_rungs = {
      "lda_": lda_xc,
      "mgga_": mgga_xc,
      "gga_": gga_xc,
      "hyb_mgga_": mgga_xc,
      "hyb_gga_": gga_xc,
  }

  for dfa in dfa_rungs:
    if dfa in func_id:
      return dfa_rungs[dfa]

  return NotImplementedError(f"functional {func_id} not supported.")


def is_hybrid(func_id):
  return 'hyb_' in func_id


def get_enh_factor_x_c(func_id, input):

  dfa = get_dfa_rung(func_id)

  # hyb_c_func workaround
  if is_hybrid(func_id) and '_c_' in func_id:
    func_id = func_id.replace('_c_', '_xc_')
    func_xc = pylibxc.LibXCFunctional(func_id, "polarized")
    if func_xc._needs_laplacian:
      dfa = mgga_xc_lapl

    input_mesh = np.meshgrid(*input, indexing='ij')
    eps_xc = dfa(func_xc, *input_mesh)
    f_xc = eps_to_enh_factor(input_mesh, eps_xc)

    # substract off exchange
    zero_r_s = np.array([0.00001])
    input_mesh = np.meshgrid(zero_r_s, *input[1:], indexing='ij')
    eps_x = dfa(func_xc, *input_mesh)
    f_x = eps_to_enh_factor(input_mesh, eps_x)

    f_c = f_xc - f_x
    return f_c
  elif is_hybrid(func_id) and '_x_' in func_id:
    func_id = func_id.replace('_x_', '_xc_')
    func_xc = pylibxc.LibXCFunctional(func_id, "polarized")
    if func_xc._needs_laplacian:
      dfa = mgga_xc_lapl

    zero_r_s = np.array([0.00001])
    input_mesh = np.meshgrid(zero_r_s, *input[1:], indexing='ij')
    eps_x = dfa(func_xc, *input_mesh)
    f_x = eps_to_enh_factor(input_mesh, eps_x)
    return f_x
  else:
    func_xc = pylibxc.LibXCFunctional(func_id, "polarized")
    if func_xc._needs_laplacian:
      dfa = mgga_xc_lapl
    input_mesh = np.meshgrid(*input, indexing='ij')

    eps_x_c = dfa(func_xc, *input_mesh)
    f_x_c = eps_to_enh_factor(input_mesh, eps_x_c)
    return f_x_c


def check_condition_work(
    func_id,
    condition,
    input,
    tol=None,
):

  r_s = input[0]
  r_s_dx = r_s[1] - r_s[0]

  if condition.__name__ == 'deriv_upper_bd_check_1':
    # add r_s = 100 (to approximate r_s -> \infty)
    input[0] = np.append(r_s, 100)

  if 'lieb_oxford_bd_check' in condition.__name__:
    f_c = get_enh_factor_x_c(func_id, input)

    # get exchange
    func_id_x = func_id.replace('_c_', '_x_')
    f_x = get_enh_factor_x_c(func_id_x, input)
    f_x_c = (f_x, f_c)
  else:
    f_x_c = get_enh_factor_x_c(func_id, input)

  input_mesh = np.meshgrid(*input, indexing='ij')
  if tol:
    result = condition(input_mesh, f_x_c, r_s_dx, tol)
  else:
    # use default tol for condition
    result = condition(input_mesh, f_x_c, r_s_dx)

  return result


def condition_string_to_fun(condition_string):
  """ get condition function from identifying string. """

  conditions = {
      "negativity_check": negativity_check,
      "deriv_lower_bd_check": deriv_lower_bd_check,
      "deriv_upper_bd_check_1": deriv_upper_bd_check_1,
      "deriv_upper_bd_check_2": deriv_upper_bd_check_2,
      "second_deriv_check": second_deriv_check,
      "lieb_oxford_bd_check_Uxc": lieb_oxford_bd_check_Uxc,
      "lieb_oxford_bd_check_Exc": lieb_oxford_bd_check_Exc,
  }

  return conditions[condition_string]


def check_condition(
    func_id,
    condition_string,
    input,
    num_splits=100,
    tol=None,
):

  func_id = func_id.lower().replace('_xc_', '_c_')

  df = {
      'xc': [func_id],
      'satisfied': [],
      'percent_violated': [],
  }

  range_labels = [key + '_range' for key in input]
  for label in range_labels:
    df[label] = []

  condition = condition_string_to_fun(condition_string)

  r_s = input['r_s']
  zeta = input['zeta']
  if len(input) == 3:
    # GGA
    s = input['s']
    std_input = [r_s, s, zeta]
  elif len(input) == 4:
    # MGGA w/o laplacian
    s = input['s']
    alpha = input['alpha']
    df['q_range'] = ['---']
    std_input = [r_s, s, zeta, alpha]
  elif len(input) == 5:
    # MGGA w/ laplacian
    s = input['s']
    alpha = input['alpha']
    q = input['q']
    std_input = [r_s, s, zeta, alpha, q]

  # total number of condition checks. Keep track of number of violations
  num_checks = np.prod(np.array([var.shape for var in std_input]))
  num_violated = 0
  cond_satisfied = True

  s_splits = np.split(s, num_splits)
  for s_split in s_splits:
    std_input[1] = s_split

    split_cond_satisfied, split_num_violated, ranges = check_condition_work(
        func_id,
        condition,
        std_input,
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


def lieb_oxford_bd_check_Uxc(
    input,
    f_x_c,
    r_s_dx,
    tol=1e-3,
    lieb_oxford_bd_const=2.27,
):
  """ 
  original Lieb-Oxford bound on Uxc:

  F_xc +  r_s (d F_c / dr_s) <= C 
  
  """
  r_s_mesh = input[0]
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
               np.amax(feature[regions])] for feature in input)
  else:
    ranges = None

  return cond_satisfied, num_violated, ranges


def lieb_oxford_bd_check_Exc(
    input,
    f_x_c,
    r_s_dx,
    tol=1e-3,
    lieb_oxford_bd_const=2.27,
):
  """ 
  Lieb-Oxford bound on Exc:

  F_xc <= C 
  
  """

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
               np.amax(feature[regions])] for feature in input)
  else:
    ranges = None

  return cond_satisfied, num_violated, ranges


def deriv_lower_bd_check(input, f_c, r_s_dx, tol=1e-5):
  """
  0 <= d F_c / dr_s
  """

  regions = np.diff(f_c, axis=0)
  regions = np.where(regions < -tol, True, False)
  regions = regions.flatten()

  cond_satisfied = not np.any(regions)
  num_violated = np.sum(regions)

  if not cond_satisfied:
    # remove first entry
    input = (feature[:-1].flatten() for feature in input)
    ranges = ([np.amin(feature[regions]),
               np.amax(feature[regions])] for feature in input)
  else:
    ranges = None

  return cond_satisfied, num_violated, ranges


def deriv_upper_bd_check_1(input, f_c, r_s_dx, tol=1e-3):
  """ 
  d F_c / dr_s <= (F_c[r_s->\infty, ...] - F_c[r_s, ...]) / r_s 
  """
  r_s_mesh = input[0]
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
               np.amax(feature[:-1][regions])] for feature in input)
  else:
    ranges = None

  return cond_satisfied, num_violated, ranges


def deriv_upper_bd_check_2(input, f_c, r_s_dx, tol=1e-3):
  """ 
  d F_c / dr_s <= F_c / r_s .
  
  Note: sufficient condition to satisfy the unproven result T_c[n] <= -E_c[n].
  """

  r_s_mesh = input[0]

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
               np.amax(feature[regions])] for feature in input)
  else:
    ranges = None

  return cond_satisfied, num_violated, ranges


def second_deriv_check(input, f_c, r_s_dx, tol=1e-3):
  """ d^2 F_c / dr_s^2 >= (-2/r_s) d F_c / dr_s . """

  r_s_mesh = input[0]

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
    ] for feature in input)
  else:
    ranges = None

  return cond_satisfied, num_violated, ranges


def negativity_check(input, f_c, r_s_dx, tol=1e-5):
  """ F_c >= 0 ."""

  regions = np.where(
      f_c < -tol,
      True,
      False,
  )

  cond_satisfied = not np.any(regions)
  num_violated = np.sum(regions)

  if not cond_satisfied:
    ranges = ([np.amin(feature[regions]),
               np.amax(feature[regions])] for feature in input)
  else:
    ranges = None

  return cond_satisfied, num_violated, ranges


if __name__ == '__main__':
  """ test check_condition. """

  input = {
      'r_s': np.linspace(0.0001, 2, 1000),
      's': np.linspace(0, 5, 200),
      'zeta': np.array([0]),
      'alpha': np.array([0.5]),
      #'q': np.array([0.5])
  }

  df = check_condition(
      "hyb_mgga_c_tpssh",
      "negativity_check",
      input,
      num_splits=1,
  )

  print(df)
