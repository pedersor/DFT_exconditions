import pylibxc
import numpy as np
import matplotlib.pyplot as plt

# start defintions ====


def get_density(r_s):
  return 3 / (4 * np.pi * (r_s**3))


def hartree_to_mRy(energy):
  return energy * 2 * 1000


def get_eps_x_unif(n):
  return -(3 / (4 * np.pi)) * ((n * 3 * np.pi**2)**(1 / 3))


def get_grad_n(s, n):
  return s * (2 * ((3 * np.pi**2)**(1 / 3)) * (n**(4 / 3)))


def get_up_dn_density(n, zeta):
  n = np.expand_dims(n, axis=1)
  zeta = np.expand_dims(zeta, axis=1)

  up_coeff, dn_ceoff = zeta_coeffs(zeta)
  up_density = up_coeff * n
  dn_density = dn_ceoff * n

  return np.concatenate((up_density, dn_density), axis=1)


def zeta_coeffs(zeta):
  return (1 + zeta) / 2, (1 - zeta) / 2


def get_tau(alpha, grad_n, n, zeta):
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

  sigma = np.expand_dims(grad_n**2, axis=1)

  up_coeff, dn_coeff = (
      np.expand_dims(coeff, axis=1) for coeff in zeta_coeffs(zeta))

  sigma = np.concatenate(
      (up_coeff**2 * sigma, up_coeff * dn_coeff * sigma, dn_coeff**2 * sigma),
      axis=1)

  return sigma


def get_lapl(q, n, zeta):
  # q is reduced density Laplacian. Eq. 14 in PhysRevA.96.052512

  n = np.expand_dims(n, axis=1)
  q = np.expand_dims(q, axis=1)
  zeta = np.expand_dims(zeta, axis=1)
  lapl = q * 4 * (3 * np.pi**2)**(2 / 3) * (n**(5 / 3))

  up_coeff, dn_ceoff = zeta_coeffs(zeta)
  up_lapl = up_coeff * lapl
  dn_lapl = dn_ceoff * lapl

  return np.concatenate((up_lapl, dn_lapl), axis=1)


# end defintions ====


def lda_c(func_id, r_s, zeta):
  func_c = pylibxc.LibXCFunctional(func_id, "polarized")

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


def gga_c(func_id, r_s, s, zeta):

  func_c = pylibxc.LibXCFunctional(func_id, "polarized")

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


def mgga_c(func_id, r_s, s, zeta, alpha, q=None):

  func_c = pylibxc.LibXCFunctional(func_id, "polarized")

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


def deriv_check(input, eps_c, tol=1e-5):

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


def deriv_upper_bd_check(input, eps_c, r_s_dx, tol=1e-3):

  r_s_mesh = input[0]
  n = get_density(r_s_mesh)
  eps_x_unif = get_eps_x_unif(n)

  f_c = eps_c.reshape(r_s_mesh.shape) / eps_x_unif

  regions_grad = np.gradient(f_c, r_s_dx, edge_order=2, axis=0)
  up_bd_regions = np.where(
      regions_grad - (f_c / r_s_mesh) > tol,
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


if __name__ == '__main__':
  example = 'gga'

  if example == "mgga_c_lapl":

    r_s = np.linspace(0.0001, 2, 500)

    r_s = np.linspace(0.0001, 0.1, 50)

    s = np.linspace(0, 5, 10)
    alpha = np.linspace(0, 5, 10)
    zeta = np.linspace(0, 1.0, 10)
    q = np.linspace(0, 5.0, 50)

    input = np.meshgrid(r_s, s, zeta, alpha, q, indexing='ij')

    func_id = "MGGA_C_SCANL"
    func_c = pylibxc.LibXCFunctional(func_id, "polarized")
    eps_c = mgga_c_lapl(func_c, *input)

    cond_satisfied, ranges = deriv_check(input, eps_c)

    print(cond_satisfied)
    if ranges is not None:
      for r in ranges:
        print(r)

  if example == 'lda':
    r_s = np.linspace(0.0001, 2, 5000)
    zeta = np.linspace(0, 1.0, 50)

    input = np.meshgrid(r_s, zeta, indexing='ij')

    eps_c = lda_c("lda_c_pw", *input)

    cond_satisfied, ranges = deriv_check(input, eps_c)

    print(cond_satisfied)
    if ranges is not None:
      for r in ranges:
        print(r)

  if example == 'gga':
    r_s = np.linspace(0.001, 2, 1000)
    s = np.linspace(0, 5, 50)
    zeta = np.linspace(0, 1, 50)
    input = np.meshgrid(r_s, s, zeta, indexing='ij')

    eps_c = gga_c("gga_c_pbe", *input)

    r_s_dx = r_s[1] - r_s[0]
    cond_satisfied, ranges = deriv_upper_bd_check(input, eps_c, r_s_dx)

    print(cond_satisfied)
    if ranges is not None:
      for r in ranges:
        print(r)

  if example == "mgga":

    r_s = np.linspace(0.001, 2, 200)

    s = np.linspace(0, 5, 50)
    alpha = np.linspace(0, 5, 50)
    zeta = np.linspace(0, 1.0, 50)

    input = np.meshgrid(r_s, s, zeta, alpha, indexing='ij')

    eps_c = mgga_c("MGGA_C_SCAN", *input)

    r_s_dx = r_s[1] - r_s[0]
    cond_satisfied, ranges = deriv_upper_bd_check(input, eps_c, r_s_dx)
    #cond_satisfied, ranges = deriv_check(input, eps_c)

    print(cond_satisfied)
    if ranges is not None:
      for r in ranges:
        print(r)
