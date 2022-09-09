import copy

import numpy as np

from pyscf import gto, dft, lib, cc, scf
from pyscf.dft import numint


class CondChecker():
  """ check conditions on self-consistent densities. """

  def __init__(self, mf, gams=np.linspace(0.01, 2)):
    self.mf = mf
    self.gams = gams
    self.mol = mf.mol
    self.xc = mf.xc
    self.xctype = mf._numint._xc_type(mf.xc)
    self.weights = mf.grids.weights
    self.nelec = self.mol.nelec
    self.spin = self.mol.spin

    # setup density (rho)
    ao_value = numint.eval_ao(self.mol, mf.grids.coords, deriv=2)
    if self.spin == 0:
      dm = mf.make_rdm1()
      self.rho = numint.eval_rho(mol, ao_value, dm, xctype=self.xctype)
    else:
      dm_up, dm_dn = mf.make_rdm1()
      rho_up = numint.eval_rho(mol, ao_value, dm_up, xctype=self.xctype)
      rho_dn = numint.eval_rho(mol, ao_value, dm_dn, xctype=self.xctype)
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

  def get_Exc_gam(self, gam):

    scaled_rho, scaled_weights = self.get_scaled_sys(gam)
    eps_xc = dft.libxc.eval_xc(self.xc, scaled_rho, spin=self.spin)[0]

    if self.spin == 0:
      rho = scaled_rho[0]
      int_nelec = np.einsum('i,i->', rho, scaled_weights)
      nelec = sum(list(self.mol.nelec))
      np.testing.assert_allclose(int_nelec, nelec)

      exc = np.einsum('i,i,i->', eps_xc, rho, scaled_weights)

    else:
      rho_up = scaled_rho[0][0]
      rho_dn = scaled_rho[1][0]

      int_nelec_up = np.einsum('i,i->', rho_up, scaled_weights)
      np.testing.assert_allclose(int_nelec_up, self.nelec[0])
      int_nelec_dn = np.einsum('i,i->', rho_dn, scaled_weights)
      np.testing.assert_allclose(int_nelec_dn, self.nelec[1])

      exc = np.einsum('i,i,i->', eps_xc, rho_up + rho_dn, scaled_weights)

    return exc

  def get_Exc_gams(self):
    exc_gams = np.array([self.get_Exc_gam(gam) for gam in self.gams])
    return exc_gams


if __name__ == '__main__':

  import matplotlib.pyplot as plt

  mol = gto.M(
      atom='He 0 0 0',
      basis='ccpv5z',
      spin=0,
  )

  xc = 'pbe'
  mf = dft.RKS(mol)
  mf.xc = xc
  mf.kernel()
  mf.xc = ',pbe'

  gams = np.linspace(0.01, 2)
  checker = CondChecker(mf, gams)

  exc_gams = checker.get_Exc_gams()

  plt.plot(gams, exc_gams)
