import numpy as np
from pyscf import gto, dft, lib
from pyscf.dft import numint


def he_check(gams, xc='pbe', xctype='GGA'):

  mol_hf = gto.M(atom='He 0 0 0', basis='ccpvdz')
  mf_hf = dft.RKS(mol_hf)
  mf_hf.xc = 'hf'
  mf_hf.kernel()
  dm = mf_hf.make_rdm1()

  # dummy calculation to get larger range of grids and weights
  mol = gto.M(atom='Be 0 0 0', basis='ccpvdz')
  mf = dft.RKS(mol)
  mf.xc = 'lda,vwn'
  mf.kernel()
  coords = mf.grids.coords
  weights = mf.grids.weights

  e_c_gam = []
  numint_check = []
  for gam in gams:

    # Use default mesh grids and weights
    scaled_coords = gam * coords
    ao_value = numint.eval_ao(mol_hf, scaled_coords, deriv=2)
    # The first row of rho is electron density, the rest three rows are electron
    # density gradients which are needed for GGA functional
    rho = numint.eval_rho(mol_hf, ao_value, dm, xctype=xctype)
    rho[0] = (gam**3) * rho[0]
    rho[1:4] = (gam**4) * rho[1:4]
    if rho.shape[0] > 4:
      rho[4:] = (gam**5) * rho[4:]

    e_c = dft.libxc.eval_xc(f',{xc}', rho)[0]

    numint_check.append(np.einsum('i,i->', rho[0], weights))
    e_c_gam.append(np.einsum('i,i,i->', e_c, rho[0], weights))

  # TODO: check numint against nelec
  print('min numint: ', min(numint_check))

  return e_c_gam


if __name__ == '__main__':
  import matplotlib.pyplot as plt

  title = 'He atom'
  gams = np.linspace(0.15, 2, num=20)

  Ec_ref = -0.042

  xcs = [
      ('pbe', 'GGA'),
      ('P86', 'GGA'),
      ('lyp', 'GGA'),
      ('scan', 'MGGA'),
      ('m05', 'MGGA'),
      ('m11', 'MGGA'),
      ('mn15', 'MGGA'),
  ]
  for xc, xctype in xcs:
    e_c_gam = he_check(gams, xc=xc, xctype=xctype)
    plt.plot(gams, e_c_gam, label=xc)

  plt.plot(gams, Ec_ref * gams, color='black', label='$\gamma E^*_c $')
  plt.axvline(x=1, alpha=0.4, color='k', linestyle='--')

  plt.legend()
  plt.title(title)
  plt.xlabel('$\gamma$')
  plt.xlim(left=0)
  plt.ylim(top=0)
  plt.grid(alpha=0.2)
  plt.savefig('idk.pdf', bbox_inches='tight')
