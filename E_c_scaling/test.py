import numpy as np
from pyscf import gto, dft, lib, cc, scf
from pyscf.dft import numint


def e_c_check(mol, base_mf, ccsd_mf, gams, xc='pbe', xctype='GGA'):

  dm = ccsd_mf.make_rdm1()

  coords = base_mf.grids.coords
  weights = base_mf.grids.weights

  e_c_gam = []
  numint_check = []
  for gam in gams:

    # Use default mesh grids and weights
    scaled_coords = gam * coords
    ao_value = numint.eval_ao(mol, scaled_coords, deriv=2)
    # The first row of rho is electron density, the rest three rows are electron
    # density gradients which are needed for GGA functional
    rho = numint.eval_rho(mol, ao_value, dm, xctype=xctype)
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

  title = 'h2'
  gams = np.linspace(0.2, 2, num=20)

  # CCSD(T) calculation
  mol = gto.M(
      atom='H 0 0 0;H 0 0 0.74',  # in Angstrom
      basis='ccpv5z',
      symmetry=True)
  mf = scf.HF(mol).run()
  ccsd_mf = cc.CCSD(mf).run()
  et = ccsd_mf.ccsd_t()
  ccsd_t_en = ccsd_mf.e_tot + et
  e_c = ccsd_t_en - mf.e_tot

  # dummy calculation to get larger range of grids and weights
  tmp_mol = gto.M(atom='Be 0 0 0;Be 0 0 0.74', basis='ccpv5z')
  base_mf = dft.RKS(tmp_mol)
  base_mf.xc = 'lda,vwn'
  base_mf.max_cycle = 1
  base_mf.kernel()

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
    e_c_gam = e_c_check(mol, base_mf, mf, gams, xc=xc, xctype=xctype)
    plt.plot(gams, e_c_gam, label=xc)

  plt.plot(gams, e_c * gams, color='black', label='$\gamma E^*_c $')
  plt.axvline(x=1, alpha=0.4, color='k', linestyle='--')

  plt.legend()
  plt.title(title)
  plt.xlabel('$\gamma$')
  plt.xlim(left=0)
  plt.ylim(top=0)
  plt.grid(alpha=0.2)
  title = title.replace(' ', '_')
  plt.savefig(f'{title}.pdf', bbox_inches='tight')
