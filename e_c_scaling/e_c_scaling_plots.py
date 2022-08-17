import sys

sys.path.append('../')

import numpy as np
from pyscf import gto, dft, lib, cc, scf
from pyscf.dft import numint

import utils


def get_E_c_gamma(mol, mf, gams, xc='pbe', xctype='GGA'):

  dm = mf.make_rdm1()

  # dummy calculation to get grids and weights
  base_mf = dft.RKS(mol)
  base_mf.xc = f',{xc}'
  base_mf.max_cycle = 0
  base_mf.kernel()

  coords = base_mf.grids.coords
  weights = base_mf.grids.weights

  e_c_gam = []
  nelec_check = []
  for gam in gams:

    # Use default mesh grids and weights
    scaled_coords = coords
    ao_value = numint.eval_ao(mol, scaled_coords, deriv=2)
    # The first row of rho is electron density, the rest three rows are electron
    # density gradients which are needed for GGA functional
    rho = numint.eval_rho(mol, ao_value, dm, xctype=xctype)
    s = utils.get_s(rho[0], rho[1:4])
    rho[0] = (gam**3) * rho[0]
    rho[1:4] = (gam**4) * rho[1:4]
    if rho.shape[0] > 4:
      rho[4:] = (gam**5) * rho[4:]

    eps_c = dft.libxc.eval_xc(f',{xc}', rho)[0]

    nelec_check.append(np.einsum('i,i->', rho[0], weights / gam**3))
    e_c_gam.append(np.einsum('i,i,i->', eps_c, rho[0], weights / gam**3))

  # check whether integral over density yields correct nelec
  print('min nelec: ', min(nelec_check))

  return np.asarray(e_c_gam)


def plot_E_c_gamma(mol, title, gams=np.linspace(0.01, 2, num=40)):

  # HF density calculation
  mf = scf.HF(mol).run()

  xcs = [
      ('pbe', 'GGA'),
      ('lyp', 'GGA'),
      ('scan', 'MGGA'),
      ('m06', 'MGGA'),
  ]
  for xc, xctype in xcs:
    e_c_gam = get_E_c_gamma(mol, mf, gams, xc=xc, xctype=xctype)
    plt.plot(gams, e_c_gam, label=xc)

  plt.axvline(x=1, alpha=0.4, color='k', linestyle='--')

  plt.legend()
  plt.title(title)
  plt.xlabel(r'$\gamma$')
  plt.ylabel(r'$E_c[n^{HF}_{\gamma}]$')
  plt.xlim(left=0)
  plt.ylim(top=0)
  plt.grid(alpha=0.2)
  file_name = title.replace('$', '').replace('_', '')
  plt.savefig(f'{file_name}__E_c_gamma.pdf', bbox_inches='tight')


def plot_E_c_gamma_diff(mol, title, gams=np.linspace(0.01, 2, num=40)):

  # HF density calculation
  mf = scf.HF(mol).run()

  xcs = [
      ('pbe', 'GGA'),
      ('lyp', 'GGA'),
      ('scan', 'MGGA'),
      ('m06', 'MGGA'),
  ]
  for xc, xctype in xcs:
    e_c_gam = get_E_c_gamma(mol, mf, gams, xc=xc, xctype=xctype)
    e_c = get_E_c_gamma(mol, mf, [1], xc=xc, xctype=xctype)

    plt.plot(gams, e_c_gam - gams * e_c, label=xc)

  plt.axvline(x=1, alpha=0.4, color='k', linestyle='--')
  plt.axhline(0, color='k')

  plt.legend()
  plt.title(title)
  plt.xlabel(r'$\gamma$')
  plt.ylabel(r'$E_c[n^{HF}_{\gamma}] - {\gamma} E_c[n^{HF}]$')
  plt.xlim(left=0)
  plt.grid(alpha=0.2)
  file_name = title.replace('$', '').replace('_', '')
  plt.savefig(f'{file_name}_E_c_gamma_diff.pdf', bbox_inches='tight')


def plot_E_c_deriv_gamma():
  title = 'h2'
  gams = np.linspace(0.2, 2, num=20)
  gam_dx = gams[1] - gams[0]

  # CCSD(T) calculation
  mol = gto.M(
      atom='H 0 0 0;H 0 0 0.74',  # in Angstrom
      basis='ccpv5z',
      symmetry=True)
  mf = scf.HF(mol).run()

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
    e_c_gam = get_E_c_gamma(mol, base_mf, mf, gams, xc=xc, xctype=xctype)
    p, = plt.plot(
        gams,
        e_c_gam / gams,
        '--',
        alpha=0.4,
    )
    plt.plot(
        gams,
        np.gradient(e_c_gam, gams),
        label=f'{xc} dEc/dgam',
        color=p.get_color(),
    )

  plt.axvline(x=1, alpha=0.4, color='k', linestyle='--')

  plt.legend()
  plt.title(title)
  plt.xlabel(r'$\gamma$')
  plt.xlim(left=0)
  plt.ylim(top=0)
  plt.grid(alpha=0.2)
  title = title.replace(' ', '_')
  plt.savefig(f'{title}.pdf', bbox_inches='tight')


def get_example_mol(title):

  if title == 'N$_2$':
    mol = gto.M(
        atom='N 0 0 0;N 0 0 1.09',  # in Angstrom
        basis='ccpv5z',
    )
    return mol

  elif title == 'H$_2$':
    mol = gto.M(
        atom='H 0 0 0;H 0 0 0.74',  # in Angstrom
        basis='ccpv5z',
    )
    return mol


if __name__ == '__main__':
  import matplotlib.pyplot as plt

  title = 'N$_2$'
  mol = get_example_mol(title)

  utils.use_standard_plotting_params()
  plot_E_c_gamma_diff(mol, title)
