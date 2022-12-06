import pytest
import sys

sys.path.append('../../')

import numpy as np
from pyscf import gto, dft, lib, cc, scf
from pyscf.dft import numint

from exact_condition_checks.exact_conds import CondChecker

# polarized system
mol1 = gto.M(
    atom='Li 0 0 0',
    basis='ccpv5z',
    spin=1,
)
# unpolarized system
mol2 = gto.M(
    atom='He 0 0 0',
    basis='ccpv5z',
    spin=0,
)
mols = [mol1, mol2]

gams = np.linspace(0.01, 2)


def test_ec_consistency():
  """ Test different ways of obtaining Ec:
    1) From explicit corr. functional.
    2) By taking the limit definition for Ex.
    
    We should obtain same result from either 1) or 2).
  """

  xcs = [
      ('hyb_mgga_x_m08_hx,mgga_c_m08_hx', 'mgga_c_m08_hx'),
      ('m06', 'mgga_c_m06'),
      ('scan', 'mgga_c_scan'),
      ('pbe', 'gga_c_pbe'),
      ('gga_x_am05,gga_c_am05', 'gga_c_am05'),
      ('b3lyp', '.81 * LYP + .19 * VWN'),
  ]

  for mol in mols:
    for xc, c in xcs:
      if mol.spin == 0:
        mf = dft.RKS(mol)
      else:
        mf = dft.UKS(mol)
      mf.xc = xc
      mf.kernel()
      checker = CondChecker(mf, gams=gams)

      # from explicit corr. functional
      checker.xc = f',{c}'
      ec_gams_1 = checker.get_Exc_gams(gams)

      # from limit definition
      checker.xc = xc
      ec_gams_2 = checker.get_Ec_gams(gams)

      np.testing.assert_allclose(ec_gams_1, ec_gams_2, atol=1e-3)


def test_exact_conds():

  xcs = [
      ('scan', 'mgga_c_scan'),
      ('pbe', 'gga_c_pbe'),
      ('gga_x_am05,gga_c_am05', 'gga_c_am05'),
      ('b3lyp', '.81 * LYP + .19 * VWN'),
  ]

  for mol in mols:
    for xc, _ in xcs:
      if mol.spin == 0:
        mf = dft.RKS(mol)
      else:
        mf = dft.UKS(mol)
      mf.xc = xc
      mf.kernel()
      checker = CondChecker(mf, gams=gams)
      check = checker.check_conditions()

      all_true = np.all([satisfied for cond, satisfied in check.items()])
      assert all_true


if __name__ == '__main__':
  # for simple debugging
  test_ec_consistency()
  test_exact_conds()
