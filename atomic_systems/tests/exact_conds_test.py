import pytest
import sys

import numpy as np
from pyscf import gto, dft, lib, cc, scf
from pyscf.dft import numint

from atomic_systems.exact_conds import CondChecker

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

xcs = [
    ('scan', 'mgga_c_scan'),
    ('pbe', 'gga_c_pbe'),
    ('gga_x_am05,gga_c_am05', 'gga_c_am05'),
    ('b3lyp', '.81 * LYP + .19 * VWN'),
]
gams = np.linspace(0.01, 2)


def test_ec_consistency():
  """ Test different ways of obtaining Ec:
    1) From explicit corr. functional.
    2) By taking the limit definition for Ex.
    
  """

  for mol in mols:
    for xc, c in xcs:
      if mol.spin == 0:
        mf = dft.RKS(mol)
      else:
        mf = dft.UKS(mol)
      mf.xc = xc
      mf.kernel()
      checker = CondChecker(mf, gams)

      # from explicit corr. functional
      checker.xc = f',{c}'
      ec_gams_1 = checker.get_Exc_gams(gams)

      # from limit definition
      checker.xc = xc
      ec_gams_2 = checker.get_Ec_gams(gams)

      np.testing.assert_allclose(ec_gams_1, ec_gams_2, atol=1e-3)


def test_exact_conds():
  for mol in mols:
    for xc, _ in xcs:
      if mol.spin == 0:
        mf = dft.RKS(mol)
      else:
        mf = dft.UKS(mol)
      mf.xc = xc
      mf.kernel()
      checker = CondChecker(mf, gams)
      check = checker.check_conditions()

      all_true = np.all([satisfied for cond, satisfied in check.items()])
      assert all_true