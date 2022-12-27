import pytest

import numpy as np
from pyscf import gto, dft

from dft_exconditions.exact_conditions.exact_conds import CondChecker

test_mols = [
    # fully spin-polarized system
    gto.M(
        atom='Li 0 0 0',
        basis='ccpv5z',
        spin=1,
    ),
    # spin-unpolarized system
    gto.M(
        atom='He 0 0 0',
        basis='ccpv5z',
        spin=0,
    ),
]

test_xcs = [
    ('hyb_mgga_x_m08_hx,mgga_c_m08_hx', 'mgga_c_m08_hx'),
    ('m06', 'mgga_c_m06'),
    ('scan', 'mgga_c_scan'),
    ('pbe', 'gga_c_pbe'),
    ('gga_x_am05,gga_c_am05', 'gga_c_am05'),
    ('b3lyp', '.81 * LYP + .19 * VWN'),
]

test_gams = [np.linspace(0.01, 2)]


@pytest.mark.parametrize('func_ids', test_xcs)
@pytest.mark.parametrize('mol', test_mols)
@pytest.mark.parametrize('gams', test_gams)
def test_ec_consistency(func_ids, mol, gams):
  """Test different ways of obtaining Ec:
    1) From explicit corr. functional.
    2) By taking the limit definition for Ex.
    
    We should obtain same result from either 1) or 2).
  """

  xc, c = func_ids
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


@pytest.mark.parametrize('func_ids', test_xcs[2:])
@pytest.mark.parametrize('mol', test_mols)
@pytest.mark.parametrize('gams', test_gams)
def test_exact_conds(func_ids, mol, gams):
  """Test that the exact conditions are satisfied for a given xc functional."""

  xc = func_ids[0]

  if mol.spin == 0:
    mf = dft.RKS(mol)
  else:
    mf = dft.UKS(mol)
  mf.xc = xc
  mf.kernel()
  checker = CondChecker(mf, gams=gams)
  check = checker.check_conditions()

  assert np.all([satisfied for _, satisfied in check.items()])
