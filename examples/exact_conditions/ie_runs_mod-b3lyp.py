import argparse
from collections import defaultdict

import pandas as pd
import numpy as np
from pyscf import dft

from dft_exconditions.evaluator import PyscfEvaluator
from dft_exconditions.dataset import Dataset
from dft_exconditions.exact_condition_checks import CondChecker

# if True, only run the first few systems.
DEBUG = True


class ModCondChecker(CondChecker):
  """Custom CondChecker class with modified B3LYP correlation energy."""

  def get_eps_xc(self, xc: str, rho: np.ndarray) -> np.ndarray:
    """Obtain exchange-correlation (XC) energy density \epsilon_{xc}[n] for a 
      given density.
      
      Args:
        xc: XC functional id.
        rho: Pyscf rho variable.

      Returns:
        eps_xc: XC energy density on a grid with shape (num_density_grids,)
    """

    s_min = 1.82
    s = self.get_reduced_grad()
    s_step_fun = np.where(s < s_min, 1, 0)

    eps_xc = dft.libxc.eval_xc(xc, rho, spin=self.unrestricted)[0]
    mod_eps_xc = eps_xc * s_step_fun

    return mod_eps_xc


def main():

  xc = 'b3lyp'
  xc_label = 'MOD-B3LYP'

  dset = Dataset('ie_atoms.yaml')

  scf_args = {
      'max_cycle': 200,
      'conv_tol': 1e-6,
      'diis_space': 12,
      'chkfile': False,
      'verbose': 4,
  }

  all_sys_errors = defaultdict(list)
  for i in range(len(dset)):
    print('i = ', i)
    # evaluate on HF densities
    evl = PyscfEvaluator(xc=None, hf=True, scf_args=scf_args)

    curr_calc = dset[i]
    label = curr_calc["name"].split(' ')[-1]
    all_sys_errors['System'].append(label)

    evl.xc = xc
    evl.get_non_scf_mfs(curr_calc)

    # use modified b3lyp correlation energy
    for i, mf in enumerate(evl.non_scf_mfs):
      checker = CondChecker(mf)
      e_c = checker.get_Exc_gam(gam=1, xc=',.81 * LYP + .19 * VWN')
      mod_checker = ModCondChecker(mf)
      mod_e_c = mod_checker.get_Exc_gam(gam=1, xc=',.81 * LYP + .19 * VWN')
      evl.non_scf_mfs[i].e_tot += -e_c + mod_e_c

    # benchmark errors
    sys_error = evl.get_error(curr_calc, use_non_scf=True)
    all_sys_errors["error"].append(sys_error)

    # remove cached mf objects
    evl.reset_mfs()

    if DEBUG and i == 3:
      break

  # benchmark errors
  errs_out_file = f'ie_errs_{xc_label}.csv'
  df = pd.DataFrame.from_dict(all_sys_errors)
  df.to_csv(errs_out_file, index=None)
  print('output energy errors to file: ', errs_out_file)


if __name__ == '__main__':
  main()