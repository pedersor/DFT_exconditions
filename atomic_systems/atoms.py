import sys

sys.path.append('../')

import numpy as np
from pyscf import gto, dft, lib, cc, scf
from pyscf.dft import numint

import utils


def ha_to_ev(ha_en):

  return 27.2114 * ha_en


if __name__ == '__main__':
  ip = 0

  mol = gto.M(
      atom='He 0 0 0',
      basis='aug-pcseg-4',
      charge=0,
      spin=0,
  )

  mf = scf.HF(mol).run()
  ip = -mf.e_tot

  mol = gto.M(
      atom='He 0 0 0',
      basis='aug-pcseg-4',
      charge=1,
      spin=1,
  )

  mf = scf.UHF(mol).run()

  ip += mf.e_tot

  print(ha_to_ev(ip))

  print()