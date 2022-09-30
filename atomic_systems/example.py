import matplotlib.pyplot as plt
import numpy as np
from pyscf import gto, dft, lib, cc, scf
from pyscf.dft import numint

from exact_conds import CondChecker

li_mol = gto.M(
    atom='Li 0 0 0',
    basis='aug-pcseg-4',
    spin=1,
)

mf = scf.UHF(li_mol)
mf.kernel()

checker = CondChecker(mf, xc='HF')

s_grids, g3_s = checker.reduced_grad_dist()

plt.plot(s_grids, g3_s)
plt.savefig('g3_s.pdf', bbox_inches='tight')
