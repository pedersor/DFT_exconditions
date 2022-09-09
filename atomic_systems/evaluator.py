from __future__ import annotations
from typing import Dict, Union, List, Optional
import sys

sys.path.append('../')

import numpy as np
from pyscf import gto, dft, lib, cc, scf
from pyscf.dft import numint

import utils
from dataset import Entry, System
from exact_conds import CondChecker


class PyscfEvaluator():

  def __init__(self, xc, c=None):
    xc = xc.lower()
    self.xc = xc
    self.c = c

    if xc == "ccsd":
      self.calc = "ccsd"
    elif xc == 'hf':
      self.calc = "hf"
    else:
      self.calc = "ksdft"

  def run(self, system: System):

    mol = system.get_pyscf_system()
    if self.calc == "ksdft":
      # KS-DFT calculation

      if mol.spin == 0:
        mf = dft.RKS(mol)
      else:
        mf = dft.UKS(mol)
      mf.xc = self.xc
      mf.kernel()
      return mf

    elif self.calc == "hf":
      # HF calculation

      if mol.spin == 0:
        mf = scf.RHF(mol).run()
      else:
        mf = scf.UHF(mol).run()

      return mf

  def evaluate(self, entry: Union[Entry, Dict]):
    mfs = [self.run(system) for system in entry.get_systems()]
    val = entry.get_val(mfs)
    return val

  def get_error(self, entry: Union[Entry, Dict]):
    val = self.evaluate(entry)
    return val - entry.get_true_val()

  def exact_cond_checks(
      self,
      entry: Union[Entry, Dict],
      gams=np.linspace(0.01, 2),
  ):
    if self.c is None:
      raise ValueError('specify correlation functional for condition checks.')

    mfs = [self.run(system) for system in entry.get_systems()]

    sys_checks = []
    for mf in mfs:
      mf.xc = self.c
      checker = CondChecker(mf, gams)
      checks = checker.check_conditions()
      sys_checks.append(checks)

    return sys_checks