from __future__ import annotations
from typing import Dict, Union, List, Optional
import sys

import numpy as np
from pyscf import gto, dft, lib, cc, scf
from pyscf.dft import numint

from dataset import Entry, System
from exact_conds import CondChecker


class PyscfEvaluator():

  def __init__(self, xc, scf_args={}):
    xc = xc.lower()
    self.xc = xc
    self.scf_args = scf_args
    self.mfs = None

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
      mf = self.use_scf_args(mf)
      mf.kernel()
    elif self.calc == "hf":
      # HF calculation

      if mol.spin == 0:
        mf = scf.RHF(mol)
      else:
        mf = scf.UHF(mol)
      mf = self.use_scf_args(mf)
      mf.kernel()

    return mf

  def use_scf_args(self, mf):

    for key in self.scf_args:
      setattr(mf, key, self.scf_args[key])

    return mf

  def get_mfs(self, entry: Union[Entry, Dict]):

    if self.mfs is not None:
      return

    self.mfs = []
    for system in entry.get_systems():
      mf = self.run(system)
      if not mf.converged:
        raise ValueError(f'SCF cycle did not converge for system: {system}')

      self.mfs.append(mf)

    return

  def reset_mfs(self):
    self.mfs = None

  def evaluate(self, entry: Union[Entry, Dict]):

    self.get_mfs(entry)

    val = entry.get_val(self.mfs)
    return val

  def get_error(self, entry: Union[Entry, Dict]):
    val = self.evaluate(entry)
    return val - entry.get_true_val()

  def exact_cond_checks(
      self,
      entry: Union[Entry, Dict],
      gams=np.linspace(0.01, 2),
  ):

    self.get_mfs(entry)

    sys_checks = []
    for mf in self.mfs:
      mf.xc = self.xc
      checker = CondChecker(mf, gams)
      checks = checker.check_conditions()
      sys_checks.append(checks)

    return sys_checks