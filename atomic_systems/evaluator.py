from __future__ import annotations
from typing import Dict, Union, List, Optional
import sys

sys.path.append('../')

import numpy as np
from pyscf import gto, dft, lib, cc, scf
from pyscf.dft import numint

import utils
from dataset import Entry, System


class PyscfEvaluator():

  def __init__(self, xc):
    xc = xc.lower()
    self.xc = xc

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