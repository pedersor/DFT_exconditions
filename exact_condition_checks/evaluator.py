from __future__ import annotations
from typing import Dict, Union, List, Optional
import sys

import numpy as np
from pyscf import gto, dft, lib, cc, scf
from pyscf.dft import numint

from dataset import Entry, System
from exact_conds import CondChecker


class PyscfEvaluator():

  def __init__(self, xc=None, hf=False, scf_args=[{}]):
    self.xc = xc
    self.hf = hf
    self.scf_args = scf_args
    self.grids_level = 6
    self.grids_prune = None

    # cached mf objects
    self.mfs = None
    self.non_scf_mfs = None

  @property
  def xc(self):
    return self._xc

  @xc.setter
  def xc(self, xc):
    if xc is None:
      self._xc = None
    else:
      xc = xc.lower()
      self._xc = xc
      self.non_scf_mfs = None

  @property
  def scf_args(self):
    return self._scf_args

  @scf_args.setter
  def scf_args(self, scf_args):
    if not isinstance(scf_args, list):
      scf_args = [scf_args]
    # append additional scf args to try
    scf_args.extend(self._other_scf_args())
    self._scf_args = scf_args

  def _other_scf_args(self):
    """ Additional scf args to try. """
    other_scf_args = [{'DIIS': scf.ADIIS, 'diis_space': 12}, 'newton']
    return other_scf_args

  def run(self, system: System):

    mol = system.get_pyscf_system()
    if self.hf:
      # HF calculation
      if mol.spin == 0 and mol.charge != -1:
        mf = scf.RHF(mol)
      else:
        mf = scf.UHF(mol)
    else:
      # KS-DFT calculation
      if mol.spin == 0:
        mf = dft.RKS(mol)
      else:
        mf = dft.UKS(mol)
      mf.xc = self.xc
      mf.grids.level = self.grids_level
      mf.grids.prune = self.grids_prune

    # try default scf_args and others
    for scf_args in self.scf_args:
      mf = self._use_scf_args(mf, scf_args)
      mf.kernel()
      if mf.converged:
        break

    return mf

  def run_non_scf(self, system: System, init_dm):
    mol = system.get_pyscf_system()
    if mol.spin == 0 and mol.charge != -1:
      mf = dft.RKS(mol)
    else:
      mf = dft.UKS(mol)

    mf.grids.level = self.grids_level
    mf.grids.prune = self.grids_prune
    mf.xc = self.xc
    mf.init_guess = init_dm
    mf.max_cycle = 0
    mf.kernel()
    return mf

  def _use_scf_args(self, mf, scf_args):

    if scf_args == 'newton':
      return mf.newton()

    for key in scf_args:
      setattr(mf, key, scf_args[key])

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

  def get_non_scf_mfs(self, entry: Union[Entry, Dict]):

    if self.non_scf_mfs is not None:
      return

    self.get_mfs(entry)

    self.non_scf_mfs = []
    for system, mf in zip(entry.get_systems(), self.mfs):
      init_dm = mf.make_rdm1()
      non_scf_mf = self.run_non_scf(system, init_dm)
      self.non_scf_mfs.append(non_scf_mf)

    return

  def reset_mfs(self):
    self.mfs = None
    self.non_scf_mfs = None

  def evaluate(self, entry: Union[Entry, Dict]):
    val = entry.get_val(self)
    return val

  def get_error(self, entry: Union[Entry, Dict]):
    val = self.evaluate(entry)
    return val - entry.get_true_val()

  def get_exact_cond_checks(
      self,
      entry: Union[Entry, Dict],
      gams=np.linspace(0.01, 2),
  ):

    return entry.exact_cond_checks(self, gams, self.xc)
