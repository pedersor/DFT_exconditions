from __future__ import annotations
import yaml
from typing import List, Dict, Callable, Union

import numpy as np
from pyscf import gto

from dft_exconditions.exact_condition_checks import CondChecker

HAR_TO_EV = 27.2114
HAR_TO_KCAL = 627.5


class System(dict):
  """
    Interface to the system in the dataset.
    No calculations are performed in this class.
    To initialize, use `System.create()`.
    """

  created_systems: Dict[str, System] = {}

  @classmethod
  def create(cls, system: Dict) -> System:
    """Create the system if it has not been created already. 
    Else, return the previously created system."""

    system_str = str(system)
    if system_str not in cls.created_systems:
      cls.created_systems[system_str] = System(system)
    return cls.created_systems[system_str]

  def __init__(self, system: Dict):
    super().__init__(system)

    # caches
    self._caches = {}

  def get_pyscf_system(self):
    """Convert the system dictionary to PySCF system."""

    systype = self["type"]
    if systype == "mol":
      kwargs = self["kwargs"]
      return gto.M(
          atom=kwargs["moldesc"],
          basis=kwargs["basis"],
          spin=kwargs.get("spin", 0),
          unit="Bohr",
          charge=kwargs.get("charge", 0),
      )
    else:
      raise RuntimeError(f"Unknown system type: {systype}")

  def get_cache(self, s: str) -> Union[object, None]:
    """Generic cache getter."""
    return self._caches.get(s, None)

  def set_cache(self, s: str, obj: object) -> None:
    """Generic cache setter."""
    self._caches[s] = obj


class Dataset():

  def __init__(self, fpath):
    with open(fpath, "r") as f:
      self.obj = [Entry.create(a) for a in yaml.safe_load(f)]

  def __len__(self) -> int:
    return len(self.obj)

  def __getitem__(self, i: int) -> Entry:
    return self.obj[i]

  def get_indices(self, filtfcn: Callable[[Dict], bool]) -> List[int]:
    """ Return the id of the datasets that passes the filter function. """
    return [i for (i, obj) in enumerate(self.obj) if filtfcn(obj)]


class Entry(dict):
  """
  Interface to the entry of the dataset.
  Entry class should not be initialized directly, but created through
  `Entry.create`.
  """

  created_entries: Dict[str, Entry] = {}

  @classmethod
  def create(cls, entry_dct):

    s = str(entry_dct)
    if s not in cls.created_entries:
      tpe = entry_dct["type"]
      kwargs = {
          "entry_dct": entry_dct,
      }
      obj = {
          "ie": EntryIE,
          "ea": EntryEA,
      }[tpe](**kwargs)
      cls.created_entries[s] = obj
    return cls.created_entries[s]

  def __init__(
      self,
      entry_dct: Dict,
  ):
    super().__init__(entry_dct)
    self._systems = [System.create(p) for p in entry_dct["systems"]]

  def get_systems(self) -> List[System]:
    """Returns the list of systems in the entry."""
    return self._systems


class EntryIE(Entry):
  """Entry for ionization energy (IE)"""

  @property
  def entry_type(self) -> str:
    return "ie"

  def get_val(self, evl: object, use_non_scf: bool = False) -> float:
    """Obtain the IE value from the system."""

    if use_non_scf:
      evl.get_non_scf_mfs(self)
      mfs = evl.non_scf_mfs
    else:
      evl.get_mfs(self)
      mfs = evl.mfs

    tot_energies = np.array([mf.e_tot for mf in mfs])
    ie = np.dot(np.array(self["dotvec"]), tot_energies)
    return ie

  def get_true_val(self) -> float:
    """Retrieve the true reference IE value from the entry."""
    return self["true_val"]

  def exact_cond_checks(
      self,
      evl: object,
      gams: np.ndarray,
      xc: str,
      use_non_scf: bool = False,
  ) -> List[Dict]:
    """Perform exact condition checks for the entry.
    
    Args:
      evl: evaluator object (e.g., PyscfEvaluator).
      gams: array of gamma values to test.
      xc: exchange-correlation functional to use.
    
    Returns:
      List of dictionaries, each containing the results of the exact 
        condition check.
    """

    if use_non_scf:
      evl.get_non_scf_mfs(self)
      mfs = evl.non_scf_mfs
    else:
      evl.get_mfs(self)
      mfs = evl.mfs

    sys_checks = []
    for mf in mfs:
      mf.xc = xc
      checker = CondChecker(mf, gams=gams)
      checks = checker.check_conditions()
      sys_checks.append(checks)

    return sys_checks


class EntryEA(Entry):
  """Entry for Electron affinities (EA)"""

  @property
  def entry_type(self) -> str:
    return "ea"

  def get_val(self, evl):
    """Obtain the EA value from the system."""

    evl.get_non_scf_mfs(self)
    tot_energies = np.array([mf.e_tot for mf in evl.non_scf_mfs])
    ea = np.dot(np.array(self["dotvec"]), tot_energies)
    return ea

  def get_true_val(self):
    """Retrieve the true reference EA value from the entry."""
    return self["true_val"]

  def exact_cond_checks(
      self,
      evl: object,
      gams: np.ndarray,
      xc: str,
  ) -> List[Dict]:
    """Perform exact condition checks for the entry.

    Args:
      evl: evaluator object (e.g., PyscfEvaluator).
      gams: array of gamma values to test.
      xc: exchange-correlation functional to use.
    
    Returns:
      List of dictionaries, each containing the results of the exact 
        condition check.
    """

    evl.get_non_scf_mfs(self)

    sys_checks = []
    for mf in evl.non_scf_mfs:
      mf.xc = xc
      checker = CondChecker(mf, gams=gams)
      checks = checker.check_conditions()
      sys_checks.append(checks)

    return sys_checks
