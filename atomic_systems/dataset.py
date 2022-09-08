from __future__ import annotations
import yaml
from typing import List, Dict, Callable, Optional

import numpy as np
from pyscf import gto, cc, scf


class System(dict):
  """
    Interface to the system in the dataset.
    No scientific calculation should be performed in this class.
    Please do not initialize this class directly, instead, use
    ``System.create()``.
    """

  created_systems: Dict[str, System] = {}

  @classmethod
  def create(cls, system: Dict) -> System:
    # create the system if it has not been created
    # otherwise, return the previously created system

    system_str = str(system)
    if system_str not in cls.created_systems:
      cls.created_systems[system_str] = System(system)
    return cls.created_systems[system_str]

  def __init__(self, system: Dict):
    super().__init__(system)

    # caches
    self._caches = {}

  def get_pyscf_system(self):
    # convert the system dictionary PySCF system

    systype = self["type"]
    if systype == "mol":
      kwargs = self["kwargs"]
      return gto.M(atom=kwargs["moldesc"],
                   basis=kwargs["basis"],
                   spin=kwargs.get("spin", 0),
                   unit="Bohr",
                   charge=kwargs.get("charge", 0))
    else:
      raise RuntimeError("Unknown system type: %s" % systype)

  def get_cache(self, s: str):
    return self._caches.get(s, None)

  def set_cache(self, s: str, obj) -> None:
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
    # return the id of the datasets that passes the filter function
    return [i for (i, obj) in enumerate(self.obj) if filtfcn(obj)]


class Entry(dict):
  """
  Interface to the entry of the dataset.
  Entry class should not be initialized directly, but created through
  ``Entry.create``
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
    """
    Returns the list of systems in the entry
    """
    return self._systems


class EntryIE(Entry):
  """Entry for Ionization Energy (IE)"""

  @property
  def entry_type(self) -> str:
    return "ie"

  def get_val(self, mfs: List):
    tot_energies = np.array([mf.e_tot for mf in mfs])
    ie = np.dot(np.array(self["dotvec"]), tot_energies)
    return ie

  def get_true_val(self):
    return self["true_val"]


if __name__ == '__main__':
  from evaluator import PyscfEvaluator

  def ha_to_ev(ha_en):
    return 27.2114 * ha_en

  dset = Dataset('ie_atoms.yaml')
  evl = PyscfEvaluator('hf')

  df = {}
  for i in range(len(dset)):
    curr_calc = dset[i]
    label = curr_calc["name"].split(' ')[-1]

    error = evl.get_error(dset[i])
    df[label] = error

  print(df)
