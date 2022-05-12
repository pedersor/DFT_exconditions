import numpy
from pyscf import gto, dft, lib
from pyscf.dft import numint

mol_hf = gto.M(atom='He 0 0 0', basis='ccpvdz')
mf_hf = dft.RKS(mol_hf)
mf_hf.xc = 'hf'
mf_hf.kernel()
dm = mf_hf.make_rdm1()

# dummy calculation to get larger range of grids and weights
mol = gto.M(atom='Be 0 0 0', basis='ccpvdz')
mf = dft.RKS(mol)
mf.xc = 'lda,vwn'
mf.kernel()
coords = mf.grids.coords
weights = mf.grids.weights

gam = 0.4
# Use default mesh grids and weights
scaled_coords = gam * coords
ao_value = numint.eval_ao(mol_hf, scaled_coords, deriv=2)
# The first row of rho is electron density, the rest three rows are electron
# density gradients which are needed for GGA functional
rho = numint.eval_rho(mol_hf, ao_value, dm, xctype='MGGA')
rho[0] = (gam**3) * rho[0]
rho[1:4] = (gam**4) * rho[1:4]
if rho.ndim > 4:
  rho[4:] = (gam**5) * rho[4:]

print(rho.shape)

#ex, vx = dft.libxc.eval_xc('B88', rho)[:2]
#ec, vc = dft.libxc.eval_xc(',P86', rho)[:2]

ec, vc = dft.libxc.eval_xc(',scan', rho)[:2]

print('integral check: %.12f' % numpy.einsum('i,i->', rho[0], weights))
print('Ec[n_gam] = %.12f' % numpy.einsum('i,i,i->', ec, rho[0], weights))

print()
