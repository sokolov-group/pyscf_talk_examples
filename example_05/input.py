import numpy as np
import math
import pyscf.gto
import pyscf.mrpt
import pyscf.scf
import pyscf.mcscf
import pyscf.fci
import pyscf.mp
import time
from pyscf.tools import molden
from pyscf.tools import cubegen
import prism.interface
import prism.mr_adc
import prism.nevpt
from prism.tools import trans_prop

np.set_printoptions(linewidth=150, edgeitems=10, suppress=True)

xyz_geom = """
Cu        10.40860       15.58560       12.97560
Cu        11.14480       18.23200       13.71070
O          9.63130       17.30350       13.12420
O         11.00730       14.86390       15.51460
O         11.74370       17.31780       16.19380
O         13.00610       16.58180       12.29760
N         11.90340       16.86770       12.48230
N         11.60110       13.97130       13.02000
N          8.85880       14.46490       13.67080
N         10.20310       19.45570       15.06040
N         12.92550       18.82230       14.36480
C         12.46140       13.69710       12.06680
C         13.27830       12.60670       12.10250
C         13.18620       11.79500       13.21950
C         12.31380       12.08690       14.27700
C         11.52050       13.19960       14.16380
C         10.52830       13.59810       15.20630
C          9.15010       13.57460       14.65230
C          8.17270       12.63790       15.09760
C          6.89620       12.73490       14.54510
C          6.70870       13.60790       13.60530
C          7.64870       14.49130       13.13020
C         10.37010       15.40990       16.78060
C          9.54880       14.66920       17.60570
C          9.09320       15.14210       18.85970
C          9.50940       16.46840       19.19480
C         10.27870       17.16890       18.36820
C         10.74610       16.61890       17.19460
C          9.13520       20.16460       14.74310
C          8.52000       21.01170       15.65160
C          9.01520       21.07310       17.01290
C         10.15180       20.38720       17.27360
C         10.71890       19.55430       16.27570
C         11.92020       18.69270       16.58850
C         13.08320       18.97850       15.66950
C         14.24980       19.47020       16.20720
C         15.34150       19.74710       15.33290
C         15.11090       19.63120       13.94930
C         13.88450       19.09750       13.49360
H         12.52155       14.33374       11.25681
H         13.93386       12.39653       11.33368
H         13.77678       10.95054       13.27549
H         12.26995       11.48408       15.11349
H         10.44575       12.96283       16.06334
H          8.40043       11.91722       15.80030
H          6.12603       12.13158       14.87352
H          5.77093       13.64277       13.17589
H          7.41901       15.15570       12.37467
H          9.25406       13.73017       17.29525
H          8.50396       14.57157       19.48605
H          9.20879       16.88653       20.08913
H         10.53007       18.14038       18.60914
H          8.74698       20.08560       13.79018
H          7.71775       21.59269       15.36201
H          8.52620       21.61188       17.74476
H         10.60529       20.46994       18.19693
H         12.06111       18.87901       17.63269
H         14.33320       19.63529       17.22249
H         16.26572       20.02208       15.70063
H         15.83062       19.93311       13.27411
H         13.73675       18.92303       12.48724
"""

tstart = time.time()
mol = pyscf.gto.Mole()
mol.verbose = 4
mol.atom = xyz_geom
mol.basis = "def2-svp"
mol.symmetry = False
mol.max_memory = 85000
mol.spin = 0
mol.charge = 1
mol.build()

mf = pyscf.scf.RHF(mol).density_fit('def2-universal-jkfit').x2c()
mf.conv_tol = 1e-8
mf.max_cycle = 100
ehf = mf.scf()
mf.analyze()
trhf = time.time()

# Get mp2 natural orbitals
pt = pyscf.mp.MP2(mf).run()
noon, uno = pyscf.mcscf.addons.make_natural_orbitals(pt)
np.set_printoptions(linewidth=150, edgeitems=10, threshold=np.inf)
print(noon)
print(noon[120:180])

tmp2 = time.time()
print("Time taken for rhf:", (trhf-tstart)/60)
print("Time taken for mp2:", (tmp2-tstart)/60)

# Change the spin and charge of the molecule:
mol = pyscf.gto.Mole()
mol.verbose = 4
mol.atom = xyz_geom
mol.basis = "def2-svp"
mol.symmetry = False
mol.spin = 1
mol.charge = 2
mol.max_memory = 85000
mol.build(False,False)

# Run RHF computation
mf = pyscf.scf.UHF(mol).density_fit('def2-universal-jkfit').x2c()
mf.conv_tol = 1e-7
mf.max_cycle = 50
ehf = mf.scf()
mf.analyze()

# Run SA-CASSCF
n_states = 2
ncas = 15
nelecas = 13
weights = np.ones(n_states)/n_states

mc = pyscf.mcscf.CASSCF(mf, ncas, nelecas).state_average_(weights).density_fit('def2-universal-jkfit')
mc.conv_tol = 1e-8
mc.conv_tol_grad = 1e-4
mc.max_cycle_macro = 100
mc.kernel(uno)
mc.analyze()
casorbs = mc.mo_coeff.copy()
molden.from_mo(mol, 'casscf_15o_def2tzvp_sa.molden', casorbs)

interface = prism.interface.PYSCF(mf, mc).density_fit('def2-svp-ri')
nevpt = prism.nevpt.NEVPT(interface)
nevpt.method = "nevpt2"
nevpt.method_type = "qd"
nevpt.soc = "BP"
nevpt.nfrozen = 47
nevpt.gtensor = True
nevpt.gtensor_target_state = [1, 2]
e_tot, e_corr, osc = nevpt.kernel()

# Compute all 1-RDMs
rdms = nevpt.make_rdm1().real

for state in range(1, rdms.shape[0]):
    rdm1_diff = rdms[state, state] - rdms[0,0]
    rdm1_diff = mc.mo_coeff @ rdm1_diff @ mc.mo_coeff.T
    cubegen.density(mol, 'state_%s.cube' % str(state), rdm1_diff, nx=300, ny=300, nz=300)

