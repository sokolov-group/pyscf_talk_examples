import numpy as np
from pyscf import gto, scf, adc
from pyscf import lib
from pyscf.tools import cubegen
from pyscf.tools import molden

mol = gto.Mole()
mol.atom = """
C    -1.224333    0.651538    0.000000
C     0.000000    0.000000    0.000000
C     1.224333    0.651538    0.000000
H     0.000000   -1.083700    0.000000
H    -1.276221    1.731792    0.000000
H    -2.153820    0.103153    0.000000
H     1.276221    1.731792    0.000000
H     2.153820    0.103153    0.000000
"""
mol.basis = 'cc-pvtz'
mol.verbose = 5
mol.spin = 1
mol.max_memory = 400000
mol.build()

mf = scf.ROHF(mol).density_fit("cc-pvtz-jkfit")
mf.conv_tol = 1e-8
mf.kernel()

# ADC CALCS
from pyscf.data import elements
myadc = adc.ADC(mf, frozen = [elements.chemcore(mol), elements.chemcore(mol)]).density_fit("cc-pvtz-ri")
myadc.method = "adc(3)"
myadc.conv_tol = 1e-5
myadc.tol_residual = 1e-3
myadc.method_type = "ee"
myadc.compute_spin_square = True
e,v,p,x = myadc.kernel(nroots = 5)

rdm1_a, rdm1_b = myadc.make_rdm1(ao_repr = True)
rdm1_a = np.array(rdm1_a)
rdm1_b = np.array(rdm1_b)

for state in range(rdm1_a.shape[0]):
    spin_den = rdm1_a[state]-rdm1_b[state]
    cubegen.density(mol, 'state_%s.cube' % str(state), spin_den)

