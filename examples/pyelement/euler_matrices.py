import sympy as sym

from fea import ShapeFunctions

# Geometry
x = sym.Symbol('x')
L = sym.Symbol('L')

# Discretization
npoints = 2

# Get the shape functions
N_u   = ShapeFunctions('u'   , False, npoints)
N_phi = ShapeFunctions('phi' , False, npoints)
N_v   = ShapeFunctions('v'   , True, npoints)
N_w   = ShapeFunctions('w'   , True, npoints)

print 'shape functions'
print N_u
print N_phi
print N_v
print N_w

print ''
print 'derivative of shape functions'
Nx_u   = N_u.diff(x)
Nx_phi = N_phi.diff(x)
Nxx_v  = N_v.diff(x).diff(x)
Nxx_w  = N_w.diff(x).diff(x)
print Nx_u
print Nx_phi
print Nxx_v
print Nxx_w

# Define inertial and constitutive parameters
E   = sym.Symbol('E')
G   = sym.Symbol('G')
Ip  = sym.Symbol('Ip')
Iyy = sym.Symbol('Iyy')
Izz = sym.Symbol('Izz')
J   = sym.Symbol('J')
A   = sym.Symbol('A')
rho = sym.Symbol('rho')

# Dynamics parameters
omega = sym.Symbol('omega')

######################################################################
# Create Axial motion matrices
######################################################################

print ''
print 'creating matrices for axial motion'

scale = rho*A
mu_1  = (N_u.outer(N_u).integrate(x, 0, L))*scale

scale = -E*A
ku_1   = (Nx_u.outer(Nx_u).integrate(x, 0, L))*scale

## scale = -rho*A*omega**2
## ku_2  = (N_u.outer(N_u).integrate(x, 0, L))*scale

KU = ku_1 # + ku_2
MU = mu_1

scale = rho*A*omega**2
FU = (N_u*x).integrate(x,0,L)*scale

print "axial - stiffness matrix :", KU
print "axial - mass matrix      :", MU
print "axial - force vector     :", FU

dofs = ['u1', 'u2']

kmat = KU.matrix(dofs)
print ''
for i in xrange(len(dofs)):
    print ("K[%s,:] = ") % (i) , (kmat[i,:])[:]

mmat = MU.matrix(dofs)
print ''
for i in xrange(len(dofs)):
    print ("M[%s,:] = ") % (i) , (mmat[i,:])/(rho*A*L/420)

######################################################################
# Create torsional motion matrices
######################################################################

print ''
print 'creating matrices for torsional motion'

scale = rho*Ip
mphi_1 = (N_phi.outer(N_phi).integrate(x, 0, L))*scale

scale = -G*J
kphi_1 = (Nx_phi.outer(Nx_phi).integrate(x, 0, L))*scale

## scale = -rho*Iyy*omega**2
## kphi_2 = (N_phi.outer(N_phi).integrate(x, 0, L))*scale

KPHI = kphi_1 #+ kphi_2
MPHI = mphi_1

print "torsional - stiffness matrix :", KPHI
print "torsional - mass matrix      :", MPHI

dofs = ['phi1', 'phi2']

kmat = KPHI.matrix(dofs)
print ''
for i in xrange(len(dofs)):
    print ("K[%s,:] = ") % (i) , (kmat[i,:])[:]

mmat = MPHI.matrix(dofs)
print ''
for i in xrange(len(dofs)):
    print ("M[%s,:] = ") % (i) , (mmat[i,:])/(rho*A*L/420)

######################################################################
# Create chordwise motion matrices
######################################################################

print ''
print 'creating matrices for chordwise motion'

scale = rho*A
mv_1  = (N_v.outer(N_v).integrate(x, 0, L))*scale

scale = E*Izz
kv_1  = (Nxx_v.outer(Nxx_v).integrate(x, 0, L))*scale

KV = kv_1 #+ kv_2
MV = mv_1

print "chordwise - stiffness matrix :", KV
print "chordwise - mass matrix      :", MV

dofs = ['v1', 'v1_p', 'v2', 'v2_p']

kmat = KV.matrix(dofs)
print ''
for i in xrange(len(dofs)):
    print ("K[%s,:] = ") % (i) , (kmat[i,:])[:]

mmat = MV.matrix(dofs)
print ''
for i in xrange(len(dofs)):
    print ("M[%s,:] = ") % (i) , (mmat[i,:])/(rho*A*L/420)

######################################################################
# Create flapping motion matrices
######################################################################

print ''
print 'creating matrices for flapwise motion'

scale = rho*A
mw_1  = (N_w.outer(N_w).integrate(x, 0, L))*scale

scale = E*Iyy
kw_1  = (Nxx_w.outer(Nxx_w).integrate(x, 0, L))*scale

KW = kw_1 #+ kw_2
MW = mw_1

print "flapwise - stiffness matrix :", KW
print "flapwise - mass matrix      :", MW

dofs = ['w1', 'w1_p', 'w2', 'w2_p']

kmat = KW.matrix(dofs)
print ''
for i in xrange(len(dofs)):
    print ("K[%s,:] = ") % (i) , (kmat[i,:])[:]

mmat = MW.matrix(dofs)
print ''
for i in xrange(len(dofs)):
    print ("M[%s,:] = ") % (i) , (mmat[i,:])/(rho*A*L/420)

######################################################################
# Assemble element matrix
######################################################################

print ''
print 'element dof vector'
# Global dof vector
q = ['u1', 'v1', 'w1', 'phi1', 'v1_p', 'w1_p',
     'u2', 'v2', 'w2', 'phi2', 'v2_p', 'w2_p']
print q

# Global K matrix
print ''
print 'element stiffness matrix'
kmap = ku_1.union(kv_1.union(kw_1.union(kphi_1)))
K = kmap.matrix(q)
for i in xrange(len(q)):
    print ("K[%s,:] = ") % (i) , (K[i,:])[:]

print ''
print 'element mass matrix'
# Global M matrix
mmap = mu_1.union(mv_1.union(mw_1.union(mphi_1)))
M = mmap.matrix(q)
for i in xrange(len(q)):
    print ("M[%s,:] = ") % (i) , (M[i,:])/(rho*L/420)
