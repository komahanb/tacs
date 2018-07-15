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
Nxx_v  = N_v.diff(x)
Nxx_w  = N_w.diff(x)
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

scale = -rho*A*omega**2
ku_2  = (N_u.outer(N_u).integrate(x, 0, L))*scale


KU = ku_1 + ku_2
MU = mu_1

print "stiffness matrix :", KU
print "mass matrix      :", MU

stop

######################################################################
# Create torsional motion matrices
######################################################################

######################################################################
# Create chordwise motion matrices
######################################################################

######################################################################
# Create flapping motion matrices
######################################################################

######################################################################
# Assemble element matrix
######################################################################



stop

K = E*A*(NU.diff(x).transpose()*NU.diff(x)).integrate((x,0,L))
print "K=", K

M = rho*A*(NU.transpose()*NU).integrate((x,0,L))
print "M=", M

stop

M = (nmat.transpose()*inertia*nmat).integrate((x, 0, L))
for i in xrange(12): #ndofoernode*nnodes
    print ("M[%s,:] = ") % (i) , (M[i,:])[:]

K = (bmat.transpose()*C*bmat).integrate((x, 0, L))
print ''
for i in xrange(12):
    print ("K[%s,:] = ") % (i) , (K[i,:])[:]
    
# each equation gets a row in shape function
# ncols is as big as the numnodes*ndofpernode
# nui nvi nvpi, nuj nvj npvj, nuk nvk npvk
# 0   1   2      3   4   5     6   7   8 # global dof number

nmat = sym.zeros(6,12)
nmat[0,0]   = NU[0]
nmat[0,6]   = NU[1]
nmat[1,0+1] = NU[0]
nmat[1,6+1] = NU[1]
nmat[2,0+2] = NU[0]
nmat[2,6+2] = NU[1]
nmat[3,0+3] = NU[0]
nmat[3,6+3] = NU[1]
nmat[4,0+4] = NU[0]
nmat[4,6+4] = NU[1]
nmat[5,0+5] = NU[0]
nmat[5,6+5] = NU[1]

bmat = nmat.diff(x)

inertia = sym.zeros(6,6)
inertia[0,0] = rho*A
inertia[1,1] = rho*A
inertia[2,2] = rho*A
inertia[3,3] = rho*Ip
inertia[4,4] = rho*Iyy
inertia[5,5] = rho*Izz

C = sym.zeros(6,6)
C[0,0] = E*A
C[1,1] = 0
C[2,2] = 0
C[3,3] = G*J
C[4,4] = E*Iyy
C[5,5] = E*Izz

M = (nmat.transpose()*inertia*nmat).integrate((x, 0, L))





for i in xrange(12): #ndofoernode*nnodes
    print ("M[%s,:] = ") % (i) , (M[i,:])[:]

K = (bmat.transpose()*C*bmat).integrate((x, 0, L))
print ''
for i in xrange(12):
    print ("K[%s,:] = ") % (i) , (K[i,:])[:]
