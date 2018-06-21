"""
Script for generation of beam elemental matrices using timoshenko
thery two noded beam finite element discretization

Author: Komahan Boopathy
"""

import sympy as sym

def getShapeFunctions():
    # Axial assumed modes shape functions
    N = sym.zeros(6, 12) # nmotions, nvars
    Nprime = sym.zeros(6, 12) # nmotions, nvars

    # Bending assumed modes shape functions
    phi = sym.zeros(1, 4)
    phi[0,0] = 1
    phi[0,1] = (x/L)
    phi[0,2] = (x/L)**2
    phi[0,3] = (x/L)**3

    alpha = sym.zeros(4,1)
    alpha[0] = 1
    alpha[1] = L
    alpha[2] = L**2
    alpha[3] = L**3

    A = sym.zeros(4,4)
    A[0,0] = 1
    A[1,1] = 1
    A[2,0] = 1
    A[2,1] = 1
    A[2,2] = 1
    A[2,3] = 1
    A[3,1] = 1
    A[3,2] = 2
    A[3,3] = 3

    Ntmp = phi*(A.inv())

    # Add the shape functions for axial vibration
    N[0,0] = 1 - x/L
    N[0,0+6] = x/L    
    Nprime[0,:] = N[0,:].diff(x)
    
    # Add flap
    N[1,1] = Ntmp[0]
    N[1,1+6] = Ntmp[2]
    Nprime[1,:] = N[1,:].diff(x).diff(x)

    # Add lead lag
    N[2,2] = N[1,1]
    N[2,2+6] = N[1,1+6]
    Nprime[2,:] = N[2,:].diff(x).diff(x)

    # Add torsion
    N[3,3] = 1 - x/L
    N[3,3+6] = x/L
    Nprime[3,:] = N[3,:].diff(x)
    
    # Add flap bending
    N[4,4] = Ntmp[1]*L
    N[4,4+6] = Ntmp[3]*L
    Nprime[4,:] = N[4,:].diff(x).diff(x)
    
    # Add lead lag bending
    N[5,5] = N[4,4]
    N[5,5+6] = N[4,4+6]
    Nprime[5,:] = N[5,:].diff(x).diff(x)

    return N, Nprime

print '---------------------------------------------------------------'
print 'Generation of element stiffness and matrix for Timoshenko beam'
print '---------------------------------------------------------------'

# Define geometry of beam
x = sym.Symbol('x')
L = sym.Symbol('L')

# Create the shape functions (needs work)
N, Nprime = getShapeFunctions()

print "shape functions", len(N)/6
for i in xrange(len(N)/6):
    print 'col', i, N[:,i][:]

print ''
print "derivative of shape functions", len(Nprime)/6
for i in xrange(len(Nprime)/6):
    print 'col', i, Nprime[:,i][:]

# Define terms for the kinetic energy
rho = sym.Symbol('rho')
A = sym.Symbol('A')
Ip = sym.Symbol('Ip')
Iy = sym.Symbol('Iy')
Iz = sym.Symbol('Iz')
Iyz = sym.Symbol('Iyz')

# Weighing tensor inertia 
inertia = sym.zeros(6, 6)
inertia[0,0] = rho*A
inertia[1,1] = rho*A
inertia[2,2] = rho*A
inertia[3,3] = rho*Ip
inertia[4,4] = rho*Iy
inertia[4,5] = rho*Iyz
inertia[5,5] = rho*Iz
inertia[5,4] = rho*Iyz
print ''
print 'inertia tensor:', inertia.is_symmetric()
for i in xrange(len(inertia)/6):
    print 'col', i, inertia[:,i][:]

# Terms for strain energy
E   = sym.Symbol('E')
G   = sym.Symbol('G')
J   = sym.Symbol('J')
k   = sym.Symbol('k')
Ay  = sym.Symbol('Ay')
Az  = sym.Symbol('Az')

# Weighing consitutive matrix
constitutive = sym.zeros(6, 6)
constitutive[0,0] = E*A
constitutive[1,1] = G*Ay
constitutive[2,2] = G*Az
constitutive[3,3] = G*J
constitutive[4,4] = E*Iy
constitutive[4,5] = E*Iyz
constitutive[5,5] = E*Iz
constitutive[5,4] = E*Iyz

print ''
print 'constitutive tensor:', constitutive.is_symmetric()
for i in xrange(len(constitutive)/6):
    print 'col', i, constitutive[:,i][:]

# Form the mass matrix matrix
mass_matrix = (N.transpose()*inertia*N).integrate((x, 0, L))
print ''
print 'mass matrix', mass_matrix.is_symmetric()
#print mass_matrix.print_nonzero("0")
for i in xrange(len(mass_matrix)/12):
    print ('M[%d,:] = ') % (i), mass_matrix[:,i][:] #/(rho*A*L/420)

# Form the stiffness matrix
stiffness_matrix = (Nprime.transpose()*constitutive*Nprime).integrate((x, 0, L))
print ''
print 'stiffness matrix', stiffness_matrix.is_symmetric()
#print stiffness_matrix.print_nonzero("0")
for i in xrange(len(stiffness_matrix)/12):
    print ('K[%d,:] = ') % (i), stiffness_matrix[:,i][:]

# From position vector form velocity
# From velocity form KE expression
# From strain energy expression form the potential energy
# From Energy expressions get the equations of motion using finite
# element method
