import sympy as sym

x = sym.Symbol('x')
L = sym.Symbol('L')

npts = 3

# Define the basis phi
phi = sym.zeros(1, npts)
phi[0,0] = 1
phi[0,1] = x
if npts >= 3:
    phi[0,2] = x*x
if npts >= 4:
    phi[0,3] = x*x*x

# Physical dofs parameters at each node (displacements, rotations)
ui = sym.Symbol('ui')
uj = sym.Symbol('uj')
uk = sym.Symbol('uk')
ul = sym.Symbol('ul')

# Define the right hand side nodal values
U = sym.zeros(npts, 1)
U[0,0] = ui
U[1,0] = uj
if npts >= 3:
    U[2,0] = uk
if npts >= 4:
    U[3,0] = ul

# Create the inerpolation matrix
PHI = sym.zeros(npts,npts)
if npts == 2:
    PHI[0,0] = phi.subs(x,0)
    PHI[1,0] = phi.subs(x,L)
elif npts == 3:
    PHI[0,0] = phi.subs(x,0)
    PHI[1,0] = phi.subs(x,L/2)
    PHI[2,0] = phi.subs(x,L)
elif npts == 4:
    PHI[0,0] = phi.subs(x,0)
    PHI[1,0] = phi.subs(x,L/3)
    PHI[2,0] = phi.subs(x,2*L/3)
    PHI[3,0] = phi.subs(x,L)

# Invert the interpolation matrix to find alphas 
alpha = PHI.inv()*U

# Create the polynomial as a function of actual dof and x
u =  phi.dot(alpha)

# Differentiate with respect to the each rhs and get the shape functions
print 'polynomial', u
print 'shape function', u.diff(U[0,0])
print 'shape function', u.diff(U[1,0])
if npts >= 3:
    print 'shape function', u.diff(U[2,0])
if npts >= 4:
    print 'shape function', u.diff(U[3,0])

N =  sym.zeros(1,npts)
N[0,0] = u.diff(U[0,0])
N[0,1] = u.diff(U[1,0])
if npts >= 3:
    N[0,2] = u.diff(U[2,0])
if npts >= 4:
    N[0,3] = u.diff(U[3,0])

rho = sym.Symbol('rho')
M = (N.transpose()*rho*N).integrate((x, 0, L))
#print M/(rho*L)
for i in xrange(npts):
    print ("M[%s,:] = ") % (i) , (M[i,:])[:]


Nprime = N.diff(x)
E  = sym.Symbol('E')
K = (Nprime.transpose()*E*Nprime).integrate((x, 0, L))
#print K/(E/L)
print ''
for i in xrange(npts):
    print ("K[%s,:] = ") % (i) , (K[i,:])[:]

# Trained with function values and spatial derivative
