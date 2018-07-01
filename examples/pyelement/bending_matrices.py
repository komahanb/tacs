import sympy as sym

x = sym.Symbol('x')
L = sym.Symbol('L')

nterms = 4

# Define the basis phi
phi = sym.zeros(1,nterms)
phi[0,0] = 1
phi[0,1] = x
phi[0,2] = x*x
phi[0,3] = x*x*x

# Physical dofs parameters at each node (displacements, rotations)
vi = sym.Symbol('vi')
vpi = sym.Symbol('vpi')
vj = sym.Symbol('vj')
vpj = sym.Symbol('vpj')

# Define the right hand side nodal values
U = sym.zeros(nterms, 1)
U[0,0] = vi
U[1,0] = vpi
U[2,0] = vj
U[3,0] = vpj

# Create the inerpolation matrix
PHI = sym.zeros(nterms,nterms)
PHI[0,:] = phi.subs(x,0)
PHI[1,:] = phi.diff(x).subs(x,0)
PHI[2,:] = phi.subs(x,L)
PHI[3,:] = phi.diff(x).subs(x,L)

# Invert the interpolation matrix to find alphas
alpha = PHI.inv()*U

# Create the polynomial as a function of actual dof and x
v =  phi.dot(alpha)  # = N.dot(U)

# Differentiate with respect to the each rhs and get the shape functions
print 'displacement polynomial', v

N = sym.zeros(1,nterms)
N[0] = v.diff(vi)
N[1] = v.diff(vpi)
N[2] = v.diff(vj)
N[3] = v.diff(vpj)

print 'disp  shape function', N[0]
print 'slope shape function', N[1]
print 'disp  shape function', N[2]
print 'slope shape function', N[3]

A = sym.Symbol('A')
rho = sym.Symbol('rho')
M = (N.transpose()*rho*A*N).integrate((x, 0, L))
for i in xrange(4): #ndofoernode*nnodes
    print ("M[%s,:] = ") % (i) , (M[i,:])[:]

B = N.diff(x).diff(x)
E  = sym.Symbol('E')
I  = sym.Symbol('I')
K = (B.transpose()*E*I*B).integrate((x, 0, L))
print ''
for i in xrange(4):
    print ("K[%s,:] = ") % (i) , (K[i,:])[:]
