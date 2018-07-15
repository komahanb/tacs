import sympy as sym

## class Coordinate:
##     """
##     Ge
##     """
##     def __init__(self, symbol, npts
##         self.symbol = rho
##         self.E = E
##         self.nu = nu
##         return

## class Motion:
##     def __init__(self, name, symbol, ndofs):
        
## # Map of aspect of motion and rest
## field = {}
## field{'axial'}   = axial
## field{'flap'}    = flap
## field{'leadlag'} = leadlag
## field{'twist'}   = twist

## axial_ke = axial.ke
## axial_ke = axial.pe

# Geometry
x = sym.Symbol('x')
L = sym.Symbol('L')

npoints = 4

def nodal_dof(identifier, npoints):
    dof = {}
    for n in xrange(npoints):
        key = ('%s%d') % (identifier, n+1)
        dof[n+1] = sym.Symbol(key)
    return dof

def polynomial(x, N):
    '''
    Returns N polynomials each of from 0 to N-1.
    [x^0, x^1, X^2, ... ,  X**(N-1)]
    '''
    p = {}
    for i in xrange(N):
        p[i+1] = x**i
    return p

def field(basis, coordinates):
    num_basis = len(basis)
    num_coordinates = len(coordinates)
    assert(num_basis==num_coordinates)    
    phi = []
    alpha = []
    for i in xrange(num_basis):
        phi.append(basis[i+1])
        alpha.append(coordinates[i+1])

    # Make matrices for easy multiplication
    phi = sym.Matrix(phi[:])
    alpha = sym.Matrix(alpha)

    # Return the dot product of basis and coordinates
    return phi.dot(alpha)
           
# Construct polynomials for required number of points
basis = polynomial(x, npoints)
print "polynomials are poly", basis

# Define nodal dofs for the field variable
coordinates = nodal_dof('u', npoints)
print 'coordinate are ', coordinates

u = field(basis, coordinates)
print 'field of u', u

stop

# Define the number of points
npoints = 2

# Define the basis phi based on number of points
phi = sym.zeros(1,2)
phi[0,0] = 1
phi[0,1] = x

# Physical dofs parameters at each node (displacements, rotations)
ui = sym.Symbol('ui')
uj = sym.Symbol('uj')

## vi     = sym.Symbol('vi')
## wi     = sym.Symbol('wi')
## phii   = sym.Symbol('phii')
## thetai = sym.Symbol('thetai')
## psii   = sym.Symbol('psii')

## vj     = sym.Symbol('vj')
## wj     = sym.Symbol('wj')
## phij   = sym.Symbol('phij')
## thetaj = sym.Symbol('thetaj')
## psij   = sym.Symbol('psij')

# Define the right hand side nodal values
U = sym.zeros(2, 1)
U[0,0] = ui
U[1,0] = uj

# Create the interpolation matrix
PHI = sym.zeros(2,2)
PHI[0,:] = phi.subs(x,0)
PHI[1,:] = phi.subs(x,L)

# Invert the interpolation matrix to find alphas
alpha = PHI.inv()*U

# Create the polynomial as a function of actual dof and x
u =  phi*alpha

# Differentiate with respect to the each rhs and get the shape functions
print 'axial  displacement polynomial', u

# Construct a map between nodal dof and corresponding shape function
nodal_shape = {}
nodal_shape['ui'] = u.diff(ui)[:]
nodal_shape['uj'] = u.diff(uj)[:]
print nodal_shape
stop









N['phii']   = phi.diff(phii)

N['vi']     = v.diff(vi)
N['psii']   = v.diff(psii)

N['wi']     = w.diff(wi)
N['thetai'] = w.diff(thetai)


N['phij']   = phi.diff(phij)

N['vi']     = v.djff(vi)
N['psij']   = v.diff(psij)

N['wj']     = w.diff(wj)
N['thetaj'] = w.diff(thetaj)





E   = sym.Symbol('E')
G   = sym.Symbol('G')
Ip = sym.Symbol('Ip')
Iyy = sym.Symbol('Iyy')
Izz = sym.Symbol('Izz')
J   = sym.Symbol('J')
A   = sym.Symbol('A')
rho = sym.Symbol('rho')



NU    = sym.zeros(1,2)
NU[0] = u.diff(ui)
NU[1] = u.diff(uj)





NX = NU.diff(x)
print "Nx=", NX

print 'disp shape function', NU[0]
print 'disp shape function', NU[1]

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




#['',''] = (n*n).integrate((x,0,L))

for i in xrange(12): #ndofoernode*nnodes
    print ("M[%s,:] = ") % (i) , (M[i,:])[:]

K = (bmat.transpose()*C*bmat).integrate((x, 0, L))
print ''
for i in xrange(12):
    print ("K[%s,:] = ") % (i) , (K[i,:])[:]
