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

class Map(dict):
    """
    A differentiable map implementation that extends python's
    dictionary class.

    Author: Komahan Boopathy (komahan@gatech.edu)
    """
    def __init__(self, *args, **kw):
        '''
        Constructor
        '''
        super(Map,self).__init__(*args, **kw)
        return
    
    def diff(self, x):
        '''
        Member function to differentiate the values but retain the
        keys (maybe append _x to denote diff??).
        '''
        # New diffeentiable map with same keys, but values
        # differentiated once
        return Map({key:value.diff(x) for (key,value) in self.items()})


class ShapeFunctions(Map):
    """
    Class that extends a differentiable Map and creates a of shape
    functions
    """    
    def __init__(self, dof_key_prefix, hermite, xpts, *args, **kw):
        '''        
        '''
        super(Map, self).__init__(*args, **kw)

        self.dof_key_prefix = dof_key_prefix
        self.hermite = hermite
        self.xpts = xpts
        
        if hermite is False:
            self.create_shape()
        else:
            self.create_hermite_shape()
            
        return

    def create_shape(self):

        npoints = len(self.xpts)
        
        # Create list of dofs as symbols
        alpha = []
        dkeys = []
        for n in xrange(npoints):
            key = ('%s%d') % (self.dof_key_prefix, n+1)
            dkeys.append(key)            
            alpha.append(sym.var(key))
            
        # Create a list of polynomials for corresponding basis
        phi = []
        for i in xrange(npoints):
            phi.append(x**i)

        # matrix for linear algebra
        Phi = sym.Matrix(phi)
        Alpha = sym.Matrix(alpha)
  
        # Evaluate the polynomials at the given basis and construct a
        # basis matrix
        PHI = []
        for i in xrange(npoints):
            PHI.append(Phi.subs(x, xpts[i])[:])
        PHI = sym.Matrix(PHI)
        
        # Invert the vandermonte matrix and dot with the dofs
        Beta = PHI.inv()*Alpha
        func = Phi.dot(Beta)

        for i in xrange(npoints):
            self[dkeys[i]] = func.diff(alpha[i])

        return

    def create_hermite_shape(self):
        npoints = len(self.xpts)
        ndof = npoints*2
        
        # Create list of dofs as symbols
        alpha = []
        dkeys = []
        for n in xrange(npoints):
            # variable
            key = ('%s%d') % (self.dof_key_prefix, n+1)
            dkeys.append(key)
            alpha.append(sym.var(key))
            
            # derivative of variable
            key = ('%s%d_p') % (self.dof_key_prefix, n+1)
            dkeys.append(key)
            alpha.append(sym.var(key))

        # Create a list of polynomials for corresponding basis
        phi = []
        for i in xrange(ndof):
            phi.append(x**i)

        # matrix for linear algebra
        Phi = sym.Matrix(phi)
        Alpha = sym.Matrix(alpha)

        # Evaluate the polynomials and derivative for the given basis
        # at each given points
        PHI = []
        for i in xrange(npoints):
            PHI.append(Phi.subs(x, xpts[i])[:])
            PHI.append(Phi.diff(x).subs(x, xpts[i])[:])            
        PHI = sym.Matrix(PHI)
        
        # Invert the vandermonte matrix and dot with the dofs
        Beta = PHI.inv()*Alpha
        func = Phi.dot(Beta)

        for i in xrange(ndof):
            self[dkeys[i]] = func.diff(alpha[i])

        return
    
def nodal_dof(identifier, npoints):
    dof = Map()
    for n in xrange(npoints):
        key = ('%s%d') % (identifier, n+1)
        dof[n+1] = sym.var(key)
    return dof

def polynomial(x, N):
    '''
    Returns N polynomials each of from 0 to N-1.
    [x^0, x^1, X^2, ... ,  X**(N-1)]
    '''
    p = Map()
    for i in xrange(N):
        p[i+1] = x**i
    return p

def field(basis, coordinates, xpts):
    num_basis = len(basis)
    num_coordinates = len(coordinates)
    assert(num_basis==num_coordinates)
    assert(len(xpts)==num_coordinates)    

    # Make matrices for easy multiplication from map 
    phi = []
    alpha = []
    for i in xrange(num_basis):
        phi.append(basis[i+1])
        alpha.append(coordinates[i+1])
    phi = sym.Matrix(phi)
    alpha = sym.Matrix(alpha)
    
    # Evaluate phi at points supplied and construct vandermonte matrix
    PHI = []
    for i in xrange(num_coordinates):
        PHI.append(phi.subs(x,xpts[i])[:])
    PHI = sym.Matrix(PHI)

    # Invert the interpolation matrix to find coeffsxs
    beta = PHI.inv()*alpha

    # inner product  of coeff and basis forms the function 
    f = phi.dot(beta)

    return f

def shape_functions(field, coordinates):
    N = Map()
    dofs = coordinates.values()
    num_coordinates = len(dofs)
    for dof in dofs:
        key = ('%s') % (dof)
        N[key] = field.diff(dof)
    return N

# Geometry
x = sym.Symbol('x')
L = sym.Symbol('L')

# Discretization
npoints = 2
if npoints == 2:
    xpts = [0, L]
elif npoints == 3:
    xpts = [0, L/2, L]
elif npoints == 4:
    xpts = [0, L/3, 2*L/3, L]

N = ShapeFunctions('u', False, xpts)
H = ShapeFunctions('u', True, xpts)
print 'ordinary shape functions', N
print "hermite shape functions", H
stop

# Define nodal dofs for the field variable
uhat = nodal_dof('u', npoints)
print 'coordinate are :', uhat

# Construct polynomials for required number of points
phiu = polynomial(x, npoints)
print "basis functions are; ", phiu

# Construct the field as a function of dofs    
U = field(phiu, uhat, xpts)
print 'field of u', U

# Derivative of the field with respect to the coordinates give the
# shape functions
Nu = shape_functions(U, uhat)
print 'shape functions are', Nu

# Differentiate the shape functions 
Nu_x = Nu.diff(x)
print 'differentiated shape functions are', Nu_x

#['',''] = (n*n).integrate((x,0,L))
#  
stop

# Hermite shape functions as well


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
nodal_shape = Map()
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





for i in xrange(12): #ndofoernode*nnodes
    print ("M[%s,:] = ") % (i) , (M[i,:])[:]

K = (bmat.transpose()*C*bmat).integrate((x, 0, L))
print ''
for i in xrange(12):
    print ("K[%s,:] = ") % (i) , (K[i,:])[:]
