import sympy as sym

# Geometry
x = sym.Symbol('x')
L = sym.Symbol('L')

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

    def integrate(self, x, low, up):
        '''
        Member function to differentiate the values but retain the
        keys (maybe append _x to denote diff??).
        '''
        # New diffeentiable map with same keys, but values
        # differentiated once
        return Map({key:value.integrate((x, low, up)) for (key,value) in self.items()})
    
    def __mul__(self, scalar):
        '''
        Overload multiplication operator for map
        '''
        return Map({key:value*scalar for (key,value) in self.items()})

 
    def __div__(self, scalar):
        '''
        Overload division operator for map
        '''
        return Map({key:value/scalar for (key,value) in self.items()})

    def __add__(self, B):
        '''
        overload addition operator for map
        '''
        C = Map()
        for self_key in self.keys():
            for other_key in B.keys():
                if self_key == other_key:
                    # same keys, so add the 'values' from each map
                    C[self_key] = self[self_key] + B[other_key]
                else:
                    # different keys, so place both the entries in new
                    # map
                    C[self_key] = self[self_key]
                    C[other_key] = B[other_key]
        return C

    def outer(self, B):
        C = Map()
        for self_key in self.keys():
            for other_key in B.keys():
                C[self_key,other_key] = self[self_key]*B[other_key]
        return C
    
class ShapeFunctions(Map):
    """
    Class that extends a differentiable Map and creates a of shape
    functions
    """    
    def __init__(self, dof_key_prefix, hermite, npoints, *args, **kw):
        super(Map, self).__init__(*args, **kw)
        
        self.dof_key_prefix = dof_key_prefix
        self.hermite        = hermite

        if npoints == 2:
            self.xpts = [0, L]
        elif npoints == 3:
            self.xpts = [0, L/2, L]
        elif npoints == 4:
            self.xpts = [0, L/3, 2*L/3, L]
        else:
            raise
        
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
            PHI.append(Phi.subs(x, self.xpts[i])[:])
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
            PHI.append(Phi.subs(x, self.xpts[i])[:])
            PHI.append(Phi.diff(x).subs(x, self.xpts[i])[:])            
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

if __name__ == "__main__":
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
