from mpi4py import MPI
from tacs import TACS, elements
import numpy as np

class Bar(elements.pyElement):
    """
    Implements a bar element with constant properties across the
    length
    """
    def __init__(self, num_nodes, num_disp):
        super(Bar, self).__init__(num_nodes, num_disp)
        self.E   = 70.0e9
        self.rho = 2700.0
        self.ndof = num_nodes*num_disp
        self.k   = np.asmatrix(np.array([[1,-1], [-1,1]]))
        self.m   = np.asmatrix(np.array([[2, 1], [1, 2]]))
        return
    
    def getInitConditions(self, u, udot, uddot, xpts):
        u[0]    = 0.1
        udot[0] = 0.0
        return

    def addResidual(self, time, res, xpts, u, udot, uddot):
        l = self.getElemLength(xpts) #[3] - xpts[0]

        # make matrices for easy multiplication
        q = np.asmatrix(u).transpose()
        qdot = np.asmatrix(udot).transpose()
        qddot = np.asmatrix(uddot).transpose()

        # Compute residual
        K = self.getStiffnessMatrix(l, self.E)
        M = self.getMassMatrix(l, self.rho)
        r = np.matmul(K, q) + np.matmul(M, qddot)

        # Add the residual 
        res += r.A1
        
        return

    def getElemLength(self, xpts):
        if self.ndof == 2:
            l = xpts[3] - xpts[0]        
        elif self.ndof == 3:
            l = xpts[2*3] - xpts[0]
        elif self.ndof == 4:
            l = xpts[3*3] - xpts[0]
        return l
    
    def getMassMatrix(self,  L, rho):
        M = np.zeros([self.ndof,self.ndof])     
        if self.ndof == 2:
            M[0,:] =  [L*rho/3, L*rho/6]
            M[1,:] =  [L*rho/6, L*rho/3]
        elif self.ndof == 3:
            M[0,:] =  [2*L*rho/15, L*rho/15, -L*rho/30]
            M[1,:] =  [L*rho/15, 8*L*rho/15, L*rho/15]
            M[2,:] =  [-L*rho/30, L*rho/15, 2*L*rho/15]
        elif self.ndof == 4:
            M[0,:] =  [8*L*rho/105, 33*L*rho/560, -3*L*rho/140, 19*L*rho/1680]
            M[1,:] =  [33*L*rho/560, 27*L*rho/70, -27*L*rho/560, -3*L*rho/140]
            M[2,:] =  [-3*L*rho/140, -27*L*rho/560, 27*L*rho/70, 33*L*rho/560]
            M[3,:] =  [19*L*rho/1680, -3*L*rho/140, 33*L*rho/560, 8*L*rho/105]
        return M

    def getStiffnessMatrix(self, L, E):
        K = np.zeros([self.ndof,self.ndof])     
        if self.ndof == 2:
            K[0,:] =  [E/L, -E/L]
            K[1,:] =  [-E/L, E/L]
        elif self.ndof == 3:
            K[0,:] =  [7*E/(3*L), -8*E/(3*L), E/(3*L)]
            K[1,:] =  [-8*E/(3*L), 16*E/(3*L), -8*E/(3*L)]
            K[2,:] =  [E/(3*L), -8*E/(3*L), 7*E/(3*L)]
        elif self.ndof == 4:
            K[0,:] =  [37*E/(10*L), -189*E/(40*L), 27*E/(20*L), -13*E/(40*L)]
            K[1,:] =  [-189*E/(40*L), 54*E/(5*L), -297*E/(40*L), 27*E/(20*L)]
            K[2,:] =  [27*E/(20*L), -297*E/(40*L), 54*E/(5*L), -189*E/(40*L)]
            K[3,:] =  [-13*E/(40*L), 27*E/(20*L), -189*E/(40*L), 37*E/(10*L)]
        return K
    
    def addJacobian(self, time, J, alpha, beta, gamma, xpts, u, udot, uddot):
        l = self.getElemLength(xpts) #[3] - xpts[0]
        K = self.getStiffnessMatrix(l, self.E)
        M = self.getMassMatrix(l, self.rho)
        J += alpha*K + gamma*M
        return

#######################################################################
# Create an Element
#######################################################################

nelems = 10
length = 1.0
dx     = length/nelems

num_disps = 1
num_nodes = 4
bar       = Bar(num_nodes, num_disps)

# Verify the symmetry of stiffness matrix
K = bar.getStiffnessMatrix(dx, bar.E)
print np.asmatrix(K) - np.asmatrix(K).transpose()

# Verify the symmetry of mass matrix
M = bar.getMassMatrix(dx, bar.rho)
print np.asmatrix(M) - np.asmatrix(M).transpose()

#######################################################################
# Create TACS using the elements
#######################################################################

elems = []
for i in xrange(nelems):
    elems.append(bar)    

xpts = []
for i in xrange((num_nodes-1)*nelems+1):
    x = [dx*i/(num_nodes-1), 0.0, 0.0]
    xpts.extend(x)
        
ptr = [0]
for i in xrange(nelems):
    ptr.extend([max(ptr)+num_nodes])

conn = []
for i in xrange(nelems):
    if num_nodes == 2:
        idx = num_nodes-1
        conn.extend([idx*i+0, idx*i+1])
    elif num_nodes == 3:
        idx = num_nodes-1
        print [idx*i+0, idx*i+1, idx*i+2]
        conn.extend([idx*i+0, idx*i+1, idx*i+2])    
    elif num_nodes == 4:
        idx = num_nodes-1
        conn.extend([idx*i+0, idx*i+1, idx*i+2, idx*i+3])

## elems  = [bar, bar, bar, bar]
## xpts   = [0.00 , 0.0, 0.0,
##           0.25 , 0.0, 0.0,
##           0.50 , 0.0, 0.0,
##           0.75 , 0.0, 0.0,
##           1.00 , 0.0, 0.0]
## conn   = [0, 1,
##           1, 2,
##           2, 3,
##           3, 4]
## ptr    = [0, 2, 4, 6, 8]
    
bcs    = [0]
bcptr  = None
bcvars = None
conn   = np.array(conn, dtype=np.intc)
ptr    = np.array(ptr, dtype=np.intc)
xpts   = np.array(xpts)        
bcs    = np.array(bcs, dtype=np.intc)
if bcptr is not None and bcvars is not None:
    bcptr = np.array(bcptr,dtype=np.intc)
    bcvars = np.array(bcvars,dtype=np.intc)

# Figure out lengths
npts   = len(xpts)/3
nelems = len(elems)

# Sanity check of data we have proper inputs
assert(max(conn)+1 == npts)
assert(nelems == ptr.shape[0]-1)

# Create TACS
comm = MPI.COMM_WORLD
vars_per_node = num_disps
creator = TACS.Creator(comm, vars_per_node)
creator.setReorderingType(TACS.PY_AMD_ORDER, TACS.PY_DIRECT_SCHUR)
if comm.Get_rank() == 0:
    ids = np.arange(0, nelems, dtype=np.intc)
    creator.setGlobalConnectivity(npts, ptr, conn, ids)
    creator.setNodes(xpts)
    creator.setBoundaryConditions(bcs, bcptr, bcvars)            
creator.setElements(elems)
tacs = creator.createTACS()

######################################################################
# Integrator
######################################################################

bdf = TACS.BDFIntegrator(tacs, 0.0, 1.0, 100, 2)
bdf.integrate()
bdf.writeRawSolution('bar.dat', 0)

# Get the steady state values
t, q, qdot, qddot = bdf.getStates(bdf.getNumTimeSteps())

# Compute the natural frequencies
num_freqs = 10
freq = bdf.lapackNaturalFrequencies(q, qdot, qddot, write_modes=0, use_gyroscopic=0)
freq = np.sort(freq[freq != 0])[0:num_freqs]
freq = np.sort(freq[freq != 1.0])[0:num_freqs]

print 'Obtained natural frequencies are:'
E   = bar.E
rho = bar.rho
n   = 0
for omega_tacs in freq:
    n = n + 1
    beta = (2*n -1)*np.pi/(2*length)
    print ('%12.2f %12.2f') % (omega_tacs, beta*np.sqrt(E/rho))
