from mpi4py import MPI
from tacs import TACS, elements
import numpy as np

class Bar(elements.pyElement):
    """
    Implements a bar element with constant properties across the
    length
    """
    def __init__(self, num_disp, num_nodes):
        super(Bar, self).__init__(num_disp, num_nodes)
        self.E   = 70.0e6
        self.rho = 2700.0
        self.k   = np.asmatrix(np.array([[1,-1], [-1,1]]))
        self.m   = np.asmatrix(np.array([[2, 1], [1, 2]]))
        return
    
    def getInitConditions(self, u, udot, uddot, xpts):
        u[0]    = 0.1
        udot[0] = 0.0
        return

    def addResidual(self, time, res, xpts, u, udot, uddot):
        l = xpts[3] - xpts[0]
        mscale = self.rho*l/6.0
        kscale = self.E/l

        # make matrices for easy multiplication
        q = np.asmatrix(u).transpose()
        qdot = np.asmatrix(udot).transpose()
        qddot = np.asmatrix(uddot).transpose()

        # Compute residual
        r = np.matmul(kscale*self.k, q) + np.matmul(mscale*self.m, qddot)

        # Add the residual 
        res += r.A1
        
        return    

    def addJacobian(self, time, J, alpha, beta, gamma, xpts, u, udot, uddot):
        l = xpts[3] - xpts[0]
        mscale = self.rho*l/6.0
        kscale = self.E/l
        J += alpha*kscale*self.k + gamma*mscale*self.m
        return

#######################################################################
# Create an Element
#######################################################################

nelems = 100
length = 1.0
dx     = length/nelems

num_disps = 1
num_nodes = 2
bar       = Bar(num_nodes, num_disps)

#######################################################################
# Create TACS using the elements
#######################################################################

elems = []
for i in xrange(nelems):
    elems.append(bar)    

xpts = []
for i in xrange(nelems+1):
    x = [dx*i, 0.0, 0.0]
    xpts.extend(x)

ptr = [0]
for i in xrange(nelems):
    ptr.extend([max(ptr)+num_nodes])

conn = []
for i in xrange(nelems):
    conn.extend([i+0, i+1])
    
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

print "frequencies", freq
