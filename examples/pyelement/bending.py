from mpi4py import MPI
from tacs import TACS, elements
import numpy as np

class EBBeamBending(elements.pyElement):
    """
    Implements a beam in bending element with constant properties
    across the length
    """
    def __init__(self, num_disp, num_nodes):
        super(EBBeamBending, self).__init__(num_disp, num_nodes)

        self.b   = 0.5           # m
        self.h   = 0.5           # m
        self.A   = self.b*self.h # m^2
        self.rho = 2700.0        # kg/m^3
        self.E   = 70.0e9        # N/m^2
        self.I  = 1.0/192.0     # m^4

        return 

    def getMassMatrix(self, l, rho, A ):        
        # Setup MassMatrix
        M = np.zeros([4,4])

        alpha = rho*A*l/420.0
        
        M[0,0] = 156
        M[0,1] = 22*l
        M[0,2] = 54
        M[0,3] = -13*l

        M[1,0] = 22*l
        M[1,1] = 4*l*l
        M[1,2] = 13*l
        M[1,3] = -3*l*l

        M[2,0] = 54
        M[2,1] = 13*l
        M[2,2] = 156
        M[2,3] = -22*l

        M[3,0] = -13*l
        M[3,1] = -3*l*l
        M[3,2] = -22*l
        M[3,3] = 4*l

        return alpha*M

    def getStiffnessMatrix(self, l, E, A, I):        
        K = np.zeros([4,4])

        alpha  = E*I/(l**3)
        
        K[0,0] = 12
        K[0,1] = 6*l
        K[0,2] = -12
        K[0,3] = 6*l

        K[1,0] = 6*l
        K[1,1] = 4*l*l
        K[1,2] = -6*l
        K[1,3] = 2*l*l

        K[2,0] = -12
        K[2,1] = -6*l
        K[2,2] = 12
        K[2,3] = -6*l

        K[3,0] = 6*l
        K[3,1] = 2*l*l
        K[3,2] = -6*l
        K[3,3] = 4*l*l

        return alpha*K
    
    def getInitConditions(self, u, udot, uddot, xpts):
        u[:]    = 0.01
        udot[:] = 0.001
        return

    def addResidual(self, time, res, xpts, u, udot, uddot):
        l = xpts[3] - xpts[0]

        # make matrices for easy multiplication
        q = np.asmatrix(u).transpose()
        qdot = np.asmatrix(udot).transpose()
        qddot = np.asmatrix(uddot).transpose()

        # Compute residual
        K = self.getStiffnessMatrix(l,
                                    self.E, self.A, self.I)        
        M = self.getMassMatrix(l,
                               self.rho,
                               self.A)

        r = np.matmul(K, q) + np.matmul(M, qddot)

        # Add the residual
        res += r.A1
        
        return    

    def addJacobian(self, time, J, alpha, beta, gamma, xpts, u, udot, uddot):
        l = xpts[3] - xpts[0]
        K = self.getStiffnessMatrix(l,
                                    self.E, self.A, self.I)        
        M = self.getMassMatrix(l,
                               self.rho,
                               self.A)
        J += alpha*K + gamma*M
        return

#######################################################################
# Create an Element
#######################################################################

nelems = 500
length = 5.0
dx     = length/nelems

num_disps = 2
num_nodes = 2
beam      = EBBeamBending(num_nodes, num_disps)

# Verify the symmetry of stiffness matrix
K = beam.getStiffnessMatrix(dx,
                            beam.E, beam.A,
                            beam.I)
print np.asmatrix(K) - np.asmatrix(K).transpose()

# Verify the symmetry of mass matrix
M = beam.getMassMatrix(dx, beam.rho, beam.A)
print np.asmatrix(M) - np.asmatrix(M).transpose()

#######################################################################
# Create TACS using the elements
#######################################################################

elems = []
for i in xrange(nelems):
    elems.append(beam)    

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
    
## elems  = [beam, beam, beam, beam]
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
creator.setReorderingType(TACS.PY_NATURAL_ORDER, TACS.PY_DIRECT_SCHUR)
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
bdf.writeRawSolution('beam.dat', 1)

# Get the steady state values
t, q, qdot, qddot = bdf.getStates(bdf.getNumTimeSteps())

# Compute the natural frequencies
num_freqs = 10
freq = bdf.lapackNaturalFrequencies(q, qdot, qddot, write_modes=0, use_gyroscopic=0)
freq = np.sort(freq[freq != 0])[0:num_freqs]
freq = np.sort(freq[freq != 1.0])[0:num_freqs]

print 'Obtained natural frequencies are:'
E   = beam.E
I   = beam.I
rho = beam.rho
A   = beam.A
n   = 0
for omega_tacs in freq:
    n = n + 1
    print ('%12.2f') % (omega_tacs)
