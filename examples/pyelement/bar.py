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
        self.E = 70.0e6
        self.rho = 2700.0
        return
    
    def getInitConditions(self, u, udot, uddot, xpts):
        u[0] = 0.1
        udot[0] = 0.0
        return

    def addResidual(self, time, res, xpts, u, udot, uddot):
        l = xpts[3] - xpts[0]
        mscale = self.rho*l/6.0
        kscale = self.E/l
        res[0] += mscale*(2.0*uddot[0] + 1.0*uddot[1]) + kscale*( 1.0*u[0]-1.0*u[1])
        res[1] += mscale*(1.0*uddot[0] + 2.0*uddot[1]) + kscale*(-1.0*u[0]+1.0*u[1])
        return    

    def addJacobian(self, time, J, alpha, beta, gamma, xpts, u, udot, uddot):
        l = xpts[3] - xpts[0]
        mscale = self.rho*l/6.0
        kscale = self.E/l
        J[0,0] += alpha*kscale*1.0 + gamma*mscale*2.0
        J[0,1] += alpha*kscale*-1.0 + gamma*mscale*1.0
        J[1,0] += alpha*kscale*-1.0 + gamma*mscale*1.0
        J[1,1] += alpha*kscale*1.0 + gamma*mscale*2.0
        return

#######################################################################
# Create an Element
#######################################################################

num_disps = 1
num_nodes = 2
bar       = Bar(num_nodes, num_disps)

#######################################################################
# Create TACS using the element
#######################################################################

elems  = [bar]
xpts   = [0.0, 0.0, 0.0,
          2.0, 0.0, 0.0]
conn   = [0, 1]
ptr    = [0, 2]
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

bdf = TACS.BDFIntegrator(tacs, 0.0, 0.1, 1000, 2)
bdf.integrate()
bdf.writeRawSolution('bar.dat', 0)
