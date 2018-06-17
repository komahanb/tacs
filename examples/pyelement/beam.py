from mpi4py import MPI
from tacs import TACS, elements
import numpy as np

class Beam(elements.pyElement):
    """
    Implements a beam element with constant properties across the
    length
    """
    def __init__(self, num_disp, num_nodes):
        super(Beam, self).__init__(num_disp, num_nodes)

        self.A = 0.001 # m^2
        self.rho = 2700 # kg/m^3
        self.E = 70.0e9 # N/m^2
        self.G = 26.0e9 # N/m

        self.Iy = 8.33333333333333e-9 # m^4
        self.Iz = 8.33333333333333e-7 # m^4
        self.J = 3.12e-8 #m^4

        #print " Why no shearing?".  Will be there if we use Timoshenko's constitutive relations
        print " Why Ip and J for polar moment?"
        
        return 

    def getMassMatrix(self, L, A, Ip ):        
        # Setup MassMatrix
        M = np.zeros([12,12])
        
        M[0,0]  = 140.0
        M[0,6]  = 70.0
        
        M[1,1]  = 156.0
        M[1,5]  = 22*L
        M[1,7]  = 54.0
        M[1,11] = -13*L
        
        M[2,2]  = 156.0
        M[2,4]  = -22*L
        M[2,8]  = 54.0
        M[2,10] = 13*L        

        M[3,3]  = 140.0*Ip/A
        M[3,9]  = 70.0*Ip/A

        M[4,2]  = -22*L
        M[4,4]  = 4*L*L
        M[4,8]  = -13*L
        M[4,10] = -3*L*L

        M[5,1]  = 22*L
        M[5,5]  = 4*L*L
        M[5,7]  = 13*L
        M[5,11] = -3*L*L
         
        M[6,0]  = 70.0
        M[6,6]  = 140.0
        
        M[7,1]  = 54.0
        M[7,5]  = 13.0*L
        M[7,7]  = 156.0
        M[7,11] = -22.0*L
        
        M[8,2]  = 54.0
        M[8,4]  = -13*L
        M[8,8]  = 156.0
        M[8,10] = 22*L      

        M[9,3] = 70.0*Ip/A
        M[9,9] = 140.0*Ip/A

        M[10,2] = 13.0*L
        M[10,4] = -3.0*L*L
        M[10,8] = 22.0*L
        M[10,10] = 4.0*L*L

        M[11,1] = -13.0*L
        M[11,5] = -3.0*L*L
        M[11,7] = -22.0*L
        M[11,11] = 4.0*L*L
        
        return M

    def getStiffnessMatrix(self, L, E, A, G, J, Iy, Iz):        
        K = np.zeros([12,12])

        # Axial u1
        K[0,0]   = E*A/L
        K[0,6]   = -E*A/L

        # Lead lag v1
        K[1,1]   = 12*E*Iy/(L**3)
        K[1,4]   = 6*E*Iy/(L**2)
        K[1,7]   = -12*E*Iy/(L**3)
        K[1,10]  = 6*E*Iy/(L**2)

        # Flap w1
        K[2,2]   = 12*E*Iz/(L**3)
        K[2,5]   = 6*E*Iz/(L**2)
        K[2,8]   = -12*E*Iz/(L**3)
        K[2,11]  = 6*E*Iz/(L**2)

        # Torsion phi1
        K[3,3]   = G*J/L
        K[3,9]   = -G*J/L

        # Lead lag Bending psi1
        K[4,1]   = 6*E*Iy/(L**2)
        K[4,4]   = 4*E*Iy/L
        K[4,7]   = -6*E*Iy/(L**2)
        K[4,10]  = 2*E*Iy/L

        # Flap Bending theta1
        K[5,2]   = 6*E*Iz/(L**2)
        K[5,5]   = 4*E*Iz/L
        K[5,8]   = -6*E*Iz/(L**2)
        K[5,11]  = 2*E*Iz/L

        # Axial u2
        K[6,0]   = -E*A/L
        K[6,6]   = E*A/L

        # Lead Lag v2
        K[7,1]   = -12*E*Iy/(L**3)
        K[7,4]   = -6*E*Iy/(L**2)
        K[7,7]   = 12*E*Iy/(L**3)
        K[7,10]  = -6*E*Iy/(L**2)

        # Flap w2
        K[8,2]   = -12*E*Iz/(L**3)
        K[8,5]   = -6*E*Iz/(L**2)
        K[8,8]   = 12*E*Iz/(L**3)
        K[8,11]  = -6*E*Iz/(L**2)

        # Torsion phi2
        K[9,3]   = -G*J/L
        K[9,9]   = G*J/L

        # Lead Lag Bending Moment psi2
        K[10,1]  = 6*E*Iy/(L**2)
        K[10,4]  = 2*E*Iy/L
        K[10,7]  = -6*E*Iy/(L**2)
        K[10,10] = 4*E*Iy/L

        # Flap Bending Moment theta2
        K[11,2]  = 6*E*Iz/(L**2)
        K[11,5]  = 2*E*Iz/L
        K[11,8]  = -6*E*Iz/(L**2)
        K[11,11] = 4*E*Iz/L  

        return K
    
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
length = 2.0
dx     = length/nelems

num_disps = 6
num_nodes = 2
beam       = Beam(num_nodes, num_disps)

# Verify the symmetry of stiffness matrix
K = beam.getStiffnessMatrix(dx,
                            beam.E, beam.A,
                            beam.G, beam.J,
                            beam.Iy, beam.Iz)
print np.asmatrix(K) - np.asmatrix(K).transpose()

# Verify the symmetry of mass matrix
M = beam.getMassMatrix(dx, beam.A, beam.J)
print np.asmatrix(M) - np.asmatrix(M).transpose()
stop

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
bdf.writeRawSolution('beam.dat', 0)

# Get the steady state values
t, q, qdot, qddot = bdf.getStates(bdf.getNumTimeSteps())

# Compute the natural frequencies
num_freqs = 10
freq = bdf.lapackNaturalFrequencies(q, qdot, qddot, write_modes=0, use_gyroscopic=0)
freq = np.sort(freq[freq != 0])[0:num_freqs]

print "frequencies", freq
