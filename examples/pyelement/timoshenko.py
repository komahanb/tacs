from mpi4py import MPI
from tacs import TACS, elements
import numpy as np
#from assemble import element_stiffness, element_mass

import matplotlib.pyplot as plt

class Beam(elements.pyElement):
    """
    Implements a beam element with constant properties across the
    length
    """
    def __init__(self, num_disp, num_nodes):
        super(Beam, self).__init__(num_disp, num_nodes)

        self.rho = 2700.0 # kg/m^3
        self.E   = 70.0e9 # N/m^2
        self.A   = 0.001 # m^2
        self.G   = 26.0e9 # N/m
        self.J   = 3.12e-8 #m^4
        self.k   = 5.0/6.0
        self.Ay  = self.k*self.A
        self.Az  = self.k*self.A
        self.Iy  = 8.33333333333333e-9 # m^4
        self.Iz  = 8.33333333333333e-7 # m^4
        self.Ip  = self.Iy + self.Iz #m^4
        self.Iyz = 0.0

        return 

    def getMassMatrix(self, L, rho, A, Ip, Iy, Iz, Iyz ):
        M = np.zeros([12,12])
        M[0,:] =  [A*L*rho/3, 0, 0, 0, 0, 0, A*L*rho/6, 0, 0, 0, 0, 0]
        M[1,:] =  [0, 13*Iz*L*rho/35, 0, 0, 0, 11*Iz*L**2*rho/210, 0, 9*Iz*L*rho/70, 0, 0, 0, -13*Iz*L**2*rho/420]
        M[2,:] =  [0, 0, 13*Iy*L*rho/35, 0, -11*Iy*L**2*rho/210, 0, 0, 0, 9*Iy*L*rho/70, 0, 13*Iy*L**2*rho/420, 0]
        M[3,:] =  [0, 0, 0, Ip*L*rho/3, 0, 0, 0, 0, 0, Ip*L*rho/6, 0, 0]
        M[4,:] =  [0, 0, -11*Iy*L**2*rho/210, 0, Iy*L**3*rho/105, 0, 0, 0, -13*Iy*L**2*rho/420, 0, -Iy*L**3*rho/140, 0]
        M[5,:] =  [0, 11*Iz*L**2*rho/210, 0, 0, 0, Iz*L**3*rho/105, 0, 13*Iz*L**2*rho/420, 0, 0, 0, -Iz*L**3*rho/140]
        M[6,:] =  [A*L*rho/6, 0, 0, 0, 0, 0, A*L*rho/3, 0, 0, 0, 0, 0]
        M[7,:] =  [0, 9*Iz*L*rho/70, 0, 0, 0, 13*Iz*L**2*rho/420, 0, 13*Iz*L*rho/35, 0, 0, 0, -11*Iz*L**2*rho/210]
        M[8,:] =  [0, 0, 9*Iy*L*rho/70, 0, -13*Iy*L**2*rho/420, 0, 0, 0, 13*Iy*L*rho/35, 0, 11*Iy*L**2*rho/210, 0]
        M[9,:] =  [0, 0, 0, Ip*L*rho/6, 0, 0, 0, 0, 0, Ip*L*rho/3, 0, 0]
        M[10,:] =  [0, 0, 13*Iy*L**2*rho/420, 0, -Iy*L**3*rho/140, 0, 0, 0, 11*Iy*L**2*rho/210, 0, Iy*L**3*rho/105, 0]
        M[11,:] =  [0, -13*Iz*L**2*rho/420, 0, 0, 0, -Iz*L**3*rho/140, 0, -11*Iz*L**2*rho/210, 0, 0, 0, Iz*L**3*rho/105]
        return M
    
    def getStiffnessMatrix(self, L, E, A, G, J, Ay, Az, Iy, Iz, Iyz):
        K = np.zeros([12,12])
        K[0,:] =  [A*E/L, 0, 0, 0, 0, 0, -A*E/L, 0, 0, 0, 0, 0]
        K[1,:] =  [0, 12*E*Iz/L**3, 0, 0, 0, 6*E*Iz/L**2, 0, -12*E*Iz/L**3, 0, 0, 0, 6*E*Iz/L**2]
        K[2,:] =  [0, 0, 12*E*Iy/L**3, 0, -6*E*Iy/L**2, 0, 0, 0, -12*E*Iy/L**3, 0, -6*E*Iy/L**2, 0]
        K[3,:] =  [0, 0, 0, G*J/L, 0, 0, 0, 0, 0, -G*J/L, 0, 0]
        K[4,:] =  [0, 0, -6*E*Iy/L**2, 0, 4*E*Iy/L, 0, 0, 0, 6*E*Iy/L**2, 0, 2*E*Iy/L, 0]
        K[5,:] =  [0, 6*E*Iz/L**2, 0, 0, 0, 4*E*Iz/L, 0, -6*E*Iz/L**2, 0, 0, 0, 2*E*Iz/L]
        K[6,:] =  [-A*E/L, 0, 0, 0, 0, 0, A*E/L, 0, 0, 0, 0, 0]
        K[7,:] =  [0, -12*E*Iz/L**3, 0, 0, 0, -6*E*Iz/L**2, 0, 12*E*Iz/L**3, 0, 0, 0, -6*E*Iz/L**2]
        K[8,:] =  [0, 0, -12*E*Iy/L**3, 0, 6*E*Iy/L**2, 0, 0, 0, 12*E*Iy/L**3, 0, 6*E*Iy/L**2, 0]
        K[9,:] =  [0, 0, 0, -G*J/L, 0, 0, 0, 0, 0, G*J/L, 0, 0]
        K[10,:] =  [0, 0, -6*E*Iy/L**2, 0, 2*E*Iy/L, 0, 0, 0, 6*E*Iy/L**2, 0, 4*E*Iy/L, 0]
        K[11,:] =  [0, 6*E*Iz/L**2, 0, 0, 0, 2*E*Iz/L, 0, -6*E*Iz/L**2, 0, 0, 0, 4*E*Iz/L]
        return K
    
    def getInitConditions(self, u, udot, uddot, xpts):
        u[6] = 0.1
        u[7] = 0.1
        u[8] = 0.1
        u[9] = 0.01
        u[10] = 0.01
        u[11] = 0.01
        return

    def addResidual(self, time, res, xpts, u, udot, uddot):
        l = xpts[3] - xpts[0]

        # make matrices for easy multiplication
        q = np.asmatrix(u).transpose()
        qdot = np.asmatrix(udot).transpose()
        qddot = np.asmatrix(uddot).transpose()

        # Compute residual
        K = self.getStiffnessMatrix(l, self.E, self.A, self.G, self.J, self.Ay, self.Az, self.Iy, self.Iz, self.Iyz)
        M = self.getMassMatrix(l, self.rho, self.A, self.Ip, self.Iy, self.Iz, self.Iyz)

        r = np.matmul(K, q) + np.matmul(M, qddot)

        # Add the residual
        res[:] += r.A1
        return

    def addJacobian(self, time, J, alpha, beta, gamma, xpts, u, udot, uddot):
        l = xpts[3] - xpts[0]
        K = self.getStiffnessMatrix(l, self.E, self.A, self.G, self.J, self.Ay, self.Az, self.Iy, self.Iz, self.Iyz)
        M = self.getMassMatrix(l, self.rho, self.A, self.Ip, self.Iy, self.Iz, self.Iyz)
        J[:,:] += alpha*K + gamma*M
        return

#######################################################################
# Create an Element
#######################################################################

nelems = 200
length = 2.0
dx     = length/nelems

num_disps = 6
num_nodes = 2
beam      = Beam(num_nodes, num_disps)

## # Verify the symmetry of stiffness matrix
K = beam.getStiffnessMatrix(dx, beam.E, beam.A, beam.G, beam.J, beam.Ay, beam.Az, beam.Iy, beam.Iz, beam.Iyz)
print np.asmatrix(K) - np.asmatrix(K).transpose()

## # Verify the symmetry of mass matrix
M = beam.getMassMatrix(dx, beam.rho, beam.A, beam.Ip, beam.Iy, beam.Iz, beam.Iyz)
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

bdf = TACS.BDFIntegrator(tacs, 0.0, 0.01, 100, 1)
bdf.setPrintLevel(1)
bdf.integrate()
bdf.writeRawSolution('euler.dat', 1)

# Get the steady state values
t, q, qdot, qddot = bdf.getStates(bdf.getNumTimeSteps())

# Compute the natural frequencies
num_freqs = 15 + num_disps
freq = bdf.lapackNaturalFrequencies(q, qdot, qddot, write_modes=0, use_gyroscopic=0)
freq = np.sort(freq[freq != 0])[0:num_freqs]
freq = np.sort(freq[freq != 1])[0:num_freqs]
print "frequencies", freq
