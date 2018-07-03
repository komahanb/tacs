from mpi4py import MPI
from tacs import TACS, elements
import numpy as np
import matplotlib.pyplot as plt

angular_rate = 0 # 109.12

def getFreqs(t, q, qdot, qddot):
    # Compute the natural frequencies
    num_freqs = 10
    freq = bdf.lapackNaturalFrequencies(q, qdot, qddot, write_modes=0, use_gyroscopic=0)
    freq = np.sort(freq[freq != 0])[0:num_freqs]
    freq = np.sort(freq[freq != 1.0])[0:num_freqs]
    num_freqs = len(freq)

    print 'Obtained natural frequencies are:'
    
    E   = beam.E
    I   = beam.I
    rho = beam.rho
    A   = beam.A
    n   = 2
    beta = [1.875, 4.694]
    n = 2
    for k in xrange(num_freqs):
        n = n + 1
        beta.append((2*n-1)*np.pi/(2.0))
    n = 0
    for omega_tacs in freq:
        omega_act = np.sqrt(E*I/(rho*A*length**4))*(beta[n])**2
        n = n + 1
        print ('%12.4f %12.4f') % (omega_tacs/109.12, omega_act/109.12)

    return

class EBBeamBending(elements.pyElement):
    """
    Implements a beam in bending element with constant properties
    across the length
    """
    def __init__(self, num_nodes, num_disp):
        super(EBBeamBending, self).__init__(num_nodes, num_disp)
        
        self.E         = 70.0e9 # N/m^2
        self.A         = 0.001 # m^2
        self.I         = 8.33333333333333e-7 # m^4 # flap
        self.rho       = 2700.0 # kg/m^3
        self.speed     = angular_rate #1.12
        self.num_nodes = num_nodes
        self.ndof      = num_nodes*num_disp

        #self.b   = 0.5           # m
        #self.h   = 0.5           # m
        #self.A   = self.b*self.h # m^2
        #self.rho = 2700.0        # kg/m^3
        #self.E   = 70.0e9        # N/m^2
        #self.I  = 1.0/192.0     # m^4

        return 

    def getTransformationMatrix(self, phi):
        
        T = np.zeros([self.ndof,self.ndof])
        
        c = np.cos(phi)
        s = np.sin(phi)
        
        if self.num_nodes == 2:

            T[0,0] = c
            T[1,1] = c
            T[0,1] = -s
            T[1,0] = s
            
            T[2,2] = c
            T[3,3] = c
            T[2,3] = -s
            T[3,2] = s
            
        elif self.num_nodes == 3:

            T[0,0] = c
            T[1,1] = c
            T[0,1] = -s
            T[1,0] = s
            
            T[2,2] = c
            T[3,3] = c
            T[2,3] = -s
            T[3,2] = s
            
            T[4,4] = c
            T[5,5] = c
            T[4,5] = -s
            T[5,4] = s
            
        elif self.num_nodes == 4:

            T[0,0] = c
            T[1,1] = c
            T[0,1] = -s
            T[1,0] = s
            
            T[2,2] = c
            T[3,3] = c
            T[2,3] = -s
            T[3,2] = s
            
            T[4,4] = c
            T[5,5] = c
            T[4,5] = -s
            T[5,4] = s

            T[6,6] = c
            T[7,7] = c
            T[6,7] = -s
            T[7,6] = s
                
        return np.asmatrix(T)
    
    def getMassMatrix(self, L, rho, A ):        

        M = np.zeros([self.ndof, self.ndof])

        if self.num_nodes == 4:
        
            M[0,:] =  [7069*A*L*rho/80080, 263*A*L**2*rho/96096, 2889*A*L*rho/128128, -2763*A*L**2*rho/320320, 108*A*L*rho/5005, -459*A*L**2*rho/160160, 349*A*L*rho/58240, -17*A*L**2*rho/43680]
            M[1,:] =  [263*A*L**2*rho/96096, A*L**3*rho/9240, 27*A*L**2*rho/20020, -261*A*L**3*rho/640640, 27*A*L**2*rho/22880, -9*A*L**3*rho/80080, 17*A*L**2*rho/43680, -3*A*L**3*rho/128128]
            M[2,:] =  [2889*A*L*rho/128128, 27*A*L**2*rho/20020, 19683*A*L*rho/80080, -729*A*L**2*rho/160160, 6561*A*L*rho/91520, -729*A*L**2*rho/320320, 108*A*L*rho/5005, -27*A*L**2*rho/22880]
            M[3,:] =  [-2763*A*L**2*rho/320320, -261*A*L**3*rho/640640, -729*A*L**2*rho/160160, 81*A*L**3*rho/20020, 729*A*L**2*rho/320320, 81*A*L**3*rho/49280, 459*A*L**2*rho/160160, -9*A*L**3*rho/80080]
            M[4,:] =  [108*A*L*rho/5005, 27*A*L**2*rho/22880, 6561*A*L*rho/91520, 729*A*L**2*rho/320320, 19683*A*L*rho/80080, 729*A*L**2*rho/160160, 2889*A*L*rho/128128, -27*A*L**2*rho/20020]
            M[5,:] =  [-459*A*L**2*rho/160160, -9*A*L**3*rho/80080, -729*A*L**2*rho/320320, 81*A*L**3*rho/49280, 729*A*L**2*rho/160160, 81*A*L**3*rho/20020, 2763*A*L**2*rho/320320, -261*A*L**3*rho/640640]
            M[6,:] =  [349*A*L*rho/58240, 17*A*L**2*rho/43680, 108*A*L*rho/5005, 459*A*L**2*rho/160160, 2889*A*L*rho/128128, 2763*A*L**2*rho/320320, 7069*A*L*rho/80080, -263*A*L**2*rho/96096]
            M[7,:] =  [-17*A*L**2*rho/43680, -3*A*L**3*rho/128128, -27*A*L**2*rho/22880, -9*A*L**3*rho/80080, -27*A*L**2*rho/20020, -261*A*L**3*rho/640640, -263*A*L**2*rho/96096, A*L**3*rho/9240]

        elif self.num_nodes == 2:

            M[0,:] =  [13*A*L*rho/35, 11*A*L**2*rho/210, 9*A*L*rho/70, -13*A*L**2*rho/420]
            M[1,:] =  [11*A*L**2*rho/210, A*L**3*rho/105, 13*A*L**2*rho/420, -A*L**3*rho/140]
            M[2,:] =  [9*A*L*rho/70, 13*A*L**2*rho/420, 13*A*L*rho/35, -11*A*L**2*rho/210]
            M[3,:] =  [-13*A*L**2*rho/420, -A*L**3*rho/140, -11*A*L**2*rho/210, A*L**3*rho/105]

        elif self.num_nodes == 3:

            M[0,:] =  [523*A*L*rho/3465, 19*A*L**2*rho/2310, 4*A*L*rho/63, -8*A*L**2*rho/693, 131*A*L*rho/6930, -29*A*L**2*rho/13860]
            M[1,:] =  [19*A*L**2*rho/2310, 2*A*L**3*rho/3465, 2*A*L**2*rho/315, -A*L**3*rho/1155, 29*A*L**2*rho/13860, -A*L**3*rho/4620]
            M[2,:] =  [4*A*L*rho/63, 2*A*L**2*rho/315, 128*A*L*rho/315, 0, 4*A*L*rho/63, -2*A*L**2*rho/315]
            M[3,:] =  [-8*A*L**2*rho/693, -A*L**3*rho/1155, 0, 32*A*L**3*rho/3465, 8*A*L**2*rho/693, -A*L**3*rho/1155]
            M[4,:] =  [131*A*L*rho/6930, 29*A*L**2*rho/13860, 4*A*L*rho/63, 8*A*L**2*rho/693, 523*A*L*rho/3465, -19*A*L**2*rho/2310]
            M[5,:] =  [-29*A*L**2*rho/13860, -A*L**3*rho/4620, -2*A*L**2*rho/315, -A*L**3*rho/1155, -19*A*L**2*rho/2310, 2*A*L**3*rho/3465]
        
        return M

    def getStiffnessMatrix(self, L, E, A, I):        

        K = np.zeros([self.ndof, self.ndof])
        
        if self.num_nodes == 4:
        
            K[0,:] =  [4539*E*I/(7*L**3), 2517*E*I/(28*L**2), -2187*E*I/(16*L**3), 12393*E*I/(56*L**2), -10935*E*I/(28*L**3), 729*E*I/(7*L**2), -13575*E*I/(112*L**3), 165*E*I/(14*L**2)]
            K[1,:] =  [2517*E*I/(28*L**2), 6157*E*I/(385*L), -10935*E*I/(308*L**2), 148959*E*I/(6160*L), -6561*E*I/(154*L**2), 4131*E*I/(385*L), -165*E*I/(14*L**2), 6893*E*I/(6160*L)]
            K[2,:] =  [-2187*E*I/(16*L**3), -10935*E*I/(308*L**2), 177147*E*I/(154*L**3), 6561*E*I/(44*L**2), -767637*E*I/(1232*L**3), 164025*E*I/(616*L**2), -10935*E*I/(28*L**3), 6561*E*I/(154*L**2)]
            K[3,:] =  [12393*E*I/(56*L**2), 148959*E*I/(6160*L), 6561*E*I/(44*L**2), 45198*E*I/(385*L), -164025*E*I/(616*L**2), 490617*E*I/(6160*L), -729*E*I/(7*L**2), 4131*E*I/(385*L)]
            K[4,:] =  [-10935*E*I/(28*L**3), -6561*E*I/(154*L**2), -767637*E*I/(1232*L**3), -164025*E*I/(616*L**2), 177147*E*I/(154*L**3), -6561*E*I/(44*L**2), -2187*E*I/(16*L**3), 10935*E*I/(308*L**2)]
            K[5,:] =  [729*E*I/(7*L**2), 4131*E*I/(385*L), 164025*E*I/(616*L**2), 490617*E*I/(6160*L), -6561*E*I/(44*L**2), 45198*E*I/(385*L), -12393*E*I/(56*L**2), 148959*E*I/(6160*L)]
            K[6,:] =  [-13575*E*I/(112*L**3), -165*E*I/(14*L**2), -10935*E*I/(28*L**3), -729*E*I/(7*L**2), -2187*E*I/(16*L**3), -12393*E*I/(56*L**2), 4539*E*I/(7*L**3), -2517*E*I/(28*L**2)]
            K[7,:] =  [165*E*I/(14*L**2), 6893*E*I/(6160*L), 6561*E*I/(154*L**2), 4131*E*I/(385*L), 10935*E*I/(308*L**2), 148959*E*I/(6160*L), -2517*E*I/(28*L**2), 6157*E*I/(385*L)]
            
        elif self.num_nodes == 2:

            K[0,:] =  [12*E*I/L**3, 6*E*I/L**2, -12*E*I/L**3, 6*E*I/L**2]
            K[1,:] =  [6*E*I/L**2, 4*E*I/L, -6*E*I/L**2, 2*E*I/L]
            K[2,:] =  [-12*E*I/L**3, -6*E*I/L**2, 12*E*I/L**3, -6*E*I/L**2]
            K[3,:] =  [6*E*I/L**2, 2*E*I/L, -6*E*I/L**2, 4*E*I/L]

        elif self.num_nodes == 3:
        
            K[0,:] =  [5092*E*I/(35*L**3), 1138*E*I/(35*L**2), -512*E*I/(5*L**3), 384*E*I/(7*L**2), -1508*E*I/(35*L**3), 242*E*I/(35*L**2)]
            K[1,:] =  [1138*E*I/(35*L**2), 332*E*I/(35*L), -128*E*I/(5*L**2), 64*E*I/(7*L), -242*E*I/(35*L**2), 38*E*I/(35*L)]
            K[2,:] =  [-512*E*I/(5*L**3), -128*E*I/(5*L**2), 1024*E*I/(5*L**3), 0, -512*E*I/(5*L**3), 128*E*I/(5*L**2)]
            K[3,:] =  [384*E*I/(7*L**2), 64*E*I/(7*L), 0, 256*E*I/(7*L), -384*E*I/(7*L**2), 64*E*I/(7*L)]
            K[4,:] =  [-1508*E*I/(35*L**3), -242*E*I/(35*L**2), -512*E*I/(5*L**3), -384*E*I/(7*L**2), 5092*E*I/(35*L**3), -1138*E*I/(35*L**2)]
            K[5,:] =  [242*E*I/(35*L**2), 38*E*I/(35*L), 128*E*I/(5*L**2), 64*E*I/(7*L), -1138*E*I/(35*L**2), 332*E*I/(35*L)]
        
        return K
    
    def getInitConditions(self, u, udot, uddot, xpts):
        u[:]    = 0.01
        udot[:] = 0.001
        return

    def addResidual(self, time, res, xpts, u, udot, uddot):
        if self.num_nodes == 4:
            l = xpts[9] - xpts[0]
        elif self.num_nodes == 3:
            l = xpts[6] - xpts[0]
        elif self.num_nodes == 2:
            l = xpts[3] - xpts[0]
        
        # make matrices for easy multiplication
        q = np.asmatrix(u).transpose()
        qdot = np.asmatrix(udot).transpose()
        qddot = np.asmatrix(uddot).transpose()

        # transform from local to global coordinates
        T = self.getTransformationMatrix(self.speed*time)
        
        # Compute matrices
        K = T.transpose()*self.getStiffnessMatrix(l, self.E, self.A, self.I)*T
        M = T.transpose()*self.getMassMatrix(l, self.rho, self.A)*T
        r = np.matmul(K, q) + np.matmul(M, qddot) - self.speed*self.speed*np.matmul(M, q)
        
        # Add the residual
        res += r.A1
        
        return    

    def addJacobian(self, time, J, alpha, beta, gamma, xpts, u, udot, uddot):
        if self.num_nodes == 4:
            l = xpts[9] - xpts[0]
        elif self.num_nodes == 3:
            l = xpts[6] - xpts[0]
        elif self.num_nodes == 2:
            l = xpts[3] - xpts[0]

        # Transform from local to global coordinates
        T = self.getTransformationMatrix(self.speed*time)

        # Compute the Jacobian
        K = T.transpose()*self.getStiffnessMatrix(l, self.E, self.A, self.I)*T
        M = T.transpose()*self.getMassMatrix(l, self.rho, self.A)*T
        J += alpha*(K - self.speed*self.speed*M) + gamma*M 

        return

#######################################################################
# Create an Element
#######################################################################

nelems    = 10
length    = 2.0
dx        = length/nelems
num_disps = 2
num_nodes = 3
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

steps_per_rotation = 360
num_rotations      = 1
angular_freq       = angular_rate/(2*np.pi)
if angular_freq != 0.0:
    tfinal         = num_rotations/angular_freq
else:
    tfinal         = 1.0
num_steps          = num_rotations*steps_per_rotation
order              = 1
bdf                = TACS.BDFIntegrator(tacs, 0.0, tfinal, num_steps, order)
bdf.integrate()
bdf.writeRawSolution('beam.dat', 1)

# Get the steady state values
#for  tt in xrange(bdf.getNumTimeSteps()):
t, q, qdot, qddot = bdf.getStates(bdf.getNumTimeSteps())
getFreqs(t, q, qdot, qddot)
## qvals = q.getArray()
## for dof in range(num_disps):
##     print dof, qvals[dof::num_disps][:]
##     plt.plot(qvals[dof::num_disps])
##     plt.show()
