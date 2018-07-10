from mpi4py import MPI
from tacs import TACS, elements
import numpy as np
#from assemble import element_stiffness, element_mass

import matplotlib.pyplot as plt

class EulerBeam(elements.pyElement):
    """
    Implements a beam element with constant properties across the
    length
    """
    def __init__(self, num_nodes, num_disp, angular_rate):
        super(EulerBeam, self).__init__(num_nodes, num_disp)
        
        self.speed = angular_rate
        print "omega=", self.speed
        self.num_nodes = num_nodes
        self.ndof      = num_nodes*num_disp

        self.density = 2700.0 # kg/m^3
        self.width  = 0.1
        self.height = 0.1*self.width
        
        self.E   = 70.0e9 # N/m^2
        self.nu  = 0.34615384615384626
        
        self.A   = 0.001 # m^2
        self.G   = 26.0e9 # N/m
        self.J   = 3.12e-8 #m^4
        self.k   = 5.0/6.0
        #self.Ay  = self.k*self.A
        #self.Az  = self.k*self.A
        self.Iy  = 8.33333333333333e-9 # m^4
        self.Iz  = 8.33333333333333e-7 # m^4
        #self.Ip  = self.Iy + self.Iz #m^4
        #self.Iyz = 0.0
        
        return 

    def getTransformationMatrix(self, phi):        
        T = np.zeros([self.ndof, self.ndof])
        
        c = np.cos(phi)
        s = np.sin(phi)

        # block transformation from global to local coordinates
        R = np.zeros([3, 3])
        R[0,0] = c
        R[1,1] = c
        R[2,2] = 1.0
        R[0,1] = s
        R[1,0] = -s

        # place the blocks
        T[0:3 , 0:3]  = R[:,:]
        T[3:6 , 3:6]  = R[:,:]
        T[6:9 , 6:9]  = R[:,:]
        T[9:12, 9:12] = R[:,:]
        
        return np.asmatrix(T)

    def getM(self, l, b, h, rho, omega):        
        M = np.zeros([12,12])
        M[0,:] =  [b*h*l*rho/3, 0, 0, 0, 0, 0, b*h*l*rho/6, 0, 0, 0, 0, 0]
        M[1,:] =  [0, 13*b*h*l*rho/35, 0, 0, 0, 11*b*h*l**2*rho/210, 0, 9*b*h*l*rho/70, 0, 0, 0, -13*b*h*l**2*rho/420]
        M[2,:] =  [0, 0, 13*b*h*l*rho/35, 0, -11*b*h*l**2*rho/210, 0, 0, 0, 9*b*h*l*rho/70, 0, 13*b*h*l**2*rho/420, 0]
        M[3,:] =  [0, 0, 0, l*(-b**3*h*rho - b*h**3*rho)/12 + l*(b**3*h*rho/12 + b*h**3*rho/12) + l*(b**3*h*rho + b*h**3*rho)/36, 0, 0, 0, 0, 0, l*(-b**3*h*rho - b*h**3*rho)/36 + l*(b**3*h*rho + b*h**3*rho)/24, 0, 0]
        M[4,:] =  [0, 0, -11*b*h*l**2*rho/210, 0, b*h*l**3*rho/105, 0, 0, 0, -13*b*h*l**2*rho/420, 0, -b*h*l**3*rho/140, 0]
        M[5,:] =  [0, 11*b*h*l**2*rho/210, 0, 0, 0, b*h*l**3*rho/105, 0, 13*b*h*l**2*rho/420, 0, 0, 0, -b*h*l**3*rho/140]
        M[6,:] =  [b*h*l*rho/6, 0, 0, 0, 0, 0, b*h*l*rho/3, 0, 0, 0, 0, 0]
        M[7,:] =  [0, 9*b*h*l*rho/70, 0, 0, 0, 13*b*h*l**2*rho/420, 0, 13*b*h*l*rho/35, 0, 0, 0, -11*b*h*l**2*rho/210]
        M[8,:] =  [0, 0, 9*b*h*l*rho/70, 0, -13*b*h*l**2*rho/420, 0, 0, 0, 13*b*h*l*rho/35, 0, 11*b*h*l**2*rho/210, 0]
        M[9,:] =  [0, 0, 0, l*(-b**3*h*rho - b*h**3*rho)/36 + l*(b**3*h*rho + b*h**3*rho)/24, 0, 0, 0, 0, 0, l*(b**3*h*rho + b*h**3*rho)/36, 0, 0]
        M[10,:] =  [0, 0, 13*b*h*l**2*rho/420, 0, -b*h*l**3*rho/140, 0, 0, 0, 11*b*h*l**2*rho/210, 0, b*h*l**3*rho/105, 0]
        M[11,:] =  [0, -13*b*h*l**2*rho/420, 0, 0, 0, -b*h*l**3*rho/140, 0, -11*b*h*l**2*rho/210, 0, 0, 0, b*h*l**3*rho/105]
        return M

    def getK(self, l, E, A, G, J, Iy, Iz, k, b, h, rho, omega):

        ## py = 0 #12*E*Iz/(k*G*A*l*l)
        ## pz = 0 #12*E*Iy/(k*G*A*l*l)

        ## scale = E*Iz/((1+py)*l*l)
        ## kz11 = scale*12.0
        ## kz12 = scale*6*l
        ## kz13 = scale*-12.0
        ## kz14 = scale*6*l
        ## kz22 = scale*(4+py)*l*l
        ## kz23 = scale*-6*l
        ## kz24 = scale*(2-py)*l*l
        ## kz33 = scale*12.0
        ## kz34 = scale*-6*l
        ## kz44 = scale*(4+py)*l*l

        ## scale1 = E*Iy/((1+pz)*l*l)
        ## ky11 = scale1*12.0
        ## ky12 = scale1*-6*l
        ## ky13 = scale1*-12.0
        ## ky14 = scale1*-6*l
        ## ky22 = scale1*(4+pz)*l*l
        ## ky23 = scale1*6*l
        ## ky24 = scale1*(2-pz)*l*l
        ## ky33 = scale1*12.0
        ## ky34 = scale1*6*l
        ## ky44 = scale1*(4+pz)*l*l

        ## # row 0
        ## K[0,0] =  E*A/l
        ## K[0,6] = -E*A/l
        ## K[6,0] = K[0,6]
        
        ## # row 1        
        ## K[1,1]  = kz11
        ## K[1,5]  = kz12
        ## K[1,7]  = kz13
        ## K[1,11] = kz14
        ## K[5,1] = K[1,5]
        ## K[7,1] = K[1,7]
        ## K[11,1] = K[1,11]
        
        ## # row 2
        ## K[2,2]  = ky11
        ## K[2,4]  = ky12
        ## K[2,8]  = ky13
        ## K[2,10] = ky14
        ## K[4,2]  = K[2,4]
        ## K[8,2]  = K[2,8]
        ## K[10,2] = K[2,10]

        ## # row 3
        ## K[3,3] = G*J/l
        ## K[3,9] = -G*J/l
        ## K[9,3] = K[3,9]

        ## # row 4
        ## K[4,4] = ky22
        ## K[4,8] = ky23
        ## K[4,10] = ky24
        ## K[8,4] = K[4,8]
        ## K[10,4] = K[4,10]

        ## # row 5
        ## K[5,5] = kz22
        ## K[5,7] = kz23
        ## K[5,11] = kz24
        ## K[7,5] = K[5,7]
        ## K[11,5] = K[5,11]

        ## # row 6
        ## K[6,6] = E*A/l

        ## # row 7
        ## K[7,7] = kz33
        ## K[7,11] = kz34
        ## K[11,7] = K[7,11]

        ## # row 8
        ## K[8,8] = ky33
        ## K[8,10] = ky34
        ## K[10,8] = K[8,10]

        ## # row 9
        ## K[9,9] = G*J/l

        ## # row 10
        ## K[10,10] = ky44

        ## # row 11
        ## K[11,11] = kz44

        L = l
        
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

        # coupled stiffness due to speed
        K1 = np.zeros([12,12])
        K1[0,:] =  [b*h*l*omega**2*rho/3, 0, 0, 0, 0, 0, b*h*l*omega**2*rho/6, 0, 0, 0, 0, 0]
        K1[1,:] =  [0, 13*b*h*l*omega**2*rho/35, 0, 0, 0, 11*b*h*l**2*omega**2*rho/210, 0, 9*b*h*l*omega**2*rho/70, 0, 0, 0, -13*b*h*l**2*omega**2*rho/420]
        K1[2,:] =  [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0]
        K1[3,:] =  [0, 0, 0, b*h**3*l*omega**2*rho/36, 0, 0, 0, 0, 0, b*h**3*l*omega**2*rho/72, 0, 0]
        K1[4,:] =  [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0]
        K1[5,:] =  [0, 11*b*h*l**2*omega**2*rho/210, 0, 0, 0, b*h*l**3*omega**2*rho/105, 0, 13*b*h*l**2*omega**2*rho/420, 0, 0, 0, -b*h*l**3*omega**2*rho/140]
        K1[6,:] =  [b*h*l*omega**2*rho/6, 0, 0, 0, 0, 0, b*h*l*omega**2*rho/3, 0, 0, 0, 0, 0]
        K1[7,:] =  [0, 9*b*h*l*omega**2*rho/70, 0, 0, 0, 13*b*h*l**2*omega**2*rho/420, 0, 13*b*h*l*omega**2*rho/35, 0, 0, 0, -11*b*h*l**2*omega**2*rho/210]
        K1[8,:] =  [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0]
        K1[9,:] =  [0, 0, 0, b*h**3*l*omega**2*rho/72, 0, 0, 0, 0, 0, b*h**3*l*omega**2*rho/36, 0, 0]
        K1[10,:] =  [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0]
        K1[11,:] =  [0, -13*b*h*l**2*omega**2*rho/420, 0, 0, 0, -b*h*l**3*omega**2*rho/140, 0, -11*b*h*l**2*omega**2*rho/210, 0, 0, 0, b*h*l**3*omega**2*rho/105]

        return K + K1
            
    def getInitConditions(self, u, udot, uddot, xpts):
        u[:] = 0.01
        udot[:] = 0.05
        return

    def addResidual(self, time, res, xpts, u, udot, uddot):
        if self.num_nodes == 4:
            l = xpts[9] - xpts[0]
        elif self.num_nodes == 3:
            l = xpts[6] - xpts[0]
        elif self.num_nodes == 2:
            l = xpts[3] - xpts[0]

        # transform from local to global coordinates
        T = self.getTransformationMatrix(self.speed*time)
        
        # make matrices for easy multiplication
        q = np.asmatrix(u).transpose()
        qdot = np.asmatrix(udot).transpose()
        qddot = np.asmatrix(uddot).transpose()

        # Compute matrices
        K = T.transpose()*self.getK(l, self.E, self.A, self.G, self.J, self.Iy, self.Iz, self.k, self.width, self.height, self.density, self.speed)*T
        M = T.transpose()*self.getM(l, self.width, self.height, self.density, self.speed)*T
        r = np.matmul(M, qddot) + np.matmul(K, q)
        
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

        # transformation matrix
        T = self.getTransformationMatrix(self.speed*time)

        # stiffness and mass matrices in global coordinates
        K = T.transpose()*self.getK(l, self.E, self.A, self.G, self.J, self.Iy, self.Iz, self.k, self.width, self.height, self.density, self.speed)*T
        M = T.transpose()*self.getM(l, self.width, self.height, self.density, self.speed)*T

        # add values to the jacobian
        J += alpha*K + gamma*M
        
        return

def frequencies(angular_rate, num_nodes, num_freqs, ref_speed=109.12):
    
    #######################################################################
    # Create an Element
    #######################################################################
    
    nelems = 100
    length = 2.0
    dx = length/nelems

    num_disps = 6
    beam = EulerBeam(num_nodes, num_disps, angular_rate)

    # Test M matrix
    #M = beam.getM(1.0, beam.width, beam.height, beam.density, beam.speed)
    #print "M = ", M
    #print "M symmetry test:", np.asmatrix(M) - np.asmatrix(M).transpose()

    # Test K matrix
    # K1 = beam.getStiffnessMatrix(1.0,
    #                            beam.E, beam.A,
    #                            beam.G, beam.J,
    #                            beam.k*beam.A, beam.k*beam.A,
    #                            beam.Iy, beam.Iz, 0)

    # K = beam.getK(1.0, beam.E, beam.A, beam.G, beam.J, beam.Iy, beam.Iz, beam.k, beam.width, beam.height, beam.density, beam.speed)
    # print "K", K-K1

    #print np.asmatrix(K) - np.asmatrix(K).transpose()
    
    # Test transformation matrix
    # T = beam.getTransformationMatrix(0.1)
    # print T
    # f = np.zeros([12])
    # f[7] = 1
    # u = np.linalg.solve(K[6:,6:], f[6:])
    # print u
    
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
            conn.extend([idx*i+0, idx*i+1, idx*i+2])    
        elif num_nodes == 4:
            idx = num_nodes-1
            conn.extend([idx*i+0, idx*i+1, idx*i+2, idx*i+3])
        
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

    ##################################################################
    # Integrator
    ##################################################################
    
    steps_per_rotation = 360
    num_rotations      = 0.01
    angular_freq       = angular_rate/(2*np.pi)
    if angular_freq != 0.0:
        tfinal         = num_rotations/angular_freq
    else:
        tfinal         = 1.0
    num_steps          = num_rotations*steps_per_rotation
    order              = 1
    bdf                = TACS.BDFIntegrator(tacs, 0.0, tfinal, num_steps, order)

    bdf.setPrintLevel(1)
    bdf.integrate()   
    bdf.writeRawSolution('euler.dat', 1)
    
    # Get the steady state values
    t, q, qdot, qddot = bdf.getStates(bdf.getNumTimeSteps())
    
    # Compute the natural frequencies
    freq = bdf.lapackNaturalFrequencies(q, qdot, qddot, write_modes=0, use_gyroscopic=0)
    freq = np.sort(freq[freq != 0])
    freq = np.sort(freq[freq != 1.0])
    nfreqs = len(freq)
    
    return np.array(freq/ref_speed)[0:num_freqs]
  
if __name__== "__main__":
    angular_rate = 109.12
    num_freqs    = 35
    num_nodes    = 2
    ref_speed    = 109.12
    omega = frequencies(angular_rate, num_nodes, num_freqs, ref_speed)
    print omega
