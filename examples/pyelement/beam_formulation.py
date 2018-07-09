import sympy as sym

print '\nDefining the parameters and variables'

print '   Material properties'
rho = sym.Symbol('rho')
E   = sym.Symbol('E')
nu  = sym.Symbol('nu')
print '      density         : rho'
print '      elastic modulus : E'
print '      poisson ratio   : nu'

print '   Cross sectional coordinates of point on the cross sectional plane'
y = sym.Symbol('y')
z = sym.Symbol('z')
print '      spanwise coordinate  : y'
print '      thickness coordinate : z'

print '   Radial distance of the point from hub'
x = sym.Symbol('x')
l = sym.Symbol('l')
print '      axial nodal location : x'
print '      length of element    : l'

print '   Deformation field'
ux = sym.Symbol('ux')
uy = sym.Symbol('uy')
uz = sym.Symbol('uz')
print '      local x deformation : ux'
print '      local y deformation : uy'
print '      local z deformation : uz'

print '   Derivative of the deformation'
uxd = sym.Symbol('uxd')
uyd = sym.Symbol('uyd')
uzd = sym.Symbol('uzd')
print '      local x deformation time rate : uxd'
print '      local y deformation time rate : uyd'
print '      local z deformation time rate : uzd'

print '   Angular velocity (constant)'
omega = sym.Symbol('omega')
print '      rotational speed of hub in rad/s : omega'

print '   degrees of freedom (generalized coordinates)'
u     = sym.Symbol('u')
v     = sym.Symbol('v')
w     = sym.Symbol('w')
phi   = sym.Symbol('phi')
theta = sym.Symbol('theta')
psi   = sym.Symbol('psi')

print '      axial deformation                       : u'
print '      leadlag deformation                     : v'
print '      flap deformation                        : w'
print '      rotation about x axis (torsion)         : phi'
print '      rotation about y axis (flap bending)    : theta'
print '      rotation about z axis (leadlag bending) : psi'

print '   time derivative of degrees of freedom (generalized coordinates)'
ud     = sym.Symbol('ud')
vd     = sym.Symbol('vd')
wd     = sym.Symbol('wd')
phid   = sym.Symbol('phid')
thetad = sym.Symbol('thetad')
psid   = sym.Symbol('psid')

print '      dot of axial deformation                       : ud'
print '      dot of leadlag deformation                     : vd'
print '      dot of flap deformation                        : wd'
print '      dot of rotation about x axis (torsion)         : phid'
print '      dot of rotation about y axis (flap bending)    : thetad'
print '      dot of rotation about z axis (leadlag bending) : psid'

######################################################################
# Velocity terms
######################################################################

print '\nVelocity vector'

print '   deformation velocities'
v0 = sym.zeros(1,3)
v0[0,0] = uxd
v0[0,1] = uyd
v0[0,2] = uzd
print "      ", v0[:]

print '   coupling between cross sectional geometry and rotation'
v1 = sym.zeros(1,3)
v1[0,0] = -omega*y
v1[0,1] =  omega*x
v1[0,2] =  0
print "      ", v1[:]

print '   coupling between cross sectional geometry and deformation'
v2 = sym.zeros(1,3)
v2[0,0] = -omega*uy
v2[0,1] =  omega*ux
v2[0,2] =  0
print "      ", v2[:]

######################################################################
# Kinetic Energy terms
######################################################################

print '\nKinetic Energy (T) per unit volume'

T0 = rho*v0.dot(v0)
T1 = rho*v1.dot(v1)
T2 = rho*v2.dot(v2)

print (("%s\t %50s\t %s\t") % ("      T0 per unit volume", T0, "purely deformation")).expandtabs(5)
print (("%s\t %50s\t %s\t") % ("      T0 per unit volume", T1, "rotation & cross section geometry")).expandtabs(5)
print (("%s\t %50s\t %s\t") % ("      T0 per unit volume", T2, "rotation & deformation")).expandtabs(5)

print '\nKinetic Energy (T) per unit volume in terms of generalized coordinates  (u, v, w, theta, psi)'

print '   Define deformation field'
print '      ux <-- u + z theta - y psi'
print '      uy <-- v - z phi'
print '      uz <-- w + y phi'

print '   Take time derivative of deformation field'
print '      uxd <-- ud + z thetad - y psid'
print '      uyd <-- vd - z phid'
print '      uzd <-- wd + y phid'

# Substitute KE terms with definitions for deformation
T0 = T0.subs(ux, u + z*theta -y*psi)
T0 = T0.subs(uy, v - z*phi)
T0 = T0.subs(uz, w + y*phi)
T1 = T1.subs(ux, u + z*theta -y*psi)
T1 = T1.subs(uy, v - z*phi)
T1 = T1.subs(uz, w + y*phi)
T2 = T2.subs(ux, u + z*theta -y*psi)
T2 = T2.subs(uy, v - z*phi)
T2 = T2.subs(uz, w + y*phi)

# replace time derivatives with dof derivatives
T0 = T0.subs(uxd, ud + z*thetad - y*psid)
T0 = T0.subs(uyd, vd - z*phid)
T0 = T0.subs(uzd, wd + y*phid)
T1 = T1.subs(uxd, ud + z*thetad - y*psid)
T1 = T1.subs(uyd, vd - z*phid)
T1 = T1.subs(uzd, wd + y*phid)
T2 = T2.subs(uxd, ud + z*thetad - y*psid)
T2 = T2.subs(uyd, vd - z*phid)
T2 = T2.subs(uzd, wd + y*phid)

print '   Kinetic energy terms in generalized coordinates'
print "      T0 per unit volume as a function of dofs : ", T0
print "      T1 per unit volume as a function of dofs : ", T1
print "      T2 per unit volume as a function of dofs : ", T2

print '\nFinding KE per unit length'

print '   defining parameters of the cross sectional geometry'
print '      height : h'
print '      width  : b'
print '      area   : A'

h = sym.Symbol('h')
b = sym.Symbol('b')
A = sym.Symbol('A')

print '   integrate along y and z varibles of cross section'

T0 = sym.integrate(T0, (y, -b/2, b/2), (z, -h/2, h/2))
T1 = sym.integrate(T1, (y, -b/2, b/2), (z, -h/2, h/2))
T2 = sym.integrate(T2, (y, -b/2, b/2), (z, -h/2, h/2))

T = (T0 + T1 + T2)/2

print "      T0 per unit length :", T0
print "      T1 per unit length :", T1
print "      T2 per unit length :", T2

######################################################################
# Spatial discretization 
######################################################################

print '\nIntroduce nodal degrees of freedom'
print '      at i-th node : (ui, vi, wi, phii, thetai, psii)'
ui     = sym.Symbol('ui')
vi     = sym.Symbol('vi')
wi     = sym.Symbol('wi')
phii   = sym.Symbol('phii')
thetai = sym.Symbol('thetai')
psii   = sym.Symbol('psii')

print '      at j-th node : (uj, vj, wj, phij, thetaj, psij)'
uj     = sym.Symbol('uj')
vj     = sym.Symbol('vj')
wj     = sym.Symbol('wj')
phij   = sym.Symbol('phij')
thetaj = sym.Symbol('thetaj')
psij   = sym.Symbol('psij')

print '\nIntroduce time derivatives of nodal degrees of freedom'
print '      at i-th node : (udi, vdi, wdi, thetadi, psidi)'
udi     = sym.Symbol('udi')
vdi     = sym.Symbol('vdi')
wdi     = sym.Symbol('wdi')
phidi   = sym.Symbol('phidi')
thetadi = sym.Symbol('thetadi')
psidi   = sym.Symbol('psidi')

print '      at j-th node : (udj, vdj, wdj, thetadj, psidj)'
udj     = sym.Symbol('udj')
vdj     = sym.Symbol('vdj')
wdj     = sym.Symbol('wdj')
phidj   = sym.Symbol('phidj')
thetadj = sym.Symbol('thetadj')
psidj   = sym.Symbol('psidj')

# Element dof vector (Generalized disps)
q = sym.zeros(1,12)
q[0] = ui
q[1] = vi
q[2] = wi
q[3] = phii
q[4] = thetai
q[5] = psii
q[6] = uj
q[7] = vj
q[8] = wj
q[9] = phij
q[10] = thetaj
q[11] = psij

# Time derivative of element dof vector (generalized velocities)
qd = sym.zeros(1,12)
qd[0] = udi
qd[1] = vdi
qd[2] = wdi
qd[3] = phidi
qd[4] = thetadi
qd[5] = psidi
qd[6] = udj
qd[7] = vdj
qd[8] = wdj
qd[9] = phidj
qd[10] = thetadj
qd[11] = psidj

# Use the shape functions for each DOF using nodal displacements
print '\n Defining shape functions'

#---------------------------------------------------------------------#
# Axial motion -- first degree of freedom
#---------------------------------------------------------------------#

Nu = sym.zeros(1,2)
Nu[0] = 1 - x/l
Nu[1] = x/l

qloc = sym.zeros(1,2)
qloc[0] = ui
qloc[1] = uj

qdloc = sym.zeros(1,2)
qdloc[0] = udi
qdloc[1] = udj

# Form interpolants of nodal dof and their time derivatives
ubar  = Nu.dot(qloc)
udbar = Nu.dot(qdloc)

# Substitute these interpolants into the kinetic energy expression
T = T.subs(u, ubar)
T = T.subs(ud, udbar)

print '   dof 1 u :', Nu[:]

#---------------------------------------------------------------------#
# Lead lag motion -- second degree of freedom
#---------------------------------------------------------------------#

Nv = sym.zeros(1,4)
Nv[0] = 1 - 3*x**2/l**2 + 2*x**3/l**3
Nv[1] = x - 2*x**2/l + x**3/l**2
Nv[2] = 3*x**2/l**2 - 2*x**3/l**3
Nv[3] = -x**2/l + x**3/l**2
print '   dof 2 v :', Nv[:]

qloc = sym.zeros(1,4)
qloc[0] = vi
qloc[1] = psii
qloc[2] = vj
qloc[3] = psij

qdloc = sym.zeros(1,4)
qdloc[0] = vdi
qdloc[1] = psidi
qdloc[2] = vdj
qdloc[3] = psidj

# Form interpolants of nodal dof and their time derivatives
vbar  = Nv.dot(qloc)
vdbar = Nv.dot(qdloc)

# Substitute these interpolants into the kinetic energy expression
T = T.subs(v, vbar)
T = T.subs(vd, vdbar)

#---------------------------------------------------------------------#
# Flap motion -- third degree of freedom
#---------------------------------------------------------------------#

Nw = sym.zeros(1,4)
Nw[0] = 1 - 3*x**2/l**2 + 2*x**3/l**3
Nw[1] = -(x - 2*x**2/l + x**3/l**2)
Nw[2] = 3*x**2/l**2 - 2*x**3/l**3
Nw[3] = -(-x**2/l + x**3/l**2)
print '   dof 3 w :', Nw[:]

qloc = sym.zeros(1,4)
qloc[0] = wi
qloc[1] = thetai
qloc[2] = wj
qloc[3] = thetaj

qdloc = sym.zeros(1,4)
qdloc[0] = wdi
qdloc[1] = thetadi
qdloc[2] = wdj
qdloc[3] = thetadj

# Form interpolants of nodal dof and their time derivatives
wbar  = Nw.dot(qloc)
wdbar = Nw.dot(qdloc)

# Substitute these interpolants into the kinetic energy expression
T = T.subs(w, wbar)
T = T.subs(wd, wdbar)

#---------------------------------------------------------------------#
# Torsional motion -- fourth degree of freedom
#---------------------------------------------------------------------#

Nphi = sym.zeros(1,2)
Nphi[0] = 1 - x/l
Nphi[1] = x/l

qloc = sym.zeros(1,2)
qloc[0] = phii
qloc[1] = phij

qdloc = sym.zeros(1,2)
qdloc[0] = phidi
qdloc[1] = phidj

# Form interpolants of nodal dof and their time derivatives
phibar  = Nphi.dot(qloc)
phidbar = Nphi.dot(qdloc)

# Substitute these interpolants into the kinetic energy expression
T = T.subs(phi, phibar)
T = T.subs(phid, phidbar)

print '   dof 4 phi :', Nphi[:]

#######################################################################
# Form the Lagrangian = KE - SE
#######################################################################

L = T #- V 

#######################################################################
# Extract mass matrix from discretized KE
#######################################################################

# Form the mass matrix from its definition
M = sym.zeros(12,12)
for i in xrange(12):
    for j in xrange(12):
        M[i,j] = sym.diff(sym.diff(L, qd[j]), qd[i]).integrate((x, 0, l))
for i in xrange(12):
    print ("      M[%s,:] = ") % (i) , sym.simplify(M[i,:])/(rho*b*h*l/420)
print 'common factor: rho*b*h*l/420\n'

# Form the stiffness matrix from its definition
K = sym.zeros(12,12)
for i in xrange(12):
    for j in xrange(12):
        K[i,j] = sym.diff(sym.diff(L, q[j]), q[i]).integrate((x, 0, l))
for i in xrange(12):
    print ("      K[%s,:] = ") % (i) , sym.simplify(K[i,:][:]) #/(rho*b*h*l*omega**2/420)
