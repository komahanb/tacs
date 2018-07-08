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
print '      axial nodal location : x'

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
theta = sym.Symbol('theta')
psi   = sym.Symbol('psi')

print '      axial deformation                      : u'
print '      leadlag deformation                    : v'
print '      flap deformation                       : w'
print '      bending about y axis (flap bending)    : theta'
print '      bending about z axis (leadlag bending) : psi'

print '   time derivative of degrees of freedom (generalized coordinates)'
ud     = sym.Symbol('ud')
vd     = sym.Symbol('vd')
wd     = sym.Symbol('wd')
thetad = sym.Symbol('thetad')
psid   = sym.Symbol('psid')

print '      dot of axial deformation                      : ud'
print '      dot of leadlag deformation                    : vd'
print '      dot of flap deformation                       : wd'
print '      dot of bending about y axis (flap bending)    : thetad'
print '      dot of bending about z axis (leadlag bending) : psid'

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
print '      uy <-- v'
print '      uz <-- w'

print '   Take time derivative of deformation field'
print '      uxd <-- ud + z thetad - y psid'
print '      uyd <-- vd'
print '      uzd <-- wd'

# Substitute KE terms with definitions for deformation
T0 = T0.subs(ux, u + z*theta -y*psi)
T0 = T0.subs(uy, v)
T0 = T0.subs(uz, w)
T1 = T1.subs(ux, u + z*theta -y*psi)
T1 = T1.subs(uy, v)
T1 = T1.subs(uz, w)
T2 = T2.subs(ux, u + z*theta -y*psi)
T2 = T2.subs(uy, v)
T2 = T2.subs(uz, w)

# replace time derivatives with dof derivatives
T0 = T0.subs(uxd, ud + z*thetad - y*psid)
T0 = T0.subs(uyd, vd)
T0 = T0.subs(uzd, wd)

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

print "      T0 per unit length :", T0
print "      T1 per unit length :", T1
print "      T2 per unit length :", T2

# Use the shape functions for each DOF using nodal displacements

# Form element matrices

# Repeat exercise for potential energy and matrices
