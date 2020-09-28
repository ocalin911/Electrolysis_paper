
#This script implements the second order nonlinear ODE (6)-(7):

# x'' = K * 3*x*y / (x^2 + y^2)^(5/2)
# y'' = K * (2 y^2 - x^2) / (x^2 + y^2)^(5/2).

#The idea is to write it as a first order ODE system with 4 equations as: 

#    x'   = v
#    y'   = w       
#    v'   = k*3*x*y / (x*x + y*y )**(5/2)
#    w'   = k*(2*y*y - x*x) / (x*x + y*y )**(5/2)

####################################################

#Import packages
import scipy.integrate
import numpy
import matplotlib.pyplot as plt

#implementing the ODE system
def ODE(var, t, k, k1):
    x, y, v, w = var
    dt    = 1
    dx   = v*dt        
    dy   = w*dt       
    dv   = k*3*x*y*dt / (x*x + y*y )**(5/2)
    dw   = k*(2*y*y - x*x)*dt / (x*x + y*y )**(5/2)
    return([dx, dy, dv, dw])

#initial conditions
x0 = 0.00001
y0 = -0.5
v0 = 0.2
w0 = 0.01
k = 1
k1 =1


#Time vector
T=0.176 #For larger T we get a computational error.
t = numpy.linspace(0, T, 1000)

#Result
solution = scipy.integrate.odeint(ODE, [x0, y0, v0, w0], t, args=(k, k1) )
solution = numpy.array(solution)

#plot results
plt.figure(figsize=[8,5])
plt.plot(t, solution[:,0], label="x(t)")
plt.plot(t, solution[:,1], label="y(t)")

plt.grid()
plt.legend()
plt.xlabel("time")
plt.ylabel("distance")
plt.title("ODE model")
plt.show()

# plot a phase portrait
x = solution[:, 0]
y = solution[:, 1]
plt.figure()
plt.grid()
plt.plot(x,y)
plt.plot(x0, y0, 'ro')
plt.xlabel('x')
plt.ylabel('y')

plt.show()
