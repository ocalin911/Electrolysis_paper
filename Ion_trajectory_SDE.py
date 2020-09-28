
#This script plots the space components of an ion in its movement from the anode to cathode
#The movement satisfies the stochastic differential equation:
#  dX_t = v_t dt + sig1 dW_t
#  dv_t = K dt + sig2 dB_t
#where W_t and B_t are two 2-dim Brownian motions.

#Import packages
import scipy.integrate
import numpy 
import matplotlib.pyplot as plt

#initializations
n_step = 10000
X1 = numpy.ones(n_step)
v = numpy.ones(n_step)
X2 = numpy.ones(n_step)
w = numpy.ones(n_step)
T=1
t = numpy.linspace(0, T, n_step) #Time vector
X1[0] = 0.25
X2[0] = 0.5
sig1 = 0.3
sig2 = 0.1
dt = T/n_step
K = 1

#implementing the SDE
for i in range(1, n_step):
    z1 = numpy.random.normal()
    z2 = numpy.random.normal()
    z3 = numpy.random.normal()
    z4 = numpy.random.normal()
    X1[i] = X1[i-1]  + v[i-1]*dt + sig1* z1*numpy.sqrt(T/n_step)
    v[i] = v[i-1] + K*dt + sig2* z2*numpy.sqrt(T/n_step)
    X2[i] = X2[i-1]  + w[i-1]*dt + sig1* z3*numpy.sqrt(T/n_step)
    w[i] = w[i-1] + K*dt + sig2* z4*numpy.sqrt(T/n_step)

#time plot 
plt.figure(figsize=[8,5])
plt.plot(t, X1, label="X1(t)")
plt.plot(t, X2, label="X2(t)")

plt.grid()
plt.legend()
plt.xlabel("time")
plt.ylabel("X1 and X2 ")
plt.title("The space components of the trajectory")
plt.show()

# phase portrait plot
plt.figure()
plt.plot(X1,X2)
plt.plot(X1[0], X2[0], 'ro')
plt.xlabel('X1')
plt.ylabel('X2')
plt.title("The  trajectory of the ion")
plt.show()
