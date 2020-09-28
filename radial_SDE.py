
#This script plots the r(t) solution of the SDE 
#The movement satisfies the stochastic differential equation:
#  dr_t = v_t dt + sig1 * dW_t
#  dv_t = (K/r) dt + sig2 * dB_t
#where W_t and B_t are two 2-dim Brownian motions.

#Import packages
import scipy.integrate
import numpy 
import matplotlib.pyplot as plt

#initializations
n_step = 10000
r = numpy.ones(n_step)
v = numpy.ones(n_step)

T=1
t = numpy.linspace(0, T, n_step) #Time vector
r[0] = 1
v[0] = 0.0

sig1 = 0.1
sig2 = 0.05
dt = T/n_step
K = 1

#implementing the SDE
for i in range(1, n_step):
    z1 = numpy.random.normal()
    z2 = numpy.random.normal()
   
    r[i] = r[i-1]  + v[i-1]*dt + sig1* z1*numpy.sqrt(T/n_step)
    v[i] = v[i-1] + K*dt/r[i-1] + sig2* z2*numpy.sqrt(T/n_step)
    
#time plot 
plt.figure(figsize=[8,5])
plt.plot(t, r, label="r(t)")
#plt.plot(t, v, label="v(t)")

plt.grid()
plt.legend()
plt.xlabel("time")
plt.ylabel("r(t)")
plt.title("The space components of the trajectory")
plt.show()


