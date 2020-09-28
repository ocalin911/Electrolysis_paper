
#This script plots the probability P(t, x_0) in terms of t
# for values v = 0.1, 0.2, 0.3 and 0.4.  

#Import packages
import scipy.integrate
import scipy.stats
import numpy 
import matplotlib.pyplot as plt


#initializations
n_step = 10000
x = numpy.ones(n_step)
prob = numpy.ones(n_step)

T =1
t  = numpy.linspace(0.001, T, n_step) #Time vector
X = 0.05
v1 = 0.1
v2 = 0.2
v3 = 0.3
v4 = 0.4
sig1 = 0.3
sig2 = 0.1
dt = T/n_step
K = 1
delta = 0.5

#implementing the probability
expectation1= X + v1*t + (0.5*K/delta)*t*t
variance = (sig2**2)*(t**3)/ 3 + (sig1**2)*t
x1 = (delta - expectation1)/numpy.sqrt(variance)
prob1 = 1 - scipy.stats.norm.cdf(x1)

expectation2= X + v2*t + (0.5*K/delta)*t*t
variance = (sig2**2)*(t**3)/ 3 + (sig1**2)*t
x2 = (delta - expectation2)/numpy.sqrt(variance)
prob2 = 1 - scipy.stats.norm.cdf(x2)

expectation3 = X + v3*t + (0.5*K/delta)*t*t
variance = (sig2**2)*(t**3)/ 3 + (sig1**2)*t
x3 = (delta - expectation3)/numpy.sqrt(variance)
prob3 = 1 - scipy.stats.norm.cdf(x3)

expectation4 = X + v4*t + (0.5*K/delta)*t*t
variance = (sig2**2)*(t**3)/ 3 + (sig1**2)*t
x4 = (delta - expectation4)/numpy.sqrt(variance)
prob4 = 1 - scipy.stats.norm.cdf(x4)

#time plot 
plt.figure(figsize=[8,5])
plt.plot(t, prob1, label="v1 = 0.1")
plt.plot(t, prob2, label="v2 = 0.2")
plt.plot(t, prob3, label="v3 = 0.3")
plt.plot(t, prob4, label="v4 = 0.4")

plt.grid()
plt.legend()
plt.xlabel("time t")
plt.ylabel("P(t, x_0)")
plt.title("The probability P(t, x_0)")
plt.show()


