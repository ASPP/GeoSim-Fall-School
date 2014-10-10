#!/usr/bin/env python
# -*- coding: utf-8 -*-

# (c) b Stefan Mauerberger <mauerber@uni-potsdam.de>

# This program is free software: you can redistribute it and/or modify
# under the terms of the GNU General Public License as published by
# the Free Software Foundation, either version 3 of the License, or
# (at your option) any later version.

# This program is distributed in the hope that it will be useful,
# but WITHOUT ANY WARRANTY; without even the implied warranty of
# MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE. See the
# GNU General Public License for more details.

# You should have received a copy of the GNU General Public License
# along with this program.  If not, see <http://www.gnu.org/licenses/>.

'''
Shall become a nice script for Bayesian travel-time inversion
Everything is Gaussian, everything is good ...
'''

def RK(x,y):
    """
    Cauchy kernel: 1/(1+(x-y)^2)

Args:
    x, y: scalar or array of locations
    """
    from numpy import exp, subtract, power, add, divide

    return divide(1.0, add(1.0, power(subtract(x, y), 2)))


def delta_c(x):
    """
    Varying part of the velocity model (spatial mean is zero)

Args:
    x: scalar or array of locations

Returns:
    a nd-array of the shape of x
    """
    from numpy import zeros, exp, subtract, power, asarray, nditer, multiply

    x = asarray(x)

    mu    = asarray([1,4,6,8,11])
    sigma = asarray([1,2,1,4,1])
    c = zeros(x.shape)-0.7
    for m, s in nditer([mu,sigma]):
        c += exp(multiply(-2.0, power(subtract(x, m)/s, 2)))

    return c

def c_0(x):
    """
    Constant part of the Velocity model. 

Args:
    x: scalar or array of locations

Returns:
    a nd-array of the shape of x
    """
    from numpy import asarray, ones
    x = asarray(x)

    return 8.0*ones(x.shape)

def traveltime(x, c):
    """
    Calculates the travel-time 

Args:
    x: scalar or array of locations 
    c: Velocity model; has to be a callable function returning the velocity at some location
    """
    from scipy.integrate import quadrature
    from numpy import isscalar, asarray, empty

    s = lambda x: 1.0/c(x)

    if isscalar(x):
        T = quadrature(func=s, a=0.0, b=x)[0]
    else:
        x = asarray(x)
        T = empty(x.shape)
        T[:] = [quadrature(func=s, a=0.0, b=b)[0] for b in x]
        
    return T

if __name__ == "__main__":
    import numpy as np
    from scipy.integrate import nquad, quad, quadrature

    ## Locations of travel-time observations
    x = np.linspace(2,15,15)

    ## Real travel-times 
    # c_0 + delta_c is the actual velocity model we want to invert for
    c_real = lambda x: c_0(x)+delta_c(x)
    T = traveltime(x=x, c=c_real)

    ## Linearized travel-times 
    # Linearized velocity
    c_lin = lambda x: c_0(x)**2/(c_0(x)-delta_c(x))
    T_lin = traveltime(x=x, c=c_lin)


    ## Travel-times for c_0
    T0 = traveltime(x=x, c=c_0)

    ## Locations we want to make predictions for
    y = np.linspace(0, 15, 85)


    ## Actual computations 
    # Unfortunately not yet documented 

    sigma_UU = np.zeros( y.shape + y.shape )
    sigma_UU[:,:] = RK( *np.meshgrid(y,y) )


    sigma_UT = np.zeros( y.shape + x.shape )
    for i, j in np.ndindex(sigma_UT.shape):
        c = lambda z: RK(y[i],z)/c_0(z)**2
        sigma_UT[i,j] = -quad(a=0.0, b=x[j], func=c)[0]


    sigma_TT = np.zeros( x.shape + x.shape )
    c = lambda y, z: RK(y, z)/(c_0(y)**2*c_0(z)**2)
    # This is where the runtime goes
    for i, j in np.ndindex(sigma_TT.shape):
        sigma_TT[i,j] = nquad(ranges=[(0.0, x[i]), (0.0, x[j])], func=c)[0]

    sigma_TT_inv = np.linalg.inv(sigma_TT)


    # Predicted mean-value of c posterior to travel-time data 
    mu = -np.dot(sigma_UT, np.dot(sigma_TT_inv, (T_lin-T0)) )

    # There is something wrong in the predicted covariance matrix 
    #sigma_inv = 1.0 - np.dot(np.dot(sigma_UT, sigma_TT_inv), sigma_UT.T)
    #sigma = np.linalg.inv(sigma_inv)


    # Inverted velocity model 
    c_inv = lambda x: np.interp(x=x, xp=y, fp=(c_0(y)-mu) )
    

    def plotting():
	"""
	Just for taking the plotting stuff out of profiling.
        """
        #import matplotlib
        #matplotlib.use('TkAgg')
        import matplotlib.pyplot as plt     

	# Just a 100 points between 0 and 15
    	xx = np.linspace(0,15,100)

        # Plot the actual velocity model
    	plt.plot(xx, (c_0(xx)+delta_c(xx)), label='actual vel.' )
        # Plot linearized velocity model
        plt.plot(xx, c_lin(xx), label='linearized vel.' )

        # Plot inverted velocity model
        plt.plot(y,  c_inv(y), label='predicted mean')
        
        # Prior model 
        plt.plot(xx, c_0(xx), label='Prior vel.' )

        # Points of travel-time observations 
        plt.scatter(x, c_0(x), label='Recs')
        # Source at 0.0
    	plt.scatter(0, c_0(0), c='r', label='source')

    	# Legend
        plt.legend()

    	# Actually show the plot
	plt.show()

