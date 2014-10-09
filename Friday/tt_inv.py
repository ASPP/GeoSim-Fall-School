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

# $Date: 2014-09-25 15:50:17 +0200 (Thu, 25 Sep 2014) $
# $Author: stefan.mauerberger $
# $Revision: 19681 $

'''
Shall become a nice script for Bayesian Traveltime Inversion
'''

def RK(x,y):
    """
    Cauchy kernel 1/(1+(x-y)^2)
    """
    from numpy import exp, subtract, power, add, divide

    return divide(1.0, add(1.0, power(subtract(x, y), 2)))


def delta_c(x):
    """
    Model perturbations
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
    from numpy import asarray, ones
    """
    Constant Model
    """
    x = asarray(x)

    return 8*ones(x.shape)

def traveltime(x, c):
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
    #import matplotlib
    #matplotlib.use('TkAgg')
    #import matplotlib.pyplot as plt     
    import numpy as np
    from scipy.integrate import nquad, quad, quadrature

    # locations
    x = np.linspace(2,15,15)

    ## Actual Traveltimes for c_0 + delta_c
    T = traveltime(x=x, c=lambda x: c_0(x)+delta_c(x))
    T_lin = traveltime(x=x, c=c_0) - traveltime(x=x, c=lambda x: (c_0(x)**2)/delta_c(x))

    ## Traveltimes for c_0
    T0 = traveltime(x=x, c=c_0)

    ## Predictions
    y = np.linspace(0, 15, 85)


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

            
    mu = -np.dot(sigma_UT, np.dot(sigma_TT_inv, (T_lin-T0)) )

    #sigma_inv = 1-np.dot(np.dot(sigma_UT,sigma_TT_inv), sigma_UT.T)
    #sigma = np.linalg.inv(sigma_inv)
    #
    #c_lin = lambda x: c_0(x)**2/(c_0(x)-delta_c(x))

    #c_inv = lambda x: np.interp(x=x, xp=y, fp=(c_0(y)-mu) )

    #T_inv = traveltime(x, c_inv)

    #xx = np.linspace(0,15,100)
    ##plt.plot(xx, (c_0(xx)+delta_c(xx)), label='actual velocity' )
    #plt.plot(xx, c_lin(xx), label='linearized velocity' )
    #plt.plot(y,  c_inv(y), label='prediction (mean)' )

    #plt.plot(xx, c_0(xx), label='prior' )
    ##plt.scatter(x, c_0(x), label='recs')
    ##plt.scatter(0, c_0(0), c='r', label='source')
    #plt.legend()

    #plt.show()
