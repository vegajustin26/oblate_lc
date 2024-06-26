import numpy as np
import matplotlib.pyplot as plt
import math
# import time
# import timeit
# import csv
# from scipy.optimize import minimize
# from scipy import integrate
# np.seterr(invalid="ignore")
# import jaxoplanet
# from jaxoplanet.core.limb_dark import light_curve
# import jax
# from jax import grad
# import jax.numpy as jnp

from orbit import on_sky
from ferrari_solve import overlap_area, qrt_coeff, qrt_solve, verify_roots

# jax.config.update(
#     "jax_enable_x64", True)  # For 64-bit precision since JAX defaults to 32-bit

# from jaxoplanet.light_curves import LimbDarkLightCurve
# from jaxoplanet.orbits import TransitOrbit
# import batman
# from juliacall import Main as jl, convert as jlconvert
# jl.include("../scripts/flux_calc.jl")
# from array import array
# export MKL_ENABLE_INSTRUCTIONS=SSE4_2 # ignoring numpy Intel warnings

class TransitParams(object):
    def __init__(self):
        self.t0 = None          # time of inferior conjunction, in days
        self.tref = None        # reference time for transit, in days
        self.per = None         # orbital period, in days
        self.req = None         # equatorial radius of planet
        self.rprstar = None     # rp/rstar
        self.rpol = None        # polar radius of planet
        self.rstar_eq = 1       # equatorial radius of star
        self.rstar_pol = 1      # polar radius of star
        self.reff = None        # effective radius of planet
        self.f = 0              # oblateness
        self.a = None           # semi-major axis, in units of R*
        self.b = None           # impact parameter
        self.ecc = None         # eccentricity, in radians?
        self.w = None           # longitude of periastron (in radians) # not sure if this is labeled correctly
        self.u = [0.0, 0.0]           # limb-darkening coefficients
        self.limb_dark = None   # limb darkening function
        self.res = None         # resolution of model
        self.phi_p = 0          # obliquity angle of planet
        self.phi_s = 0          # obliquity angle of star
        self.pos_star = (0, 0)  # position of star
        self.omega = None       # argument of periapsis (in radians)
        self.counter = 0
        self.counter1 = 0
        self.debuglist = []     # list to populate with debug info
        self.debuglist1 = []    # list to populate with debug info

    def set_f(self, value):
        self.f = value
        self.calculate_rpol()

    def calculate_rpol(self):
        if self.req is not None and self.f != 0:
            self.rpol = self.req * (1 - self.f)
            self.reff = (self.req * self.rpol)**(0.5)
        else:
            self.rpol = None

    def set_b(self, value):
        self.b = value
        self.calculate_inc()

    def calculate_inc(self):
        if self.b is not None and self.a is not None:
            self.inc = math.acos(self.b/self.a) # in radians

## main flux driving function
def flux_ratio(params, s, cy, AA, BB, CC, DD, EE, FF, H2_TR, K2_TR):

    # parameter checks
    if (params.rstar_eq is not None) or (params.rstar_pol is not None) or (params.req is not None) or (params.rpol is not None) or (params.rprstar is not None):
        if(params.rstar_eq<=0.0 or params.rstar_pol<=0.0 or params.req<=0.0 or params.rpol<=0.0):
            raise Exception("Zero or negative rstar_eq, rstar_pol, req, or rpol!")
    
    #rotation angles should be between -2pi and 2pi
    if(abs(params.phi_s)>(math.pi)):
        params.phi_s = params.phi_s%(math.pi)
    if(abs(params.phi_p)>(math.pi)):
        params.phi_p = params.phi_p%(math.pi)
    
    
    # oblate planet
    if (params.u[0] and params.u[1]) == 0.0: # if no limb-darkening
        area, nintpts = oblate_uniform(params, s, cy, AA, BB, CC, DD, EE, FF, H2_TR, K2_TR)
        return(area)
    elif (params.u[0] and params.u[1]) > 0:  # if limb-darkening
        # simple approximation
        limb_dark_area = MA_driver(params, s, cy, AA, BB, CC, DD, EE, FF, H2_TR, K2_TR) # should use another variable for this
        
        return(limb_dark_area)




        # more complicated approximation
        # carter-winn


        
        # area = oblate_uniform(params, s, cy, AA, BB, CC, DD, EE, FF, H2_TR, K2_TR, H2, K2)
        # print(area)
        # rp = (area/(math.pi))**(0.5) # rp-rstar
        # print(s)
        # params.rp = rp
        
        # ax = viz(params, PHI, H2, K2, "both")
        # plt.show()
        # do mandel-agol (quick mode)
        

            # oblate uniform should return thetas too
        # mandel agol circular (both limb-darkening and uniform source)
            # will have to set params.eq/pol to params.rpstar when doing the circular part
        # do quasi-MC sampling
        return(limb_dark_area)

####################
# cases
####################

# oblate uniform, Ferrari's Method
def oblate_uniform(params, s, cy, AA, BB, CC, DD, EE, FF, H2_TR, K2_TR):
    """
    # case 1: planet fully outside of star area
    # case 2: planet very close to outer limb of star, unsure about intersection
    # case 3: planet definitely intersecting star
    # case 4: planet very close to inner limb of star, unsure about intersection
    # case 5: planet definitely transiting star, fully within stellar bounds
    """

    if s > (params.rstar_eq + params.req): # case 1
        nintpts = 0
        xint = None
        yint = None

        OverlapArea, thetas = overlap_area(nintpts, xint, yint, params,H2_TR,K2_TR,AA,BB,CC,DD,EE,FF)

    elif (params.rstar_eq - params.req) < s < (params.rstar_eq + params.req): # case 2, 3, 4 combined
        nychk, ychk = qrt_solve(cy, params)
        nintpts, xint, yint = verify_roots(nychk, ychk, params, AA, BB, CC, DD, EE, FF)
        OverlapArea, thetas = overlap_area(nintpts, xint, yint, params,H2_TR,K2_TR,AA,BB,CC,DD,EE,FF)    

    elif s < (params.rstar_eq - params.req): # case 5
        
        nintpts = 0
        xint = None
        yint = None
        OverlapArea, thetas = overlap_area(nintpts, xint, yint, params,H2_TR,K2_TR,AA,BB,CC,DD,EE,FF)

    else:
        # definite error
        return(0)

    return(OverlapArea, nintpts)

# limb-darkening approximation with jaxoplanet
def MA_driver(params, s, cy, AA, BB, CC, DD, EE, FF, H2_TR, K2_TR):
    """
    # do the cases
    # only call oblate_uniform in cases 2, 3, and 4 and determine nintpts
    # if nintpts == 2 then need to equate areas
    # but also need to accomodate case 2, 3, and 4
    # if nintpts == 0 then either fully in or fully out
    """
    
    Aellipse, nintpts = oblate_uniform(params, s, cy, AA, BB, CC, DD, EE, FF, H2_TR, K2_TR)

    if nintpts == 2:
        
        # do spherical approximation
        s_eff = optim(params.reff, s, Aellipse, params)
        
        approx_flux = light_curve(params.u, s_eff, params.reff) # jaxoplanet

        return(-approx_flux)
        # return(Aellipse)
    elif nintpts == 0: # cases 2, 3, 4, 5 not touching
        if s > 1:
            return(0) # this is correct, should be 0
        elif s < 1:
            approx_flux = light_curve(params.u, s, params.reff)
            return(-approx_flux) # do the same as case 5, it is fully transiting


def optim(p, z0, oblate_area, params):
    """
    finds z_eff or p_eff of a circular planet
    that gives the exact area of the supplied oblate_area
    p is the "effective circular radius" of the oblate planet, used as an initial guess for p_eff optimization
    z0 is the separation distance of the oblate planet, used as an initial guess for s_eff optimization
    oblate_area is the uniform oblate obscured flux
    """
    # constraints = ({'type': 'eq', 'fun': lambda z: (MA_overlap(p, z) - oblate_area)**2})

    z_func = lambda z_eff: MA_overlap(p, z_eff, oblate_area)
    print(z0)
    if 1 + params.reff < z0 or z0 < 1 - params.reff:
        # params.counter += 1
        # result = 0.5
        print("Yaaaa")
        result = minimize(z_func, x0=z0, method = "L-BFGS-B").x[0]
    else:
        dz_func = lambda z_eff: obj_deriv(p, z_eff, oblate_area)

        result = newton_raphson(z_func, dz_func, z0)

    return(result)

def kappa0(p, z):
    return np.arccos((p**2 + z**2 - 1)/(2*p*z))

def kappa1(p, z):
    return np.arccos((1 - p**2 + z**2)/(2*z))

def MA_overlap(p, z, oblate_area):
    """ defines the Mandel-Agol circle-circle overlap function
    corresponds with eq. 1 of Mandel-Agol (2002)
    """
    MA = (1/np.pi) * (p**2 * kappa0(p, z) + kappa1(p, z) - jnp.sqrt((4*z**2 - (1 + z**2 - p**2)**2)/4))
    # print(MA)
    # print(np.square(MA - oblate_area)[0])
    return(np.square(MA - oblate_area))


def obj_deriv(p, z, oblate_area):
    return((2*((z**3*(-1 - p**2 + z**2))/np.sqrt(4*z**2 - (1 - p**2 + z**2)**2) - 
     -      (-1 + p**2 + z**2)/np.sqrt((4*z**2 - (1 - p**2 + z**2)**2)/z**2) + 
     -      (p*(-1 + p**2 - z**2))/np.sqrt((4*z**2 - (1 - p**2 + z**2)**2)/(p**2*z**2)))*
     -    (-oblate_area + (-0.5*np.sqrt(4*z**2 - (1 - p**2 + z**2)**2) + np.arccos((1 - p**2 + z**2)/(2.*z)) + 
     -         p**2*np.arccos((-1 + p**2 + z**2)/(2.*p*z)))/np.pi))/(np.pi*z**2))


def newton_raphson(f, fprime, x0, tol=1e-5):
    
    counter = 0
    diff = 1
    while diff > tol:
        f_val = f(x0)
        fprime_val = fprime(x0)
        x_new = x0 - f_val/fprime_val
        diff = abs(x0-x_new)
        
        x0 = x_new
        counter += 1
            
    # print(f"counter: {counter}")
    # params.debuglist1.append(counter)
    return(x0)


def viz(params, PHI, H2, K2, mode):
    # for visualization/debugging only!!
    star_x = params.rstar_eq* np.cos(PHI) * np.cos(params.phi_s) - params.rstar_pol * np.sin(PHI) * np.sin(params.phi_s) + params.pos_star[0]
    star_y = params.rstar_eq* np.cos(PHI) * np.sin(params.phi_s) + params.rstar_pol * np.sin(PHI) * np.cos(params.phi_s) + params.pos_star[1]
    
    fig,ax = plt.subplots(figsize = (5, 5))
    ax.plot(star_x, star_y, lw=2, label = "star")

    if mode == "oblate":
        oblate_x = params.req* np.cos(PHI) * np.cos(params.phi_p) - params.rpol * np.sin(PHI) * np.sin(params.phi_p) + H2
        oblate_y = params.req* np.cos(PHI) * np.sin(params.phi_p) + params.rpol * np.sin(PHI) * np.cos(params.phi_p) + K2
        ax.plot(oblate_x, oblate_y, lw=1, label = "oblate")
    elif mode == "circular":
        circ_x = params.req* np.cos(PHI) * np.cos(params.phi_p) - params.req * np.sin(PHI) * np.sin(params.phi_p) + H2
        circ_y = params.req* np.cos(PHI) * np.sin(params.phi_p) + params.req * np.sin(PHI) * np.cos(params.phi_p) + K2
        ax.plot(circ_x, circ_y, lw=1, label = "circular")
    elif mode == "both":
        oblate_x = params.req* np.cos(PHI) * np.cos(params.phi_p) - params.rpol * np.sin(PHI) * np.sin(params.phi_p) + H2
        oblate_y = params.req* np.cos(PHI) * np.sin(params.phi_p) + params.rpol * np.sin(PHI) * np.cos(params.phi_p) + K2
        circ_x = params.req * np.cos(PHI) * np.cos(params.phi_p) - params.req * np.sin(PHI) * np.sin(params.phi_p) + H2
        circ_y = params.req * np.cos(PHI) * np.sin(params.phi_p) + params.req * np.sin(PHI) * np.cos(params.phi_p) + K2
        ax.plot(oblate_x, oblate_y, lw=1, label = "oblate")
        ax.plot(circ_x, circ_y, lw=1, label = "circular")

    ax.legend()
    
    return(ax)

def T_14(period, rp, b, a_r):
    """
    period (in days)
    rp: rp/rstar
    b: impact parameter
    a_r: semi-major axis/ stellar radius
    """
    x = math.sqrt(((1+rp)**2-b**2)/(a_r**2-b**2))
    t14 = period/math.pi * math.asin(x)
    print("T14: {}".format(t14))
    return(t14)

def lightcurves2(params, times, PHI):

    fluxratio = []
    X, Y, Z = on_sky(params, times=times)

    if params.rpol is None:
        print("circular")
        params.rpol = params.req

    # if params.f >= 0: # if spherical condition
    cy, AA, BB, CC, DD, EE, FF, H2_TR, K2_TR = qrt_coeff(params, X, Y)
    
    X_neg = X < 0
    dist = np.linalg.norm(np.stack((X, Y), axis = 1), axis = 1)

    for i in range(0,len(X)):
        overlap_area = flux_ratio(params, dist[i], cy[i].tolist(), AA, BB, CC, DD[i], EE[i], FF[i], H2_TR[i], K2_TR[i])
        fluxratio.append(1 - overlap_area)
    
    # flux_jax = jax.vmap(light_curve, in_axes=(None, 0, None))(params.u, jnp.array(fluxratio), params.reff)

    
    # dist[X_neg] = -dist[X_neg]

    # # with open("avishi numpy coeff.csv", 'w') as csv_file:
    # #     header = ['time', 'fluxratio']
    # #     writer = csv.DictWriter(csv_file, fieldnames=header)
    # #     writer.writeheader()
    # #     arlen = np.arange(0,len(X))
    # #     for i in arlen:
    # #         writer.writerow({'time': times[i], 'fluxratio': fluxratio[i]})
    
    return fluxratio




