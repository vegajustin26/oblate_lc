import numpy as np
import matplotlib.pyplot as plt
import math
from orbit import on_sky
from ferrari_solve import overlap_area, qrt_coeff, qrt_solve, verify_roots

class TransitParams(object):
    """
    Object that stores all the transit model parameters
    
    Attributes
    ----------
    t0 : float
        time of inferior conjunction, in days
    tref : float
        reference time for transit, in days
    per : float
        orbital period, in days
    req : float
        equatorial radius of planet
    rprstar : float
        rp/rstar
    rpol : float
        polar radius of planet
    rstar_eq : float
        equatorial radius of star
    rstar_pol : float
        polar radius of star
    reff : float
        effective radius of planet
    f : float
        oblateness
    a : float
        semi-major axis, in units of R*
    b : float
        impact parameter
    ecc : float
        eccentricity, in radians?
    w : float
        longitude of periastron (in radians)
    u : list
        limb-darkening coefficients
    phi_p : float
        obliquity angle of planet
    phi_s : float
        obliquity angle of star
    pos_star : tuple
        position of star
    omega : float
        argument of periapsis (in radians)
    inc : float
        inclination angle of orbit (in radians)
    """


    def __init__(self):
        self.t0 = None          # time of inferior conjunction, in days
        self.tref = None        # reference time for transit, in days
        self.per = None         # orbital period, in days
        self.req = None         # equatorial radius of planet
        self.rprstar = None     # rp/rstar
        self.rpol = None        # polar radius of planet
        self.rstar_eq = 1       # equatorial radius of star
        self.rstar_pol = 1      # polar radius of star
        self.f = 0              # oblateness
        self.a = None           # semi-major axis, in units of R*
        self.b = None           # impact parameter
        self.ecc = None         # eccentricity, in radians?
        self.w = np.pi           # longitude of periastron (in radians) # not sure if this is labeled correctly
        self.u = [0.0, 0.0]           # limb-darkening coefficients
        self.phi_p = 0          # obliquity angle of planet
        self.phi_s = 0          # obliquity angle of star
        self.pos_star = (0, 0)  # position of star
        self.omega = None       # argument of periapsis (in radians)

    def set_f(self, value):
        """
        sets the oblateness of the planet
        """
        self.f = value
        self.calculate_rpol()

    def calculate_rpol(self):
        """
        calculates the polar radius of the planet
        """
        if self.req is not None and self.f != 0:
            self.rpol = self.req * (1 - self.f)
        else:
            self.rpol = None

    def set_b(self, value):
        """
        sets the impact parameter of the planet
        """
        self.b = value
        self.calculate_inc()

    def calculate_inc(self):
        """
        calculates the inclination angle of the orbit
        """

        if self.b is not None and self.a is not None:
            self.inc = math.acos(self.b/self.a) # in radians

## main flux driving function
def flux_driver(params, s, cy, AA, BB, CC, DD, EE, FF, H2_TR, K2_TR):
    """
    Main function that drives the flux calculation

    Args:
        params : TransitParams; object that stores all the transit model parameters
        s : distance between planet and star
        cy : list of coefficients
        H2_TR: x-coordinate of the center of the planet, translated and rotated
        K2_TR: y-coordinate of the center of the planet, translated and rotated
        AA: coefficient of x^2
        BB: coefficient of xy
        CC: coefficient of y^2
        DD: coefficient of x
        EE: coefficient of y
        FF: constant term

    Returns:
        OverlapArea : float; area of overlap between planet and star
        nintpts : int; number of intersection points between planet and star
    """

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
    if (params.u[0] or params.u[1]) == 0: # if no limb-darkening
        area, nintpts = oblate_uniform(params, s, cy, AA, BB, CC, DD, EE, FF, H2_TR, K2_TR)
        return(area, nintpts)
    elif (params.u[0] and params.u[1]) > 0:  # if limb-darkening
        raise Exception("No support for limb-darkening, please set params.u to [0, 0]")

####################
# cases
####################

def oblate_uniform(params, s, cy, AA, BB, CC, DD, EE, FF, H2_TR, K2_TR):
    """
    Calculates the area of overlap between the planet and star for an oblate planet with uniform brightness using Ferarri's method

    case 1: planet fully outside of star area
    case 2: planet very close to outer limb of star, unsure about intersection
    case 3: planet definitely intersecting star
    case 4: planet very close to inner limb of star, unsure about intersection
    case 5: planet definitely transiting star, fully within stellar bounds
    """

    if s > (params.rstar_eq + params.req): # case 1
        nintpts = 0
        xint = None
        yint = None

        OverlapArea = overlap_area(nintpts, xint, yint, params,H2_TR,K2_TR,AA,BB,CC,DD,EE,FF)

    elif (params.rstar_eq - params.req) < s < (params.rstar_eq + params.req): # case 2, 3, 4 combined
        nychk, ychk = qrt_solve(cy, params)
        nintpts, xint, yint = verify_roots(nychk, ychk, params, AA, BB, CC, DD, EE, FF)
        OverlapArea = overlap_area(nintpts, xint, yint, params,H2_TR,K2_TR,AA,BB,CC,DD,EE,FF)    

    elif s < (params.rstar_eq - params.req): # case 5
        
        nintpts = 0
        xint = None
        yint = None
        OverlapArea = overlap_area(nintpts, xint, yint, params,H2_TR,K2_TR,AA,BB,CC,DD,EE,FF)

    else:
        # definite error
        return(0)

    return(OverlapArea, nintpts)


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

def ob_lightcurve(params, times):
    """
    Makes the lightcurve

    Args:
        params: TransitParams object
        times: array of times

    Returns:
        fluxratio: array of obscured flux; the lightcurve
        rootnum: array of number of intersection points for each time
    """
    fluxratio = []
    rootnum = []
    X, Y, Z = on_sky(params, times=times)

    if params.rpol is None:
        print("circular")
        params.rpol = params.req

    # if params.f >= 0: # if spherical condition
    cy, AA, BB, CC, DD, EE, FF, H2_TR, K2_TR = qrt_coeff(params, X, Y)
    
    X_neg = X < 0
    dist = np.linalg.norm(np.stack((X, Y), axis = 1), axis = 1)

    for i in range(0,len(X)):
        overlap_area, nintpts = flux_driver(params, dist[i], cy[i].tolist(), AA, BB, CC, DD[i], EE[i], FF[i], H2_TR[i], K2_TR[i])
        fluxratio.append(1 - overlap_area)
        rootnum.append(nintpts)
    
    return np.array(fluxratio), np.array(rootnum)




