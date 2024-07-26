import numpy as np
import matplotlib.pyplot as plt
import math
from orbit import on_sky
from ferrari_solve import overlap_area, qrt_coeff, qrt_solve, verify_roots
import copy

class PlanetSystem:
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

    def __init__(
        self, 
        t0 = 0,          # time of inferior conjunction, in days
        tref = 0,        # reference time for transit, in days
        times = None,    # array of times
        per = None,          # orbital period, in days
        req = None,          # equatorial radius of planet
        rp = None,          # rp/rstar
        rpol = None,        # polar radius of planet
        rstar_eq = 1,       # equatorial radius of star
        rstar_pol = 1,      # polar radius of star
        reff = None,        # effective radius of planet, preserves area between oblate and spherical planet
        f = 0,              # oblateness
        a = None,              # semi-major axis, in units of R*
        b = None,              # impact parameter
        ecc = 0,         # eccentricity, in radians?
        w = np.pi,          # longitude of periastron (in radians) # not sure if this is labeled correctly
        u = [0.0, 0.0],     # limb-darkening coefficients
        phi_p = 0,      # obliquity angle of planet
        phi_s = 0,          # obliquity angle of star
        pos_star = (0, 0),  # position of star
        omega = None):      # argument of periapsis (in radians)

        state_keys = list(locals().keys())
        state_keys.remove("self")
        
        state = {}
        
        for key in state_keys:
            state[key] = locals()[key]
        
        self._state = state # store all the parameters and values in a dictionary
    
        # set rpol and reff if given req
        if self._state["req"] is not None and self._state["f"] != 0:
            self._state["rpol"] = self._state["req"] * (1 - self._state["f"])
            self._state["reff"] = (self._state["req"] * self._state["rpol"])**0.5
        else:
            self._state["rpol"] = None
            self._state["reff"] = self._state["rp"]

        if self._state["b"] is not None and self._state["a"] is not None:
            self._state["inc"] = math.acos(self._state["b"]/self._state["a"]) # in radians
            
    # def state(self):
    #     return(copy.deepcopy(self._state))

    def lightcurve(self, params={}):
        return(_lightcurve(state = self._state, params = params))


def _lightcurve(state, params={}):
    """
    Makes the lightcurve

    Args:
        state: TransitParams object
        times: array of times

    Returns:
        fluxratio: array of obscured flux; the lightcurve
        rootnum: array of number of intersection points for each time
    """
    for key in params.keys():
        state[key] = params[key]
    
    fluxratio = []
    nintpts = []
    X, Y = on_sky(state)

    if state["rpol"] is None:
        state["rpol"] = state["req"]

    cy, AA, BB, CC, DD, EE, FF, H2_TR, K2_TR = qrt_coeff(state, X, Y)
    
    dist = np.linalg.norm(np.stack((X, Y), axis = 1), axis = 1)

    for i in range(0, len(X)):
        overlap_area, intpts = flux_driver(state, dist[i], cy[i].tolist(), AA, BB, CC, DD[i], EE[i], FF[i], H2_TR[i], K2_TR[i])
        fluxratio.append(1 - overlap_area)
        nintpts.append(intpts)
    
    return np.array(fluxratio), np.array(nintpts)

## main flux driving function
def flux_driver(state, s, cy, AA, BB, CC, DD, EE, FF, H2_TR, K2_TR):
    """
    Main function that drives the flux calculation

    Args:
        state: TransitParams; object that stores all the transit model parameters
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
    if (state["rstar_eq"] is not None) or (state["rstar_pol"] is not None) or (state["req"] is not None) or (state["rpol"] is not None) or (state["rprstar"] is not None):
        if(state["rstar_eq"]<=0.0 or state["rstar_pol"]<=0.0 or state["req"]<=0.0 or state["rpol"]<=0.0):
            raise Exception("Zero or negative rstar_eq, rstar_pol, req, or rpol!")
    
    #rotation angles should be between -2pi and 2pi
    if(abs(state["phi_s"])>(math.pi)):
        state["phi_s"] = state["phi_s"]%(math.pi)
    if(abs(state["phi_p"])>(math.pi)):
        state["phi_p"] = state["phi_p"]%(math.pi)
    
    # oblate planet
    if (state["u"][0] or state["u"][1]) == 0: # if no limb-darkening
        area, nintpts = oblate_uniform(state, s, cy, AA, BB, CC, DD, EE, FF, H2_TR, K2_TR)
        return(area, nintpts)
    elif (state["u"][0] and state["u"][1]) > 0:  # if limb-darkening
        raise Exception("No support for limb-darkening, please set state["u"] to [0, 0]")

####################
# cases
####################

def oblate_uniform(state, s, cy, AA, BB, CC, DD, EE, FF, H2_TR, K2_TR):
    """
    Calculates the area of overlap between the planet and star for an oblate planet with uniform brightness using Ferarri's method

    case 1: planet fully outside of star area
    case 2: planet very close to outer limb of star, unsure about intersection
    case 3: planet definitely intersecting star
    case 4: planet very close to inner limb of star, unsure about intersection
    case 5: planet definitely transiting star, fully within stellar bounds
    """

    if s > (state["rstar_eq"] + state["req"]): # case 1
        nintpts = 0
        xint = None
        yint = None

        OverlapArea = overlap_area(nintpts, xint, yint, state,H2_TR,K2_TR,AA,BB,CC,DD,EE,FF)

    elif (state["rstar_eq"] - state["req"]) < s < (state["rstar_eq"] + state["req"]): # case 2, 3, 4 combined
        nychk, ychk = qrt_solve(cy, state)
        nintpts, xint, yint = verify_roots(nychk, ychk, state, AA, BB, CC, DD, EE, FF)
        OverlapArea = overlap_area(nintpts, xint, yint, state,H2_TR,K2_TR,AA,BB,CC,DD,EE,FF)    

    elif s < (state["rstar_eq"] - state["req"]): # case 5
        
        nintpts = 0
        xint = None
        yint = None
        OverlapArea = overlap_area(nintpts, xint, yint, state,H2_TR,K2_TR,AA,BB,CC,DD,EE,FF)

    else:
        # definite error
        return(0)

    return(OverlapArea, nintpts)


def viz(state, PHI, H2, K2, mode):
    # for visualization/debugging only!!
    star_x = state["rstar_eq"]* np.cos(PHI) * np.cos(state["phi_s"]) - state["rstar_pol"] * np.sin(PHI) * np.sin(state["phi_s"]) + state["pos_star"][0]
    star_y = state["rstar_eq"]* np.cos(PHI) * np.sin(state["phi_s"]) + state["rstar_pol"] * np.sin(PHI) * np.cos(state["phi_s"]) + state["pos_star"][1]
    
    fig,ax = plt.subplots(figsize = (5, 5))
    ax.plot(star_x, star_y, lw=2, label = "star")

    if mode == "oblate":
        oblate_x = state["req"]* np.cos(PHI) * np.cos(state["phi_p"]) - state["rpol"] * np.sin(PHI) * np.sin(state["phi_p"]) + H2
        oblate_y = state["req"]* np.cos(PHI) * np.sin(state["phi_p"]) + state["rpol"] * np.sin(PHI) * np.cos(state["phi_p"]) + K2
        ax.plot(oblate_x, oblate_y, lw=1, label = "oblate")
    
    elif mode == "circular":
        circ_x = state["req"]* np.cos(PHI) * np.cos(state["phi_p"]) - state["req"] * np.sin(PHI) * np.sin(state["phi_p"]) + H2
        circ_y = state["req"]* np.cos(PHI) * np.sin(state["phi_p"]) + state["req"] * np.sin(PHI) * np.cos(state["phi_p"]) + K2
        ax.plot(circ_x, circ_y, lw=1, label = "circular")
    
    elif mode == "both":
        oblate_x = state["req"]* np.cos(PHI) * np.cos(state["phi_p"]) - state["rpol"] * np.sin(PHI) * np.sin(state["phi_p"]) + H2
        oblate_y = state["req"]* np.cos(PHI) * np.sin(state["phi_p"]) + state["rpol"] * np.sin(PHI) * np.cos(state["phi_p"]) + K2
        circ_x = state["req"] * np.cos(PHI) * np.cos(state["phi_p"]) - state["req"] * np.sin(PHI) * np.sin(state["phi_p"]) + H2
        circ_y = state["req"] * np.cos(PHI) * np.sin(state["phi_p"]) + state["req"] * np.sin(PHI) * np.cos(state["phi_p"]) + K2
        ax.plot(oblate_x, oblate_y, lw=1, label = "oblate")
        ax.plot(circ_x, circ_y, lw=1, label = "circular")

    ax.legend()
    
    return(ax)





