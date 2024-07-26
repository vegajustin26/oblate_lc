import math
import numpy as np

def true_anomaly(state):
    """
    finds true anomaly given eccentricity, period, and times
    uses reference points M_ref, E_ref, and tref to calculate E, M, and f

    Args:
        state: TransitParams object
        times: array of times
    """

    def danby(E, M):
        return(E - state["ecc"]*np.sin(E) - M)
    
    f_ref = math.pi/2 - state["omega"]
    e_k = math.sqrt((1 + state["ecc"])/(1 - state["ecc"]))

    if state["ecc"] == 0:
        E_ref = f_ref
        M_ref = E_ref

        M = ((2 * math.pi)/state["per"]) * (state["times"] - state["tref"]) + M_ref
        E = M
        f = E

    elif 0 < state["ecc"] < 1:
        print("Eccentricities between 0-1 not well-tested, use with caution!")
        E_ref = 2 * math.atan(math.tan(f_ref/2)/e_k)

        if E_ref < -(math.pi/2):
            E_ref += (2 * math.pi)

        M_ref = E_ref - (state["ecc"] * math.sin(E_ref))

        diff = 999
        i=0

        M = (2 * math.pi/state["per"]) * (state["times"] - state["tref"]) + M_ref
        E = M

        while np.max(diff) > 1e-12:
            E_new = E - danby(E, M)/(1-state["ecc"]*np.cos(E))
            diff = np.abs(E-E_new)
            E = E_new
            i+=1

        f = 2 * np.arctan(e_k * np.tan(E/2))
        
    else:
        raise Exception("eccentricity is negative or greater than 1!")
        
    return(f)

def on_sky(state):
    """
    Times, period in the same units
    this is coordinate transformation from on-sky to orbital plane transit coordinates

    Args:
        state: TransitParams object
        times: array of times

    Returns:
        X, Y, Z: orbit coordinates
    """ 

    f = true_anomaly(state)
    r = (state["a"]*(1-state["ecc"]**2)) / (1+state["ecc"]*np.cos(f))
    
    X =  r * (np.cos(state["w"])*np.cos(state["omega"]+f) - np.sin(state["w"])*np.sin(state["omega"]+f)*np.cos(state["inc"]))
    Y = r * (np.sin(state["w"])*np.cos(state["omega"]+f) + np.cos(state["w"])*np.sin(state["omega"]+f)*np.cos(state["inc"]))
    # Z = r * np.sin(state["inc"])*np.sin(state["omega"]+f)

    return(X, Y)

    