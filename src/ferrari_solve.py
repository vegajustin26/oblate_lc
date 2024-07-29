import math
import numpy as np

def overlap_area(nintpts, xint, yint, state, H2_TR, K2_TR, AA, BB, CC, DD, EE, FF):
    """
    driver function for calculating the overlapping area of two ellipses
    
    Args:
        nintpts: number of intersection points
        xint: x-coordinates of the intersection points
        yint: y-coordinates of the intersection points
        params: parameters of the transit model
        H2_TR: x-coordinate of the center of the planet, translated and rotated
        K2_TR: y-coordinate of the center of the planet, translated and rotated
        AA: coefficient of x^2
        BB: coefficient of xy
        CC: coefficient of y^2
        DD: coefficient of x
        EE: coefficient of y
        FF: constant term

    Returns:
        OverlapArea: overlapping area of the two ellipses
    """

    if(nintpts == 0 or nintpts == 1):
        OverlapArea = nointpts(state, H2_TR,K2_TR,FF)
        return OverlapArea
    elif(nintpts == 2):
        OverlapArea = twointpts(xint,yint,state, H2_TR, K2_TR, AA,BB,CC,DD,EE,FF)
        return OverlapArea

def qrt_coeff(state, H2, K2):
    """
    Calculates the coefficients of Equation 15 in Hughes and Chraibi 2011

    Args:
        params: parameters of the transit model
        H2: x-coordinate of the center of the planet
        K2: y-coordinate of the center of the planet

    Returns:
        cy: coefficients of the quartic equation
        AA: coefficient of x^2
        BB: coefficient of xy
        CC: coefficient of y^2
        DD: coefficient of x
        EE: coefficient of y
        FF: constant term
        H2_TR: x-coordinate of the center of the planet, translated and rotated
        K2_TR: y-coordinate of the center of the planet, translated and rotated
    """

    cosphi = math.cos(state["phi_s"])
    sinphi = math.sin(state["phi_s"])
    H2_TR  = (H2-state["pos_star"][0])*cosphi + (K2-state["pos_star"][1])*sinphi
    K2_TR  = (state["pos_star"][0]-H2)*sinphi + (K2-state["pos_star"][1])*cosphi
    PHI_2R = state["phi_p"] - state["phi_s"]
    if(abs(PHI_2R)>(2*math.pi)):
        PHI_2R = PHI_2R%(2*math.pi)
    
    #Calculate the (implicit) polynomial coefficients for the second ellipse in its translated (by (-state["pos_star"][0],-H2)) and rotated (by -phi_1) position: AA*x^2 + BB*x*y + CC*y^2 + DD*x + EE*y + FF = 0
    
    cosphi       = math.cos(PHI_2R)
    cosphi2      = cosphi*cosphi
    sinphi       = math.sin(PHI_2R)
    sinphi2      = sinphi*sinphi
    cosphisinphi = 2.0*cosphi*sinphi
    A22          = state["req"]*state["req"]
    B22          = state["rpol"]*state["rpol"]
    tmp0         = (cosphi*H2_TR + sinphi*K2_TR)/A22
    tmp1         = (sinphi*H2_TR - cosphi*K2_TR)/B22
    tmp2         = cosphi*H2_TR + sinphi*K2_TR
    tmp3         = sinphi*H2_TR - cosphi*K2_TR
    
    #Implicit polynomial coefficients for the second ellipse
    
    AA = cosphi2/A22 + sinphi2/B22 # does not change
    BB = cosphisinphi/A22 - cosphisinphi/B22 # does not change
    CC = sinphi2/A22 + cosphi2/B22 # does not change
    DD = -2.0*cosphi*tmp0 - 2.0*sinphi*tmp1
    EE = -2.0*sinphi*tmp0 + 2.0*cosphi*tmp1
    FF = tmp2*tmp2/A22 + tmp3*tmp3/B22 - 1.0
    
    #Create and solve the quartic equation to find intersection points.
    #If execution arrives here, the ellipses are atleast 'close' to intersecting.
    #Coefficients for the quartic polynomial in y are calculated from the two implicit equations. 
    e = (state["rstar_eq"]**4.0)*AA*AA + state["rstar_pol"]*state["rstar_pol"]*(state["rstar_eq"]*state["rstar_eq"]*(BB*BB - 2.0*AA*CC) + state["rstar_pol"]*state["rstar_pol"]*CC*CC)
    d = 2.0*state["rstar_pol"]*(state["rstar_pol"]*state["rstar_pol"]*CC*EE + state["rstar_eq"]*state["rstar_eq"]*(BB*DD - AA*EE))
    c = state["rstar_eq"]*state["rstar_eq"]*((state["rstar_pol"]*state["rstar_pol"]*(2.0*AA*CC-BB*BB) + DD*DD - 2.0*AA*FF) - 2.0*state["rstar_eq"]*state["rstar_eq"]*AA*AA) + state["rstar_pol"]*state["rstar_pol"]*(2.0*CC*FF + EE*EE)
    b = 2.0*state["rstar_pol"]*(state["rstar_eq"]*state["rstar_eq"]*(AA*EE - BB*DD) + EE*FF)
    a = (state["rstar_eq"]*(state["rstar_eq"]*AA - DD) + FF)*(state["rstar_eq"]*(state["rstar_eq"]*AA + DD) + FF)

    e = np.full((H2.size, 1), e)
    
    cy = np.column_stack((a, b, c, d, e))

    
    return(cy, AA, BB, CC, DD, EE, FF, H2_TR, K2_TR)

def qrt_solve(cy, state):
    """
    Solves the quartic equation (Eq. 15 in Hughes and Chraiabi 2011) for the intersection points of the two ellipses

    Args:
        cy: coefficients of the quartic equation
        params: parameters of the transit model

    Returns:
        nychk: number of real roots
        ychk: y-values of the intersection points of the two ellipses

    """

    py = [0] * 5
    r  = [[0.0,0.0,0.0,0.0,0.0],[0.0,0.0,0.0,0.0,0.0],[0.0,0.0,0.0,0.0,0.0]]
    
    if(abs(cy[4]) > 0.0): # if first coefficient not zero

        #Quartic coefficient nonzero -> use quartic formula
        for i in range(0,4):
            py[4-i] = cy[i]/cy[4]
        py[0] = 1.0

        BIQUADROOTS(py,r)
        # print("r {}".format(r))
        nroots = 4
    
    elif(abs(cy[3])>0.0):

        #Quartic degenerates to cubic -> use cubic formula
        for i in range(0,3):
            py[3-i] = cy[i]/cy[3]
        py[0] = 1.0

        CUBICROOTS(py,r)
        nroots = 3
        
    elif(abs(cy[2])>0.0):

        #Quartic degenerates to quadratic -> use quadratic formula
        for i in range(0,2):
            py[2-i] = cy[i]/cy[2]
        py[0] = 1.0
        
        QUADROOTS(py,r)
        nroots = 2
        
    elif(abs(cy[1])>0.0):

        #Quartic degenerates to linear -> solve directly
        #cy[1]*Y + cy[0] = 0
        
        r[1][1] = (-cy[0]/cy[1])
        r[2][1] = 0.0
        nroots = 1
    else:
        # Ellipses are identical
        nroots = 0

    #Determine which roots are real; discard any complex roots
    nychk = 0
    ychk = [0] * (nroots+1)
    for i in range(1,nroots+1):
        if(abs(r[2][i])<(10**(-7))):
            nychk = nychk+1
            ychk[nychk] = r[1][i]*state["rstar_pol"]
    
    #Sort the real roots by straight insertion
    for j in range(2,nychk+1):
        tmp0 = ychk[j]
        for k in range(j-1,0,-1):
            if(ychk[k] <= tmp0):
                break
            ychk[k+1] = ychk[k]
        ychk[k+1] = tmp0
    
    return(nychk, ychk) 

def QUADROOTS(p,r):
    """
    QUADROOTS numerically solves the quadratic equation p[0]*x^2 + p[1]*x + p[2] = 0 for real roots x.
    
    Args:
        p: coefficients of the quadratic equation
        r: roots of the quadratic equation

    Returns:
        r: roots of the quadratic equation
    """

    b = -p[1]/(2.0*p[0])
    c = p[2]/p[0]
    d = b*b - c
    
    if(d>=0.0):
        if(b>0.0):
            r[1][2]=(d**0.5+b)
            b = r[1][2]
        else:
            r[1][2]=(-(d)**0.5+b)
            b = r[1][2]
        r[1][1] = c/b
        r[2][2] = 0.0
        r[2][1] = r[2][2]
    else:
        r[2][1]=(-d)**0.5
        d = r[2][1]
        r[2][2] = -d
        r[1][2] = b
        r[1][1] = r[1][2]
    return

def CUBICROOTS(p,r):
    """
    CUBICROOTS solves the cubic equation p[0]*x^3 + p[1]*x^2 + p[2]*x + p[3] = 0 for real roots x.

    Args:
        p: coefficients of the cubic equation
        r: roots of the cubic equation

    Returns:
        r: roots of the cubic equation

    """

    if(p[0] != 1.0):
        for k in range(1,5):
            p[k] = p[k]/p[0]
        p[0] = 1.0
    s = p[1]/3.0
    t = s*p[1]
    b = 0.5*(s*(t/1.5-p[2])+p[3])
    t = (t-p[2])/3.0
    c = t*t*t
    d = b*b - c
    if(d>=0.0):
        d = ((d**0.5)+abs(b))**(1.0/3.0)
        if(d!=0.0):
            if(b>0.0):
                b = -d
            else:
                b = d
            c = t/b
        r[2][2] = ((0.75)**(0.5))*(b-c)
        d = r[2][2]
        b = b+c
        r[1][2] = -0.5*b-s
        c = r[1][2]
        if((b>0.0 and s<=0.0) or (b<0.0 and s>0.0)):
            r[1][1] = c
            r[2][1] = -d
            r[1][3] = b-s
            r[2][3] = 0.0
        else:
            r[1][1] = b-s
            r[2][1] = 0.0
            r[1][3] = c
            r[2][3] = -d
    else:
        if(b == 0.0):
            d = math.atan(1.0)/1.5
        else:
            d = math.atan(((-d)**(0.5))/abs(b))/3.0
        if(b<0.0):
            b = (t**(0.5))*2.0
        else:
            b = -2.0*(t**(0.5))
        c = math.cos(d)*b
        t = -((0.75)**(0.5))*math.sin(d)*b - 0.5*c
        d = -t-c-s
        c = c-s
        t = t-s
        if(abs(c)>abs(t)):
            r[1][3] = c
        else:
            r[1][3] = t
            t = c
        if(abs(d)>abs(t)):
            r[1][2] = d
        else:
            r[1][2] = t
            t       = d
        r[1][1] = t
        for k in range(1,5):
            r[2][k] = 0.0
    return

def quad(c,b,p,r,e):
    """
    helper function for QUADROOTS
    """

    p[2] = c/b
    QUADROOTS(p,r)
    for k in range(1,3):
        for j in range(1,3):
            r[j][k+2] = r[j][k]
    p[1] = -p[1]
    p[2] = b
    QUADROOTS(p,r)
    for k in range(1,5):
        r[1][k] = r[1][k] - e
    return 

def BIQUADROOTS(p,r):
    """
    BIQUADROOTS solves the quartic equation p[0]*x^4 + p[1]*x^3 + p[2]*x^2 + p[3]*x + p[4] = 0 for real roots x.

    Args:
        p: coefficients of the quartic equation
        r: roots of the quartic equation

    Returns:
        r: roots of the quartic equation

    """
    
    if(p[0] != 1.0):
        for k in range(1,5):
            p[k] = p[k]/p[0]
        p[0] = 1.0
    e    = 0.25*p[1]
    b    = 2.0*e
    c    = b*b
    d    = 0.75*c
    b    = p[3]+b*(c-p[2])
    a    = p[2]-d
    c    = p[4]+e*(e*a-p[3])
    a    = a-d
    p[1] = 0.5*a
    p[2] = (p[1]*p[1]-c)*0.25
    p[3] = b*b/(-64.0)

    if(p[3]<0.0):
        CUBICROOTS(p,r)
        for k in range(1,4):
            if(r[2][k]==0.0 and r[1][k]>0.0):
                d = r[1][k]*4.0
                a = a+d
                if(a>=0.0 and b>=0.0):
                    p[1] = d**(0.5)
                elif(a<=0 and b<=0.0):
                    p[1] = d**(0.5)
                else:
                    p[1] = -(d)**(0.5)
                b = 0.5*(a+b/p[1])
                quad(c,b,p,r,e)
                return

    if(p[2]<0.0):
        b    = c**(0.5)
        d    = b+b-a
        p[1] = 0.0
        if(d>0.0):
            p[1] = d**(0.5)
    else:
        if(p[1]>0.0):
            b = ((p[2])**(0.5))*2.0 + p[1]
        else:
            b = -((p[2])**(0.5))*2.0 + p[1]
        if(b != 0.0):
            p[1] = 0.0
        else:
            for k in range(1,5):
                r[1][k] = -e
                r[2][k] = 0.0
            return
    quad(c,b,p,r,e)

def ellipse2tr(x,y,AA,BB,CC,DD,EE,FF):
    """
    Implicit equation of overlapping ellipses, eq. 4a in Hughes and Chraibi 2011

    Args:
        x: x-coordinate of the intersection points
        y: y-coordinate of the intersection points
        AA: coefficient of x^2
        BB: coefficient of xy
        CC: coefficient of y^2
        DD: coefficient of x
        EE: coefficient of y
        FF: constant term

    Returns:
        return: value of the implicit equation
    """

    return((AA*x*x) + (BB*x*y) + (CC*y*y) + (DD*x) + (EE*y) + (FF))

def nointpts(state, H2_TR,K2_TR,FF):
    """
    
    Routine for finding the area of the intersection of two ellipses when there are zero or one intersection points. 
    
    Args:
        params: parameters of the transit model
        H2_TR: x-coordinate of the center of the planet, translated and rotated
        K2_TR: y-coordinate of the center of the planet, translated and rotated
        FF: constant term

    Returns:
        return: zero if ellipses do not overlap, or area of first or second ellipse if one ellipse is inside the other
    """

    #The relative size of the two ellipses can be found from the axis lengths.
    relsize = (state["rstar_eq"]*state["rstar_pol"]) - (state["req"]*state["rpol"])
    if(relsize>0.0):
        #First ellipse is larger than second ellipse. (star bigger than planet)
        if(((H2_TR*H2_TR)/(state["rstar_eq"]*state["rstar_eq"])+(K2_TR*K2_TR)/(state["rstar_pol"]*state["rstar_pol"]))<1.0):
            #Ellipse 2 is inside ellipse 1.
            return (state["req"]*state["rpol"])
        else:
            #Disjoint ellipses.
            return 0.0
    elif(relsize<0.0):
        #Second ellipse is larger than first ellipse.
        if(FF<0.0):
            #Ellipse 1 inside ellipse 2.
            return (state["rstar_eq"]*state["rstar_pol"])
        else:
            #Disjoint ellipses.
            return 0.0
    else:
        #Ellipses are identical.
        return (state["rstar_eq"]*state["rstar_pol"])

def twointpts(x,y,state, H2_TR,K2_TR,AA,BB,CC,DD,EE,FF):
    """
    Routine for finding the area of the intersection of two ellipses when there are two intersection points.

    Args:
        x: x-coordinates of the intersection points
        y: y-coordinates of the intersection points
        params: parameters of the transit model
        H2_TR: x-coordinate of the center of the planet, translated and rotated
        K2_TR: y-coordinate of the center of the planet, translated and rotated
        AA: coefficient of x^2
        BB: coefficient of xy
        CC: coefficient of y^2
        DD: coefficient of x
        EE: coefficient of y
        FF: constant term

    Returns:
        return: overlapping area of the two ellipses
    """


    #Find the parametric angles for each point on ellipse 1.

    # x intersection point is bounded by state["rstar_eq"]=[-1, 1]
    if(abs(x[1])>state["rstar_eq"]):
        
        if(x[1]<0.0):
            x[1] = -state["rstar_eq"]
        else:
            x[1] = state["rstar_eq"]

    if(abs(x[2])>state["rstar_eq"]):
        if(x[2]<0.0):
            x[2] = -state["rstar_eq"]
        else:
            x[2] = state["rstar_eq"]

    # calculate angle if yint in Q3 or Q4 
    if(y[1]<0.0):
        theta1 = (2.0*math.pi) - math.acos(x[1]/state["rstar_eq"])
    else:
        theta1 = math.acos(x[1]/state["rstar_eq"])
    if(y[2]<0.0): #Quadrant III or IV
        theta2 = (2.0*math.pi) - math.acos(x[2]/state["rstar_eq"])
    else:
        theta2 = math.acos(x[2]/state["rstar_eq"])

    
    #Logic is for proceeding counterclockwise from theta1 to theta2.
    if(theta1>theta2):
        tmp    = theta1
        theta1 = theta2
        theta2 = tmp
        
    # midpoint between intersection points, on stellar boundary
    xmid = state["rstar_eq"]*math.cos((theta1+theta2)/2.0)
    ymid = state["rstar_pol"]*math.sin((theta1+theta2)/2.0)
    
    #The point (xmid,ymid) is on the first ellipse 'between' the two intersection points (x[1],y[1]) and (x[2],y[2]) when travelling counter-clockwise from (x[1],y[1]) to (x[2],y[2]). If the point (xmid,ymid) is inside the second ellipse, then the desired segment of ellipse 1 contains the point (xmid,ymid), so integrate counterclockwise from (x[1],y[1]) to (x[2],y[2]). Otherwise, integrate counterclockwise from (x[2],y[2]) to (x[1],y[1]).
    if(ellipse2tr(xmid,ymid,AA,BB,CC,DD,EE,FF)>0.0):
        tmp    = theta1
        theta1 = theta2
        theta2 = tmp
        
    #Here is the ellipse segment for the first ellipse.
    if(theta1>theta2):
        theta1 = theta1 - (2.0*math.pi)
    
    if((theta2-theta1)>math.pi):
        trsign = 1.0
    else:
        trsign = -1.0
                    
    area1 = 0.5*(state["rstar_eq"]*state["rstar_pol"]*(theta2-theta1) + trsign*abs(x[1]*y[2]-x[2]*y[1]))
    
    if(area1<0):
        area1 = area1+(state["rstar_eq"]*state["rstar_pol"])
    
    #Find ellipse 2 segment area.
    #The ellipse segment routine needs an ellipse that is centered at the origin and oriented with the coordinate axes. The intersection point (x[1],y[1]) and (x[2],y[2]) are found with both ellipses translated and rotated by (-state["pos_star"][0],-state["pos_star"][1]) and -state["phi_s"]. Further, translate and rotate the points to put the second ellipse at the origin and oriented with the coordinate axes. The translation is (-H2_TR,-K2_TR) and the rotation is -(state["phi_p"]-state["phi_s"]).
    cosphi = math.cos(state["phi_s"]-state["phi_p"])
    sinphi = math.sin(state["phi_s"]-state["phi_p"])
    
    # translated intersection points to center around origin
    x1_tr = (x[1]-H2_TR)*cosphi + (y[1]-K2_TR)*(-sinphi)
    y1_tr = (x[1]-H2_TR)*sinphi + (y[1]-K2_TR)*(cosphi)
    x2_tr = (x[2]-H2_TR)*cosphi + (y[2]-K2_TR)*(-sinphi)
    y2_tr = (x[2]-H2_TR)*sinphi + (y[2]-K2_TR)*(cosphi)

    #Determine which branch of the ellipse to integrate by finding a point of the second ellipse and checking whether it is inside the first ellipse (in their once translated+rotated positions).
    #Find the parametric angles for each point on ellipse 1.
    
    if(abs(x1_tr)>state["req"]):
        
        if(x1_tr<0.0):
            x1_tr = -state["req"] 
        else:
            x1_tr = state["req"]
    if(abs(x2_tr)>state["req"]):
        
        if(x2_tr<0.0):
            x2_tr = -state["req"]
        else:
            x2_tr = state["req"]

    # calculate angle if yint in Q3 or Q4   
    if(y1_tr<0.0): #Quadrant III or IV
        theta1 = (2.0*math.pi)-math.acos(x1_tr/state["req"])
    else: #Quadrant I or II
        theta1 = math.acos(x1_tr/state["req"])
    
    if(y2_tr<0.0):
        theta2 = (2.0*math.pi)-math.acos(x2_tr/state["req"])
    else: #Quadrant I or II
        theta2 = math.acos(x2_tr/state["req"])
    
    #Logic for proceeding counterclockwise from theta1 to theta2.
    if(theta1>theta2): # ensures theta2 > theta1
        tmp    = theta1
        theta1 = theta2
        theta2 = tmp


    #Find a point on the second ellipse that is different from the two intersection points.
    xmid = state["req"]*math.cos((theta1+theta2)/2.0)
    ymid = state["rpol"]*math.sin((theta1+theta2)/2.0)
    
    
    #Translate the point back to the second ellipse in its once translated+rotated position.
    cosphi  = math.cos(state["phi_p"]-state["phi_s"])
    sinphi  = math.sin(state["phi_p"]-state["phi_s"])
    
    xmid_rt = xmid*cosphi + ymid*(-sinphi) + H2_TR
    ymid_rt = xmid*sinphi + ymid*(cosphi)  + K2_TR 
    
    #The point (xmid_rt,ymid_rt) is on the second ellipse 'between' the intersection points (x[1],y[1]) and (x[2],y[2]) when traveling counterclockwise from (x[1],y[1]) to (x[2],y[2]). If the point (xmid_rt,ymid_rt) is inside the first ellipse, then the desired segment of ellipse 2 contains the point (xmid_rt,ymid_rt), so integrate counterclockwise from (x[2],y[2]) to (x[1],y[1]).
    if(((xmid_rt*xmid_rt)/(state["rstar_eq"]*state["rstar_eq"]) + (ymid_rt*ymid_rt)/(state["rstar_pol"]*state["rstar_pol"]))>1.0):
        tmp    = theta1
        theta1 = theta2 
        theta2 = tmp
        
    #Here is the ellipse segment routine for the second ellipse
    
    if(theta1>theta2):
        theta1 = theta1 - (2.0*math.pi)
    
    if((theta2-theta1)>math.pi):
        trsign = 1.0
    else:
        trsign = -1.0
    
    area2 = 0.5*(state["req"]*state["rpol"]*(theta2-theta1)+trsign*abs(x1_tr*y2_tr - x2_tr*y1_tr))
    
    if(area2<0.0):
        area2 = area2+(state["req"]*state["rpol"])

    #Two intersection points

    return ((area1+area2)/math.pi)

def verify_roots(nychk, ychk, state,  AA, BB, CC, DD, EE, FF):
    """
    Numerically verifies the roots of the quartic equation

    Args:
        nychk: number of real roots
        ychk: y-values of the intersection points of the two ellipses
        params: parameters of the transit model
        AA: coefficient of x^2
        BB: coefficient of xy
        CC: coefficient of y^2
        DD: coefficient of x
        EE: coefficient of y
        FF: constant term

    Returns:
        nintpts: number of intersection points
        xint: x-coordinates of the intersection points
        yint: y-coordinates of the intersection points
    """

    nintpts = 0
    xint = [0] * (nychk + 1)
    yint = [0] * (nychk + 1)
    
    for i in range(1,nychk+1):
        #check for multiple roots
        m = 0
        if(i>1):
            for j in range(0,i):
                if(math.isclose(ychk[i],ychk[j])):
                    m = 1
            if(m == 1):
                continue
            
        #check intersection points for ychk[i]
        if(abs(ychk[i])>state["rstar_pol"]):
            x1 = 0.0
        else:
            x1 = state["rstar_eq"]*((1.0-(ychk[i]*ychk[i])/(state["rstar_pol"]*state["rstar_pol"]))**0.5)
        x2 = -x1

        if(abs(ellipse2tr(x1,ychk[i],AA,BB,CC,DD,EE,FF))<(10**(-6))):
            nintpts = nintpts + 1
            if(nintpts>4):
                #Error in intersection points
                return -3.0
            xint[nintpts] = x1
            yint[nintpts] = ychk[i]

        if(abs(ellipse2tr(x2,ychk[i],AA,BB,CC,DD,EE,FF))<(10**(-6)) and abs(x2-x1)>(10**(-6))):
            nintpts = nintpts+1
            if(nintpts>4):
                #Error in intersection points
                return -4.0
            xint[nintpts] = x2
            yint[nintpts] = ychk[i]
    return(nintpts, xint, yint)
