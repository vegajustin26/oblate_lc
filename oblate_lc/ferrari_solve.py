import math
import numpy as np

# for oblate uniform
def overlap_area(nintpts, xint, yint, params,H2_TR,K2_TR,AA,BB,CC,DD,EE,FF):
    if(nintpts == 0 or nintpts == 1):
        OverlapArea = nointpts(params,H2_TR,K2_TR,FF)
        thetas = [0, 2*math.pi]
        return OverlapArea, thetas
    elif(nintpts == 2):
        OverlapArea, thetas = twointpts(xint,yint,params,H2_TR, K2_TR, AA,BB,CC,DD,EE,FF)
        return OverlapArea, thetas

def qrt_coeff(params, H2, K2):

    cosphi = math.cos(params.phi_s)
    sinphi = math.sin(params.phi_s)
    H2_TR  = (H2-params.pos_star[0])*cosphi + (K2-params.pos_star[1])*sinphi
    K2_TR  = (params.pos_star[0]-H2)*sinphi + (K2-params.pos_star[1])*cosphi
    PHI_2R = params.phi_p - params.phi_s
    if(abs(PHI_2R)>(2*math.pi)):
        PHI_2R = PHI_2R%(2*math.pi)
    
    #Calculate the (implicit) polynomial coefficients for the second ellipse in its translated (by (-params.pos_star[0],-H2)) and rotated (by -phi_1) position: AA*x^2 + BB*x*y + CC*y^2 + DD*x + EE*y + FF = 0
    
    cosphi       = math.cos(PHI_2R)
    cosphi2      = cosphi*cosphi
    sinphi       = math.sin(PHI_2R)
    sinphi2      = sinphi*sinphi
    cosphisinphi = 2.0*cosphi*sinphi
    A22          = params.req*params.req
    B22          = params.rpol*params.rpol
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
    e = (params.rstar_eq**4.0)*AA*AA + params.rstar_pol*params.rstar_pol*(params.rstar_eq*params.rstar_eq*(BB*BB - 2.0*AA*CC) + params.rstar_pol*params.rstar_pol*CC*CC)
    d = 2.0*params.rstar_pol*(params.rstar_pol*params.rstar_pol*CC*EE + params.rstar_eq*params.rstar_eq*(BB*DD - AA*EE))
    c = params.rstar_eq*params.rstar_eq*((params.rstar_pol*params.rstar_pol*(2.0*AA*CC-BB*BB) + DD*DD - 2.0*AA*FF) - 2.0*params.rstar_eq*params.rstar_eq*AA*AA) + params.rstar_pol*params.rstar_pol*(2.0*CC*FF + EE*EE)
    b = 2.0*params.rstar_pol*(params.rstar_eq*params.rstar_eq*(AA*EE - BB*DD) + EE*FF)
    a = (params.rstar_eq*(params.rstar_eq*AA - DD) + FF)*(params.rstar_eq*(params.rstar_eq*AA + DD) + FF)
    
    e = np.full((H2.size, 1), e)
    
    cy = np.column_stack((a, b, c, d, e))

    return(cy, AA, BB, CC, DD, EE, FF, H2_TR, K2_TR)

def qrt_solve(cy, params):
    
    py = [0] * 5
    r  = [[0.0,0.0,0.0,0.0,0.0],[0.0,0.0,0.0,0.0,0.0],[0.0,0.0,0.0,0.0,0.0]]
    # cy = cy[::-1] # needed because this is how the C-style code was written    
    # print("cy: {}".format(cy))
    #Once the coefficents for the quartic equation in y are known, the roots of the quartic polynomial will represent y-values of the intersection points of the two ellipse curves.
    #The quartic sometimes degenerates into a polynomial of lesser degree, so handle all possible cases.
    
    if(abs(cy[4]) > 0.0): # if not zero

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

        #Completely degenerates quartic -> ellipses identical?
        #A completely degenerate quartic, which would seem to indicate that the ellipses are identical. However, some configurations lead to a degenerate quartic with no points of intersection.
        nroots = 0
    

    #Determine which roots are real; discard any complex roots
    nychk = 0
    ychk = [0] * (nroots+1)
    for i in range(1,nroots+1):
        # print(r[2][i])
        # print(abs(r[2][i]))
        if(abs(r[2][i])<(10**(-7))):
            # print("winner {}".format(r[2][i]))
            nychk = nychk+1
            ychk[nychk] = r[1][i]*params.rstar_pol
    
    #Sort the real roots by straight insertion
    for j in range(2,nychk+1):
        tmp0 = ychk[j]
        for k in range(j-1,0,-1):
            if(ychk[k] <= tmp0):
                break
            ychk[k+1] = ychk[k]
        ychk[k+1] = tmp0
    
    return(nychk, ychk)  

# def circ_solve(params.rstar_eq, params.rpol, H2, K2):
#     dcrim = -(H2*H2) * (params.rstar_eq*params.rstar_eq + ((-(params.rpol*params.rpol) + H2*H2 + K2*K2))**2 - 2 * params.rstar_eq * (params.rpol*params.rpol + H2*H2 + K2*K2))
#     dmr = 2 * (H2*H2 + K2*K2)
#     numr = K2 * (params.rstar_eq - params.rpol*params.rpol + H2*H2 + K2*K2)

#     sols = []
#     sol1 = (numr - np.emath.sqrt(dcrim))/dmr
#     sol2 = (numr + np.emath.sqrt(dcrim))/dmr

#     if np.isreal(sol1) == True:
#         sols.append(sol1)
#         if np.isreal(sol2) == True:
#             sols.append(sol2)
    
#     return(np.array(sols))

# def newton_qrt_solve(cy, circ_roots):
    cy = cy[::-1] # needed because this is how the C-style code was written    

    deriv1 = np.polyder(cy, m=1) # gets coefficients of above function for derivatives
    # deriv2 = np.polyder(cy, m=2)
    
    def qrt(y): # ellipse quartic function
        return(cy[0] * y**4 + cy[1] * y**3 + cy[2] * y*y + cy[3]*y + cy[4])

    def qrt_jac(y): # derivative of ellipse quartic function
        return(deriv1[0] * y**3 + deriv1[1] * y*y + deriv1[2]*y + deriv1[3])

    # def qrt_jac2(y):
    #     return(deriv2[0] * y*y + deriv2[1] * y + deriv2[2])

    # solve for roots
    sol = newton(qrt, circ_roots, fprime = qrt_jac)#, fprime2 = qrt_jac2)
    ychk = sol
    ychk = np.sort(ychk)
    nychk = len(ychk)

    # not sure exactly why this is needed
    ychk = np.insert(ychk, [0], 0)
    ychk = np.insert(ychk, [3], [0, 1])
    
    # fig,ax = plt.subplots(figsize = (5,5))
    # ax.plot(xs1, ys1, color='g', lw=0.1)
    # ax.plot(xs2, ys2, color='b', lw=0.1)
    # plt.show()

    return(ychk, nychk)

def QUADROOTS(p,r):
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
    return((AA*x*x) + (BB*x*y) + (CC*y*y) + (DD*x) + (EE*y) + (FF))

def nointpts(params,H2_TR,K2_TR,FF):
    #The relative size of the two ellipses can be found from the axis lengths.
    relsize = (params.rstar_eq*params.rstar_pol) - (params.req*params.rpol)
    if(relsize>0.0):
        #First ellipse is larger than second ellipse.
        #If the second ellipse center(H2_TR,K2_TR) is inside the first ellipse, then the second ellipse is completely inside the first ellipse; otherwise, the ellipses are disjoint.
        if(((H2_TR*H2_TR)/(params.rstar_eq*params.rstar_eq)+(K2_TR*K2_TR)/(params.rstar_pol*params.rstar_pol))<1.0):
            #Ellipse 2 is inside ellipse 1.
            return (math.pi*params.req*params.rpol)
        else:
            #print("Disjoint ellipses")
            #Disjoint ellipses.
            return 0.0
    elif(relsize<0.0):
        #Second ellipse is larger than first ellipse.
        #If the first ellipse center (0,0) is inside the second ellipse, then ellipse 1 is completely inside ellipse 2. Otherwise, the ellipses are disjoint.
        if(FF<0.0):
            #Ellipse 1 inside ellipse 2.
            return (math.pi*params.rstar_eq*params.rstar_pol)
        else:
            #Disjoint ellipses.
            return 0.0
    else:
        #If execution arrives here, the relative sizes are identical.
        #Check the parameters to see whether the two ellipses are identical.
        #Ellipses are identical.
        return (math.pi*params.rstar_eq*params.rstar_pol)

def twointpts(x,y,params,H2_TR,K2_TR,AA,BB,CC,DD,EE,FF):
    #If execution arrives here, the intersection points are not tangents.
    #Determine which direction to integrate in ellipse_segment() routine for each ellipse.
    #Find the parametric angles for each point on ellipse 1.

    # x intersection point is bounded by params.rstar_eq=[-1, 1]
    if(abs(x[1])>params.rstar_eq): # useless?
        
        if(x[1]<0.0):
            x[1] = -params.rstar_eq
        else:
            x[1] = params.rstar_eq

    if(abs(x[2])>params.rstar_eq): # useless?
        print("yas")
        if(x[2]<0.0):
            x[2] = -params.rstar_eq
        else:
            x[2] = params.rstar_eq

    # calculate angle if yint in Q3 or Q4 
    if(y[1]<0.0):
        theta1 = (2.0*math.pi) - math.acos(x[1]/params.rstar_eq)
    else:
        theta1 = math.acos(x[1]/params.rstar_eq)
    if(y[2]<0.0): #Quadrant III or IV
        theta2 = (2.0*math.pi) - math.acos(x[2]/params.rstar_eq)
    else:
        theta2 = math.acos(x[2]/params.rstar_eq)

    
    #Logic is for proceeding counterclockwise from theta1 to theta2.
    if(theta1>theta2):
        tmp    = theta1
        theta1 = theta2
        theta2 = tmp
        
    # midpoint between intersection points, on stellar boundary
    xmid = params.rstar_eq*math.cos((theta1+theta2)/2.0)
    ymid = params.rstar_pol*math.sin((theta1+theta2)/2.0)
    
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
                    
    area1 = 0.5*(params.rstar_eq*params.rstar_pol*(theta2-theta1) + trsign*abs(x[1]*y[2]-x[2]*y[1]))
    
    if(area1<0):
        area1 = area1+(params.rstar_eq*params.rstar_pol)
    
    #Find ellipse 2 segment area.
    #The ellipse segment routine needs an ellipse that is centered at the origin and oriented with the coordinate axes. The intersection point (x[1],y[1]) and (x[2],y[2]) are found with both ellipses translated and rotated by (-params.pos_star[0],-params.pos_star[1]) and -params.phi_s. Further, translate and rotate the points to put the second ellipse at the origin and oriented with the coordinate axes. The translation is (-H2_TR,-K2_TR) and the rotation is -(params.phi_p-params.phi_s).
    cosphi = math.cos(params.phi_s-params.phi_p)
    sinphi = math.sin(params.phi_s-params.phi_p)
    
    # translated intersection points to center around origin
    x1_tr = (x[1]-H2_TR)*cosphi + (y[1]-K2_TR)*(-sinphi)
    y1_tr = (x[1]-H2_TR)*sinphi + (y[1]-K2_TR)*(cosphi)
    x2_tr = (x[2]-H2_TR)*cosphi + (y[2]-K2_TR)*(-sinphi)
    y2_tr = (x[2]-H2_TR)*sinphi + (y[2]-K2_TR)*(cosphi)

    
    # ax = viz(params, PHI, H2_TR, K2_TR)
    # ax.scatter(x[1:], y[1:], s = 20, zorder = 10, color = "black")
    # ax.scatter(H2_TR, K2_TR, s = 5) # this holds so long that stellar obliquity is 0
    # # ax.scatter([x1_tr, x2_tr], [y1_tr, y2_tr], s = 10, color = "purple") # this holds so long that stellar obliquity is 0
    # ra = [x[2], K2_TR]
    # center = np.array([H2_TR, K2_TR])
    # xtop = np.array([x[2], y[2]])
    # ax.scatter(x[2], K2_TR)
    
    # print(theta1)
    # print(theta2)
    
    # plt.show()

    #Determine which branch of the ellipse to integrate by finding a point of the second ellipse and checking whether it is inside the first ellipse (in their once translated+rotated positions).
    #Find the parametric angles for each point on ellipse 1.
    
    
    if(abs(x1_tr)>params.req): # also going to never be true
        
        if(x1_tr<0.0):
            x1_tr = -params.req 
        else:
            x1_tr = params.req
    if(abs(x2_tr)>params.req):
        
        if(x2_tr<0.0):
            x2_tr = -params.req
        else:
            x2_tr = params.req


    # calculate angle if yint in Q3 or Q4   
    if(y1_tr<0.0): #Quadrant III or IV
        theta1 = (2.0*math.pi)-math.acos(x1_tr/params.req)
    else: #Quadrant I or II
        theta1 = math.acos(x1_tr/params.req)
    
    if(y2_tr<0.0):
        theta2 = (2.0*math.pi)-math.acos(x2_tr/params.req)
    else: #Quadrant I or II
        theta2 = math.acos(x2_tr/params.req)
    
    #Logic for proceeding counterclockwise from theta1 to theta2.
    if(theta1>theta2): # ensures theta2 > theta1
        tmp    = theta1
        theta1 = theta2
        theta2 = tmp
    
    # theta1 and theta2 are angles from ellipse center to intersection points, theta1 < theta2 ALWAYS
    thetas = [theta1, theta2]

    #Find a point on the second ellipse that is different from the two intersection points.
    xmid = params.req*math.cos((theta1+theta2)/2.0)
    ymid = params.rpol*math.sin((theta1+theta2)/2.0)
    
    
    #Translate the point back to the second ellipse in its once translated+rotated position.
    cosphi  = math.cos(params.phi_p-params.phi_s)
    sinphi  = math.sin(params.phi_p-params.phi_s)
    
    xmid_rt = xmid*cosphi + ymid*(-sinphi) + H2_TR
    ymid_rt = xmid*sinphi + ymid*(cosphi)  + K2_TR 
    
    #The point (xmid_rt,ymid_rt) is on the second ellipse 'between' the intersection points (x[1],y[1]) and (x[2],y[2]) when traveling counterclockwise from (x[1],y[1]) to (x[2],y[2]). If the point (xmid_rt,ymid_rt) is inside the first ellipse, then the desired segment of ellipse 2 contains the point (xmid_rt,ymid_rt), so integrate counterclockwise from (x[2],y[2]) to (x[1],y[1]).
    if(((xmid_rt*xmid_rt)/(params.rstar_eq*params.rstar_eq) + (ymid_rt*ymid_rt)/(params.rstar_pol*params.rstar_pol))>1.0):
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
    
    area2 = 0.5*(params.req*params.rpol*(theta2-theta1)+trsign*abs(x1_tr*y2_tr - x2_tr*y1_tr))
    
    if(area2<0.0):
        area2 = area2+(params.req*params.rpol)

    #Two intersection points

    return (area1+area2, thetas)

def verify_roots(nychk, ychk, params, AA, BB, CC, DD, EE, FF):

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
        if(abs(ychk[i])>params.rstar_pol):
            x1 = 0.0
        else:
            x1 = params.rstar_eq*((1.0-(ychk[i]*ychk[i])/(params.rstar_pol*params.rstar_pol))**0.5)
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
