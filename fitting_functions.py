# -*- coding: utf-8 -*-
"""
Created on Sat Jan 19 23:19:32 2019

@author: FMQ3_1
"""

import numpy as np
from scipy.integrate import quad
from matplotlib import pyplot as pl
from useful import constants

#-------------------------------------------------------------------------------------------------------------------------------------------------------------
'''gaussian, lorentzian and voigt line shapes'''
#-------------------------------------------------------------------------------------------------------------------------------------------------------------
def gaussian(x,a,b,c,d):
    '''
    FWHM = c = 2*sqrt(2*ln(2))*s = 2.35482*s\n
    Area = a
    a/sqrt(2*pi*s**2)*np.exp(-(x-b)**2/(2*s**2))+d    
    '''
    s=c/(2*np.sqrt(2*np.log(2)))
    return a/np.sqrt(2*np.pi*s**2)*np.exp(-(x-b)**2/(2*s**2))+d

def double_gaussian(x, a1, b1, c1, a2, b2, c2, d):
    return gaussian(x, a1, b1, c1, d/2) + gaussian(x, a2, b2, c2, d/2)

def triple_gaussian(x, a1, b1, c1, a2, b2, c2, a3, b3, c3, d):
    return gaussian(x, a1, b1, c1, d/3) + gaussian(x, a2, b2, c2, d/3) +  gaussian(x, a3, b3, c3, d/3)

def quadruple_gaussian(x, a1, b1, c1, a2, b2, c2, a3, b3, c3, a4, b4, c4, d):
    return gaussian(x, a1, b1, c1, d/4) + gaussian(x, a2, b2, c2, d/4) + gaussian(x, a3, b3, c3, d/4) + gaussian(x, a4, b4, c4, d/4)

def lorentzian(x,a,b,c,d):
    '''FWHM = c'''
    return a/(np.pi)*(c/2.)/((x-b)**2+(c/2.)**2) + d

def voigt(x,a,b,c,d,e):
    s=c/(2*np.sqrt(2*np.log(2)))
    G = 1/np.sqrt(2*np.pi*s**2)*np.exp(-(x-b)**2/(2*s**2))
    L = 1/(np.pi)*(c/2.)/((x-b)**2+(c/2.)**2)
    return a*(e*L + (1-e)*G) + d

def double_voigt(x, a1, b1, c1, e1, a2, b2, c2, e2, d):
    return voigt(x, a1, b1, c1, e1, d/2) + voigt(x, a2, b2, c2, e2, d/2)

def asymmetric_gaussian(x,a,b,c1,c2,d):
    '''
    FWHM = c = 2*sqrt(2*ln(2))*s = 2.35482*s\n
    Area = a
    a/sqrt(2*pi*s**2)*np.exp(-(x-b)**2/(2*s**2))+d    
    '''
    s = (c1 + c2*(x-b)) / (2*np.sqrt(2*np.log(2)))
    return a/np.sqrt(2*np.pi*s**2)*np.exp(-(x-b)**2/(2*s**2))+d



#-------------------------------------------------------------------------------------------------------------------------------------------------------------
'''different fitting functions curie weiß and MPMS squid response, insulating behavior'''
#-------------------------------------------------------------------------------------------------------------------------------------------------------------
def linear_fit(x,a,b):
    '''linear curve'''
    return a*x + b

def insulator_fit(x,a,b):
    '''
    0.01*a*np.exp(0.001*constants.e*b/(constants.kB*x))\n
    b in meV
    '''
    return 0.01*a*np.exp(0.001*constants.e*b/(constants.kB*x))

def double_insulator_fit(x,a1,b1,a2,b2):
    '''
    0.01*a1*np.exp(0.001*constants.e*b1/(constants.kB*x)) + 0.01*a2*np.exp(0.001*constants.e*b2/(constants.kB*x))\n
    b1,b2  in meV
    '''
    return 0.01*a1*np.exp(0.001*constants.e*b1/(constants.kB*x))+0.01*a2*np.exp(0.001*constants.e*b2/(constants.kB*x))

def curie_weiss(x,C,T_CW,X0):
    '''**Curie-Weiss:** C/(T-T\ :sub:`CW`\ )+0.0001*X\ :sub:`0`\ \n
    **mu\ :sub:`eff`\ ** =sqrt(3*k\ :sub:`B`\ *C/N\ :sub:`A`\ )=2.8295*sqrt(C)*mu\ :sub:`B`\ '''
    return C/(x-T_CW)+0.0001*np.abs(X0)

def curie_weiss_2(x,a,b):
    '''without chi_0'''
    return a/(x-b)

def double_curie_weiss(x, C1, T_CW1, C2, T_CW2, X0):
    '''two curie weiss components as for example for powder with anisotropic susceptibility'''
    return 2./3*C1/(x-T_CW1) + 1./3*C2/(x-T_CW2) + 0.0001*np.abs(X0)

def MPMS_SQUID_response(x,a3,a4):
    '''SQUID response function for 5T MPMS (no linear correction term)
    '''
    R=0.97
    G=1.519    
    return a3*(2*(R**2+(x+a4)**2)**(-1.5)-(R**2+(G+(x+a4))**2)**(-1.5)-(R**2+(-G+(x+a4))**2)**(-1.5))

def MPMS_SQUID_response_lin_correction(x,a1,a2,a3,a4):
    '''SQUID response function with linear correction for 5T MPMS
    a1: slope
    a2: constant
    a3: moment
    a4: center position
    '''
    R=0.97
    G=1.519    
    return a2+a1*x+a3*(2*(R**2+(x+a4)**2)**(-1.5)-(R**2+(G+(x+a4))**2)**(-1.5)-(R**2+(-G+(x+a4))**2)**(-1.5))

def MPMS3_SQUID_response_lin_correction(x,a1,a2,a3,a4):
    '''SQUID response function with liner correction for MPMS3
    a1: slope
    a2: constant
    a3: moment
    a4: center position
    '''
    R=8.4455
    G=8.255  
    return a1+a2*x+a3*(2*(R**2+(x+a4)**2)**(-1.5)-(R**2+(G+(x+a4))**2)**(-1.5)-(R**2+(-G+(x+a4))**2)**(-1.5))


#-------------------------------------------------------------------------------------------------------------------------------------------------------------
'''NMR powder lineshapes'''
#-------------------------------------------------------------------------------------------------------------------------------------------------------------
def g(x, b, c):
    return gaussian(x,1,b,c,0)

def K(m):
    result = []
    for i in m:
        f = lambda x: 1/np.sqrt(1 - i*np.sin(x)**2) 
        result.append(quad(f, 0.0, np.pi/2)[0])
    return np.array(result)

def I(w, wa, wb, wc):
    wa,wb,wc = float(wa), float(wb), float(wc)
    f = (wb - wa) / (wc - wa)
    eps = (wc-wa)/1e3
    if (wb-wa)>eps and (wc-wb)>eps:
        w1 = wa+eps
        w2 = wb-eps
        w3 = wc-eps
        return np.piecewise(w, 
                            [(w <= wa-eps),
                            (wa-eps < w) & (w <= wa+eps),
                            (wa+eps < w) & (w < wb-eps),
                            (wb-eps <= w) & (w <= wb+eps),
                            (wb+eps < w) & (w < wc-eps),
                            (wc-eps <= w) & (w < wc+eps),
                            (wc+eps <= w)],
                            [0,
                             4/np.sqrt(f*(1-((w1 - wa) / (wc - wa)))) * K([(1-f)*((w1 - wa) / (wc - wa)) / (f*(1-((w1 - wa) / (wc - wa))))]),
                             lambda w: 4/np.sqrt(f*(1-((w - wa) / (wc - wa)))) * K((1-f)*((w - wa) / (wc - wa)) / (f*(1-((w - wa) / (wc - wa))))),
                             4/np.sqrt(f*(1-((w2 - wa) / (wc - wa)))) * K([(1-f)*((w2 - wa) / (wc - wa)) / (f*(1-((w2 - wa) / (wc - wa))))]),
                             lambda w: 4/np.sqrt(((w - wa) / (wc - wa))*(1-f)) * K((1-((w - wa) / (wc - wa)))*f / (((w - wa) / (wc - wa))*(1-f))),
                             4/np.sqrt(((w3 - wa) / (wc - wa))*(1-f)) * K([(1-((w3 - wa) / (wc - wa)))*f / (((w3 - wa) / (wc - wa))*(1-f))]),
                             0])
    elif (wb-wa)<=eps:
        w2 = wb+eps
        w3 = wc-eps
        return np.piecewise(w, 
                            [(w <= wa-eps),
                            (wa-eps < w) & (w <= wb+eps),
                            (wb+eps < w) & (w < wc-eps),
                            (wc-eps <= w) & (w < wc+eps),
                            (wc+eps <= w)],
                            [0,
                             4/np.sqrt(((w2 - wa) / (wc - wa))*(1-f)) * K([(1-((w2 - wa) / (wc - wa)))*f / (((w2 - wa) / (wc - wa))*(1-f))]),
                             lambda w: 4/np.sqrt(((w - wa) / (wc - wa))*(1-f)) * K((1-((w - wa) / (wc - wa)))*f / (((w - wa) / (wc - wa))*(1-f))),
                             4/np.sqrt(((w3 - wa) / (wc - wa))*(1-f)) * K([(1-((w3 - wa) / (wc - wa)))*f / (((w3 - wa) / (wc - wa))*(1-f))]),
                             0])
    elif (wc-wb)<=eps:
        w1 = wa+eps
        w2 = wb-eps
        return np.piecewise(w, 
                            [(w <= wa-eps),
                            (wa-eps < w) & (w <= wa+eps),
                            (wa+eps < w) & (w < wb-eps),
                            (wb-eps <= w) & (w <= wc+eps),
                            (wc+eps <= w)],
                            [0,
                             4/np.sqrt(f*(1-((w1 - wa) / (wc - wa)))) * K([(1-f)*((w1 - wa) / (wc - wa)) / (f*(1-((w1 - wa) / (wc - wa))))]),
                             lambda w: 4/np.sqrt(f*(1-((w - wa) / (wc - wa)))) * K((1-f)*((w - wa) / (wc - wa)) / (f*(1-((w - wa) / (wc - wa))))),
                             4/np.sqrt(f*(1-((w2 - wa) / (wc - wa)))) * K([(1-f)*((w2 - wa) / (wc - wa)) / (f*(1-((w2 - wa) / (wc - wa))))]),
                             0])
                     
def powder_peak(w, a, c, wa, wb, wc):
    b = np.mean(w)
    return a*1e-3*np.convolve(g(w,b,c), I(w, wa, wb, wc), mode='same')

'''powder peak with axial symmetry'''
def I_axial(w, wa, wc):
    if wc > wa:
        return np.piecewise(w, [(w <= wa),
                                (wa < w) & (w < wc),
                                (wc <= w)],
                                [0, 
                                 lambda w: 1 / (2*np.sqrt((w - wa) / (wc - wa))),
                                 0])
    elif wa > wc:
        return np.piecewise(w, [(w <= wc),
                            (wc < w) & (w < wa),
                            (wa <= w)],
                            [0, 
                             lambda w: 1 / (2*np.sqrt((w - wa) / (wc - wa))),
                             0])

def powder_peak_axial(w, a, c, wa, wc):
    b = np.mean(w)
    return a*np.convolve(g(w,b,c), I_axial(w, wa, wc), mode='same')



#-------------------------------------------------------------------------------------------------------------------------------------------------------------
'''plot multiple line shapes'''
#-------------------------------------------------------------------------------------------------------------------------------------------------------------
def plot_double_gaussian(x, popt, color_main='k', color_minor=['k','k'], minor=True, offset=0, ax=None):
    if ax==None:
        ax = pl.gca()
    ax.plot(x, double_gaussian(x, *popt)+ offset, c=color_main)
    if minor:
        ax.plot(x, gaussian(x, *popt[:3], d=popt[-1])+ offset, c=color_minor[0], lw=1)
        ax.plot(x, gaussian(x, *popt[3:6], d=popt[-1])+ offset, c=color_minor[1], lw=1)

def plot_triple_gaussian(x, popt, color_main='b', color_minor=['k','k','k'], minor=True, offset=0, ax=None):
    if ax==None:
        ax = pl.gca()
    ax.plot(x, triple_gaussian(x, *popt) + offset, c=color_main)
    if minor:
        ax.plot(x, gaussian(x, *popt[:3], d=popt[-1]) + offset, c=color_minor[0], lw=1)
        ax.plot(x, gaussian(x, *popt[3:6], d=popt[-1]) + offset, c=color_minor[1], lw=1)
        ax.plot(x, gaussian(x, *popt[6:9], d=popt[-1]) + offset, c=color_minor[2], lw=1)

def plot_quadruple_gaussian(x, popt):
    ax = pl.gca()
    ax.plot(x, quadruple_gaussian(x, *popt), c='k')
    ax.plot(x, gaussian(x, *popt[:3], d=popt[-1]), c='r', lw=1)
    ax.plot(x, gaussian(x, *popt[3:6], d=popt[-1]), c='g', lw=1)
    ax.plot(x, gaussian(x, *popt[6:9], d=popt[-1]), c='b', lw=1)
    ax.plot(x, gaussian(x, *popt[9:12], d=popt[-1]), c='c', lw=1)
