# -*- coding: utf-8 -*-
"""
Created on Fri Nov 11 18:00:49 2016

@author: admin
"""

import numpy as np
from matplotlib import pyplot as pl
from glob import glob
from scipy.optimize import curve_fit
from scipy.interpolate import UnivariateSpline
from matplotlib import rc,cm
from useful_stuff.useful import *
from matplotlib import rc_params as params


rc('text', usetex=True)
rc('font', family='serif',serif='Computer Modern Roman')
rc('text.latex', unicode=True)
rc('text.latex', preamble='\usepackage{siunitx}')
rc('lines',linewidth = 1.0)


d0=np.genfromtxt('161024cernox-R07-C12.txt',delimiter='\t',skiprows=1)
d1,h1=read_ppms_RT('161109_RT_sensor-calib.dat')

d1_d,d1_u=Temp_sep(d1)

fig1=pl.figure(1,figsize=(10,6))
ax1=pl.axes([0.15,0.15,0.95-0.15,0.95-0.15])
ax1.plot(d1_d[:,3],d1_d[:,21],'hb',label='161109')
ax1.plot(d1_u[:,3],d1_u[:,21],'hg')
ax1.plot(d0[:,0],d0[:,1],'r-',linewidth=1.5,label='previous calibration (161024)')
ax1.set_xlabel('Temperature (K)')
ax1.set_ylabel(r'Resistance (\si{\ohm})')

ax1.set_xlim(1.5,310)
ax1.set_ylim(30,700)
ax1.set_yscale('log')
ax1.set_xscale('log')

ax1.grid(True, which='both')
ax1.legend()