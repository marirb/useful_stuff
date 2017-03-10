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


#d0=np.genfromtxt('161024cernox-R07-C12.txt',delimiter='\t',skiprows=1)
d1,h1=read_ppms_RT('161117_RT_sensor-calibration.dat')
d2,h2=read_ppms_RT('161117_RT_sensor-calibration2.dat')
d3,h3=read_ppms_RT('161117_R-time-100K_sensor-calibration.dat')

#d4,h4=read_ppms_RT('161115-RT_cernox_c12_r07.dat')
#d5,h5=read_ppms_RT('161116-RT_cernox_c12_r07.dat')

t=np.linspace(0,350,2000)
r=np.linspace(30,700,2000)
spl=UnivariateSpline(d1[:,20],d1[:,3],k=3,s=0.01)

#data=np.array([d1[:,3],d1[:,20]])
#np.savetxt('161117-cernox-R07-C12.txt',data.T,delimiter='\t',header='Temperature (K)\t Resistance (Ohm)')

'''##############################################################################'''

fig1=pl.figure(1,figsize=(10,6))
ax1=pl.axes([0.15,0.15,0.95-0.15,0.95-0.15])
ax1.plot(d1[:,3],d1[:,20],'hb',label='161117')
#ax1.plot(d2[3],d2[20],'or')
ax1.plot(spl(r),r,'k-')
#ax1.plot(d4[:,3],d4[:,20],'hg',label='161115')
#ax1.plot(d5[:,3],d5[:,20],'hr',label='161116')
#ax1.plot(d0[:,0],d0[:,1],'r-',linewidth=1.5,label='previous calibration (161024)')
ax1.set_xlabel('Temperature (K)')
ax1.set_ylabel(r'Resistance (\si{\ohm})')

ax1.set_xlim(1.5,310)
ax1.set_ylim(30,700)
ax1.set_yscale('log')
ax1.set_xscale('log')

ax1.grid(True, which='both')
ax1.legend()
#pl.close()



'''##############################################################################'''

fig2=pl.figure(2,figsize=(10,6))
ax2=pl.axes([0.15,0.15,0.95-0.15,0.95-0.15])

ax2.plot(t,t,'k-')
ax2.plot(d1[:,3],spl(d1[:,20]),'hb',label='161117')
#ax2.plot(d5[:,3],spl(d5[:,20]),'hr',label='161116')
#ax2.plot(d0[:,0],d0[:,1],'r-',linewidth=1.5,label='previous calibration (161024)')
ax2.set_xlabel('System Temperature (K)')
ax2.set_ylabel(r'cernox temperature (K))')

#ax2.set_xlim(1.5,310)
#ax2.set_ylim(30,700)
#ax2.set_yscale('log')
#ax2.set_xscale('log')

ax2.grid(True, which='both')
ax2.legend()
#pl.close()



'''##############################################################################'''

#fig3=pl.figure(3,figsize=(10,6))
#ax3=pl.axes([0.15,0.15,0.95-0.15,0.95-0.15])
#
#
#ax3.plot(d3[:,1]-d3[0,1],spl(d3[:,20]),'hb',label=r'161117-T_{sensor}')
#ax3.plot(d3[:,1]-d3[0,1],d3[:,3],'hr',label=r'161117-T_{sys}')
##ax2.plot(d0[:,0],d0[:,1],'r-',linewidth=1.5,label='previous calibration (161024)')
#ax3.set_xlabel('time')
#ax3.set_ylabel(r'Resistance (\si{\ohm})')
#
##ax3.set_xlim(1.5,310)
##ax3.set_ylim(99.9,101.2)
##ax3.set_yscale('log')
##ax3.set_xscale('log')
#
#ax3.grid(True, which='both')
#ax3.legend()



















