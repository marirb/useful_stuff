# -*- coding: utf-8 -*-
"""
Created on Mon Jan 25 11:47:40 2016

@author: admin
"""

import numpy as np
from scipy.interpolate import UnivariateSpline
from mpmath import re,pi,cos,sin,spherharm
from scipy.special import sph_harm


def read_xrd_sim(name):
    '''read data where columns are seperated by whitespace.\n no skiprow option implemented yet'''
    data=[]
    with open(name,'r') as f:
        lines=f.readlines()
        for line in lines:
            data.append(line.split())
        data=np.array(data,dtype=float)
    return data 


def read_xrd(name,normalize=False):
    '''**read xrd data** where columns are seperated by komma and whitespace.\n function looks for the keyword '[Data]' and starts reading data in the next line
        if normalize is True, highest peak gets normalized to 100'''
    data=[]
    with open(name,'r') as f:
        ind=10000000
        data=[]
        for i, line in enumerate(f):
            if line.find('[Data]')>=0:
                ind=i
            if i>ind+1:
                data.append(line.split(',')[:-1])
        data=np.array(data,dtype=float)
        if normalize:
            data[:,1]=100*data[:,1]/np.max(data[:,1])
        return data
        
        
def read_ppms_tto(name):
    '''reads in data from thermal transport meaasurement option of PPMS\n
    **input:** filename as string\n
    **output:** data nd-array and header containing the physical measure of every column'''
    data=[]
    with open(name,'r') as f:
        ind=10000000000
        for i, line in enumerate(f):
            if line.find('[Data]')>=0:
                ind=i
            elif i==ind+1:
                header=np.array(line.split(',')[:-1])
    ind=ind+2
    data=np.genfromtxt(name,delimiter=',',skip_header=ind,missing_values=('',))
    return data,header

    
def read_ppms_tto_raw(name):
    data11=[]
    data9=[]
    with open(name,'r') as f:
        ind=10000000000
        for i, line in enumerate(f):
            if line.find('[Data]')>=0:
                ind=i
            elif i==ind+1:
                header=np.array(line.split(',')[:-1])
            elif i>ind+5:
                a=line.split(',')
                a[-1]=a[-1][:-1]
                if len(a[0])==0:
                    for j,k in enumerate(a):
                        if len(k)==0:
                            a[j]='nan'
                        else:
                            pass
                    if len(a)==9:
                        data9.append(a)
                    elif len(a)==11:
                        data11.append(a)
    data9=np.array(data9,dtype=float)
    data11=np.array(data11,dtype=float)
    return data9,data11,header
    

def read_ppms_RT(name):
    data=[]
    with open(name,'r') as f:
        ind=10000000000
        for i, line in enumerate(f):
            if line.find('[Data]')>=0:
                ind=i
            elif i==ind+1:
                header=np.array(line.split(',')[:-1])
    ind=ind+2
    data=np.genfromtxt(name,delimiter=',',skip_header=ind,missing_values=('',))
    return data,header   
     
def read_mpms_MT(name,m_mol=796.612,mass=None,factor=3,cols=(3,4,2,0),NAN=False,save=False):
    '''read mpms data where columns are seperated by komma.\n functions looks for the keyword '[Data]' and starts reading data in the next line\n
    NAN: if true, read in rows containing NAN values (which is empty entry or entry starting with #) get deleted'''
    info={'NAME':'', 'WEIGHT':'', 'AREA':'', 'LENGTH':'', 'SHAPE':'', 'COMMENT':'', 'SEQUENCE FILE':'','FILEOPENTIME':''}
    with open(name,'r') as f:
        ind=10000000000
        for i, line in enumerate(f):
            if line.find('[Data]')>=0:
                ind=i
            elif i==ind+1:
                header=np.array(line.split(',')[:-1])[list(cols)]
            elif line.find('WEIGHT')>=0:
                info["WEIGHT"] = line[line.find("WEIGHT,")+8:-1]
            elif line.find('AREA')>=0:
                info["AREA"] = line[line.find("AREA,")+6:-1]
            elif line.find('LENGTH')>=0:
                info["LENGTH"] = line[line.find("LENGTH,")+8:-1]
            elif line.find('SHAPE')>=0:
                info["SHAPE"] = line[line.find("SHAPE,")+7:-1]
            elif line.find('COMMENT')>=0:
                info["COMMENT"] = line[line.find("COMMENT,")+9:-1]
            elif line.find('SEQUENCE FILE')>=0:
                info["SEQUENCE FILE"] = line[line.find("SEQUENCE FILE,")+22:-1]
            elif line.find('FILEOPENTIME')>=0:
                info["FILEOPENTIME"] = line.split(' ')[-3:]
        ind=ind+2
        if not mass:
            mass=float(info['WEIGHT'])/1000
        else:
            mass=mass/1000
        data=np.genfromtxt(name,delimiter=',',skip_header=ind,missing_values=('',),usecols=cols)
        l=len(data)
        B=data[0,2]
        m_emu_g=data[:,1]/mass
        m_emu_mol=data[:,1]*m_mol/mass
        chi_emu_mol=data[:,1]*m_mol/(mass*B)
        chi_emu_factor_mol=data[:,1]*m_mol/(mass*B*factor)
        one_over_chi=1/chi_emu_factor_mol
        data=np.hstack((data,m_emu_g.reshape(l,1),m_emu_mol.reshape(l,1),chi_emu_mol.reshape(l,1),chi_emu_factor_mol.reshape(l,1),one_over_chi.reshape(l,1)))
        print(str(mass)+' g',str(B)+' Oe',str(m_mol)+' g/mol')        
        if save:
            name=name[:-3]+'txt'                
            np.savetxt(name,data,delimiter='\t',header='Temperature [K]\t long magnetic moment [emu]\t magnetic field [Oe]\t timestamp [s]\t magnetic moment [emu/g]\t magnetic moment [emu/mol]\t Chi [emu/mol]\t Chi [emu/factor-mol]\t 1/chi [factor-mol/emu]')
        
        if NAN:
            data=data[~np.isnan(data).any(axis=1)]
        return data,header,info


def read_spectra(name):
    data=np.genfromtxt(name,skip_header=14)
    return data 


        
def emu2emu_per_mol(data,mass,B,m_mol=796.612,factor=3,cols=[1,]):
    '''converts magnetic moment [emu] into susceptability chi [emu/mol]. default values are molar mass of na4ir3o8 and factor=3 to get to [emu/Ir-mol]\n
    data: array\n
    mass: mass in mg of sample measured\n
    B: magnetic field in Oersted\n
    m_mol: na4ir308: 796.612 g/mol\n
    factor: to translate to mol of one participant (for 1 mol na4ir3o8 3 mol ir02 is needed: hence factor 3 for [emu/Ir-mol])\n
    cols: choose cols to convert. default column 2'''
    mass=mass/1000
    data[:,cols]=data[:,cols]*m_mol/(mass*factor*B)
    return data
    


def find_nearest(a, a0):
    "returns index of Element in nd array `a` closest to the scalar value `a0`"
    a=np.array(a)
    idx = np.abs(a - a0).argmin()
    return idx
    
    
def match_data_sets(x1,y1,x2,y2):
    '''**input:**  two data sets x,y; \t  (x_i, y_i must have same length).
        the two data sets can have different length but must have a union of x-values\n 
    **output:** one x-array with union range and the smaller stepwidth of the two original sets and the two y-data sets y1, y2'''
    x_min1, x_min2 = np.min(x1), np.min(x2)
    x_max1 ,x_max2 = np.max(x1), np.max(x2)
    x_min, x_max = np.max([x_min1,x_min2]), np.min([x_max1,x_max2])
    d1 ,d2 = (x_max1-x_min1)/(len(x1)+1), (x_max2-x_min2)/(len(x2)+1)
    d = np.min([d1,d2])
    n = (int((x_max-x_min)/d)+1)*2
    x = np.linspace(x_min,x_max,n)
    spl1, spl2 = UnivariateSpline(x1,y1,k=1,s=0), UnivariateSpline(x2,y2,k=1,s=0)
    y1, y2 = spl1(x), spl2(x)
    return x, y1, y2
    
    
def point_data_distance(x,y,x0,y0):
    '''calculates the shortest distance of a point (x0,y0) to a data curve (x,y)\n
    output: distance d'''
    x,y=np.array(x),np.array(y)
    d=np.min(np.sqrt((x-x0)**2+(y-y0)**2))
    return d
    

def Gauss_1(x,a,b,c,d):
    '''FWHM = 2*sqrt(2*ln(2))*c = 2.35482*c'''
    return a*np.exp(-(x-b)**2/(2*c))+d

def Gauss_2(x,a1,b1,c1,a2,b2,c2,d):
    '''FWHM = 2*sqrt(2*ln(2))*c = 2.35482*c'''
    return a1*np.exp(-(x-b1)**2/(2*c1))+a2*np.exp(-(x-b2)**2/(2*c2))+d
        
def Lorentz_1(x,a,b,c,d):
    '''FWHM = 2*c'''
    return a*c/((x-b)**2+c**2)+d
    
def Lorentz_2(x,a1,b1,c1,a2,b2,c2,d):
    '''FWHM = 2*c'''
    return a1*c1/((x-b1)**2+c1**2)+a2*c2/((x-b2)**2+c2**2)+d
    

def Temp_sep(data,col=3,turn=1):
    '''seperates the cooling up and cooling down curve of a temperaure sweep.\n
    col: is the column of temperature\n
    turn=1: turningpoint is at lowest temperature) or\n
    turn=2: turning point is at highest temperature\n
    output: two data sets up and down (same order as during measurement)'''
    Tmin=np.min(data[:,col])
    imin=np.argmin(data[:,col])
    Tmax=np.max(data[:,col])
    imax=np.argmax(data[:,col])
    if turn==1:
        if (data[imin+1,col]-Tmin)<(data[imin-1,col]-Tmin):
            down=data[:imin+1,:]
            up=data[imin+1:,:]
        else:
            down=data[:imin,:]
            up=data[imin:,:] 
    elif turn==2:
        if (data[imax+1,col]-Tmax)<(data[imax-1,col]-Tmax):
            down=data[:imax+1,:]
            up=data[imax+1:,:]
        else:
            down=data[:imax,:]
            up=data[imax:,:] 
    else:
        print('unvalid entry for variable "turn"')
    
    return down,up


'''################################################################################################################################'''   
 
'''spherical harmonics, p-orbitals and d-orbitals'''

def Y_lm(l,m):
    '''returns absolute value of real part of spherical harmonics so that it can be plotted as \n
    mpmath.splot(Y_lm(l,m), [0,pi], [0,2*pi], points=100, keep_aspect=True, axes=False)'''
    def g(theta,phi):
        R = abs(re(spherharm(l,m,theta,phi)))
        x = R*cos(phi)*sin(theta)
        y = R*sin(phi)*sin(theta)
        z = R*cos(theta)
        return [x,y,z]
    return g
    
    
def combine_Ylm(l,m1,m2,a,b):
    def g(theta,phi):
        R = abs(re(a*spherharm(l,m1,theta,phi)+b*spherharm(l,m2,theta,phi)))**2
        x = R*cos(phi)*sin(theta)
        y = R*sin(phi)*sin(theta)
        z = R*cos(theta)
        return [x,y,z]
    return g

def p_z():
    '''plot with mpmath.splot(p_z(), [0,pi], [0,2*pi])'''
    return Y_lm(1,0)
    
def p_x():
    '''plot with mpmath.splot(p_z(), [0,pi], [0,2*pi])'''
    return combine_Ylm(1,-1,1,1,-1)

def p_y():
    '''plot with mpmath.splot(p_z(), [0,pi], [0,2*pi])'''
    return combine_Ylm(1,-1,1,1j,1j)

def d_xy():
    '''plot with mpmath.splot(p_z(), [0,pi], [0,2*pi])'''
    return combine_Ylm(2,-2,2,1j,-1j)

def d_xz():
    '''plot with mpmath.splot(p_z(), [0,pi], [0,2*pi])'''
    return combine_Ylm(2,-1,1,1,-1)
    
def d_yz():
    '''plot with mpmath.splot(p_z(), [0,pi], [0,2*pi])'''
    return combine_Ylm(2,-1,1,1j,1j)
    
def d_x2y2():
    return combine_Ylm(2,-2,2,1,1)
    
def d_z2():
    '''plot with mpmath.splot(p_z(), [0,pi], [0,2*pi])'''
    return Y_lm(2,0)

    
'''################################################################################################################################'''

'''numerical spherical harmonics (numpy/scipy)'''    

def Y_lm_num(l,m,phimax=np.pi,thetamax=2*np.pi,num_points=100):    
    '''returns numerical spherical harmonics as [x,y,z],s where x,y,z are cartesian coordinates and s is realvalue of spherical harmonic or orbital\n
    plot in the following way\n\n
    
    Y,s=Y_lm_num(1,1)\n
    colormap = cm.ScalarMappable( cmap=pl.get_cmap("cool"))
    
    fig = pl.figure(1,figsize=(10,10))\n 
    ax = fig.gca(projection='3d')\n
    surf = ax.plot_surface(*Y, rstride=2, cstride=2, facecolors=colormap.to_rgba(s), \n
                           linewidth=0.5, antialiased=False)\n
    
    lim=0.15 \n
    ax.set_xlabel('x') \n
    ax.set_ylabel('y') \n
    ax.set_zlabel('z') \n
    ax.set_xlim(-lim,lim) \n
    ax.set_ylim(-lim,lim) \n
    ax.set_zlim(-lim,lim) \n
    ax.set_aspect("equal")'''
    
    phi = np.linspace(0, phimax, num_points)
    theta = np.linspace(0, thetamax, num_points)
    phi, theta = np.meshgrid(phi, theta)
    s=np.real(sph_harm(m,l,theta,phi))
    r=np.abs(s)**2
    x = r*np.sin(phi) * np.cos(theta)
    y = r*np.sin(phi) * np.sin(theta)
    z = r*np.cos(phi)
    return [x,y,z],s
    
   
def combine_Ylm_num(l,m1,m2,a,b,phimax=np.pi,thetamax=2*np.pi,num_points=100):    
    '''returns numerical spherical harmonics as [x,y,z],s where x,y,z are cartesian coordinates and s is realvalue of spherical harmonic or orbital\n
    plot in the following way\n\n
    
    Y,s=Y_lm_num(1,1)\n
    colormap = cm.ScalarMappable( cmap=pl.get_cmap("cool"))
    
    fig = pl.figure(1,figsize=(10,10))\n 
    ax = fig.gca(projection='3d')\n
    surf = ax.plot_surface(*Y, rstride=2, cstride=2, facecolors=colormap.to_rgba(s), \n
                           linewidth=0.5, antialiased=False)\n
    
    lim=0.15 \n
    ax.set_xlabel('x') \n
    ax.set_ylabel('y') \n
    ax.set_zlabel('z') \n
    ax.set_xlim(-lim,lim) \n
    ax.set_ylim(-lim,lim) \n
    ax.set_zlim(-lim,lim) \n
    ax.set_aspect("equal")'''
    
    phi = np.linspace(0, phimax, num_points)
    theta = np.linspace(0, thetamax, num_points)
    phi, theta = np.meshgrid(phi, theta)
    s=np.real(a*sph_harm(m1,l,theta,phi)+b*sph_harm(m2,l,theta,phi))
    r=np.abs(s)**2
    x = r*np.sin(phi) * np.cos(theta)
    y = r*np.sin(phi) * np.sin(theta)
    z = r*np.cos(phi)
    return [x,y,z],s

def p_z_num():
    '''plot with mpmath.splot(p_z(), [0,pi], [0,2*pi])'''
    return Y_lm_num(1,0)
    
def p_x_num():
    '''plot with mpmath.splot(p_z(), [0,pi], [0,2*pi])'''
    return combine_Ylm_num(1,-1,1,1,-1)

def p_y_num():
    '''plot with mpmath.splot(p_z(), [0,pi], [0,2*pi])'''
    return combine_Ylm_num(1,-1,1,1j,1j)

def d_xy_num():
    '''plot with mpmath.splot(p_z(), [0,pi], [0,2*pi])'''
    return combine_Ylm_num(2,-2,2,1j,-1j)

def d_xz_num():
    '''plot with mpmath.splot(p_z(), [0,pi], [0,2*pi])'''
    return combine_Ylm_num(2,-1,1,1,-1)
    
def d_yz_num():
    '''plot with mpmath.splot(p_z(), [0,pi], [0,2*pi])'''
    return combine_Ylm_num(2,-1,1,1j,1j)
    
def d_x2y2_num():
    return combine_Ylm_num(2,-2,2,1,1)
    
def d_z2_num():
    '''plot with mpmath.splot(p_z(), [0,pi], [0,2*pi])'''
    return Y_lm_num(2,0)
    
    
    
    
    
    
    
    
    
    
    
    
    
    

    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
