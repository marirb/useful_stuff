# -*- coding: utf-8 -*-
"""
Created on Mon Jan 25 11:47:40 2016

@author: admin
"""

import numpy as np
import pandas as pd
from scipy.interpolate import UnivariateSpline,interp1d
from mpmath import re,pi,cos,sin,spherharm
from scipy.special import sph_harm
from sympy.matrices import Matrix, eye, zeros, ones, diag
from scipy.optimize import curve_fit
from scipy.signal import savgol_filter
from sympy import I,sqrt
from matplotlib import rc,cm
from matplotlib import pyplot as pl
import re
from diamagnetic_correction import get_diamag_corr





import os
dir = os.path.dirname(__file__)
filename = os.path.join(dir, 'data','X91774.txt')
filename2 = os.path.join(dir, 'data','161117-cernox-R07-C12.txt')
filename3 = os.path.join(dir, 'data','Pt100-calibration.txt')
filename4 = os.path.join(dir, 'data','171023_x124321_calibration.txt')
filename_GE = os.path.join(dir, 'data','GEVarnish_1T.txt')
filename_quartz_rod = os.path.join(dir, 'data','180404_MoverH-quartz-rod.txt')
filename_brass_holder = os.path.join(dir, 'data','brass-holder','brass-holder-bkgr.txt')

filename_quartz_rod_1T = os.path.join(dir, 'data', 'quartz-holder', 'quartz-rod_1T.txt')
filename_quartz_rod_2T = os.path.join(dir, 'data', 'quartz-holder', 'quartz-rod_2T.txt')
filename_quartz_rod_3T = os.path.join(dir, 'data', 'quartz-holder', 'quartz-rod_3T.txt')
filename_quartz_rod_4T = os.path.join(dir, 'data', 'quartz-holder', 'quartz-rod_4T.txt')
filename_quartz_rod_5T = os.path.join(dir, 'data', 'quartz-holder', 'quartz-rod_5T.txt')
filename_quartz_rod_6T = os.path.join(dir, 'data', 'quartz-holder', 'quartz-rod_6T.txt')
filename_quartz_rod_7T = os.path.join(dir, 'data', 'quartz-holder', 'quartz-rod_7T.txt')



'''constants'''
class constants:
    kB=1.380648*10**(-23)
    kB_gauss = 1.38064853e-16
    e=1.602177*10**(-19)
    eps0=8.854187817*10**(-12)
    h=6.626070*10**(-34)
    hbar=1.054571*10**(-34)
    mu_bohr=9.274010*10**(-24)
    mu_bohr_gauss = 9.274009995e-21
    mu_0=4*np.pi*10**(-7)
    R=8.314460
    NA=6.022141*10**23
    C_to_mueff=2.8295



#####################################################################################################################################################

'''Puk sensor calibration spline function (of pressure cell puk)'''
T_puk_calibration_data=np.genfromtxt(filename,skip_header=3,usecols=(0,1)) 

'''bare chip cernox calibration spline function'''
cernox_R07_C12_calibration_data=np.genfromtxt(filename2,skip_header=1) 
def cernox_R2Temp(x):
    '''
    **input:** Resistance of cernox R07-C12
    **output:** Temperature value according to calibration file '161117-cernoc-R07-C12'
    '''
    spl=UnivariateSpline(cernox_R07_C12_calibration_data[:,1],cernox_R07_C12_calibration_data[:,0],k=3,s=0.01)
    return spl(x)


'''Pt100 calibration spline function'''
pt100_calibration_data=np.genfromtxt(filename3,delimiter='\t')
def pt100_R2Temp(x):
    '''
    **input:** Resistance of Pt100
    **output:** Temperature value according to calibration file '160911_RT_sensor-calibr_Pt100_CePt3P.dat'
    '''
    spl=UnivariateSpline(pt100_calibration_data[:,1],pt100_calibration_data[:,0],k=3,s=0.1)
    return spl(x)


'''cernox x124321 calibration spline function'''
cernox_x124321_calibration_data=np.genfromtxt(filename4,skip_header=1) 
def x124321_R2Temp(x):
    '''
    **input:** Resistance of cernox x124321
    **output:** Temperature value according to calibration file '171023_RT_calibration_CX_X124321_best-points.dat'
    '''
    R,T=cernox_x124321_calibration_data[:,1],cernox_x124321_calibration_data[:,0]
    Z=np.log10(R)
     
    spl=UnivariateSpline(Z,T,k=3,s=0.001)
    return spl(np.log10(x))



'''spline to GEVarnish data (taken at 1T)'''
MT_Ge = np.genfromtxt(filename_GE, delimiter='\t')
def GEVarnish_MT(x):
    '''
    **input:** temperature
    **output:** magnetic moment of GEVarnish
    '''
    spl=UnivariateSpline(MT_Ge[:,0],MT_Ge[:,1], k=3, s=2.5e-12)
    return spl(x)


'''spline to empty quartz rod data (taken at different fields)'''
def quartz_rod(x,field):
    '''
    **input:** temperature and magnetilx field value
    **output:** M/H of quartz rod: multiply by field in Oe
    '''
    if field < 15000:
        T_, M_ = np.genfromtxt(filename_quartz_rod_1T, delimiter='\t', unpack =True)
        savgol1 = savgol_filter(M_,15,3)
        spl = UnivariateSpline(T_[:-300], 1e6*savgol1[:-300],k=3, s=3.7e-3, ext=3)
        B=10000
    if field < 25000:
        T_, M_ = np.genfromtxt(filename_quartz_rod_2T, delimiter='\t', unpack =True)
        savgol2 = savgol_filter(M_,51,3)
        spl = UnivariateSpline(T_, 1e6*savgol2,k=3, s=1e-2, ext=3)
        B=20000
    if field < 35000:
        T_, M_ = np.genfromtxt(filename_quartz_rod_3T, delimiter='\t', unpack =True)
        savgol3 = savgol_filter(M_,41,3)
        print T_[150:-1][::-6]
        spl = UnivariateSpline(T_[:-150], 1e6*savgol3[:-150],k=3, s=3e-2, ext=3)
        B=30000
    if field < 45000:
        T_, M_ = np.genfromtxt(filename_quartz_rod_4T, delimiter='\t', unpack =True)
        savgol4 = savgol_filter(M_,91,3)
        spl = UnivariateSpline(T_[:-100], 1e6*savgol4[:-100],k=2, s=2e-2, ext=3)
        B=40000
    if field < 55000:
        T_, M_ = np.genfromtxt(filename_quartz_rod_5T, delimiter='\t', unpack =True)
        savgol5 = savgol_filter(M_,21,3)
        spl = UnivariateSpline(T_[:-250], 1e6*savgol5[:-250],k=3, s=9.0e-2, ext=3)
        B=50000
    if field < 65000:
        T_, M_ = np.genfromtxt(filename_quartz_rod_6T, delimiter='\t', unpack =True)
        savgol6 = savgol_filter(M_,51,3)
        spl = UnivariateSpline(T_[:-260], 1e6*savgol6[:-260],k=3, s=5e-3, ext=3)
        B=60000
    if field < 75000:
        T_, M_ = np.genfromtxt(filename_quartz_rod_7T, delimiter='\t', unpack =True)
        savgol7 = savgol_filter(M_,11,3)
        spl = UnivariateSpline(T_[:-250], 1e6*savgol7[:-250],k=3, s=8e-2, ext=3)
        B=70000
    else:
        pass
    return 1e-6*spl(x) / B * field





'''spline to empty brass rod with powder container'''
MT_brass = np.genfromtxt(filename_brass_holder, delimiter='\t')
def brass_rod_01T(x):
    '''
    **input:** temperature
    **output:** M/H of brass rod: multiply by field in Oe
    '''
    spl=UnivariateSpline(MT_brass[:,0],MT_brass[:,1], k=3, s=0, ext=3)
    return spl(x)
def brass_rod_5T(x):
    '''
    **input:** temperature
    **output:** M/H of brass rod: multiply by field in Oe
    '''
    spl=UnivariateSpline(MT_brass[:,0],MT_brass[:,2], k=3, s=0, ext=3)
    return spl(x)


#####################################################################################################################################################
'''XRAY'''


def read_tsv(name):
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


#####################################################################################################################################################
'''PPMS'''
        
        
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


    

def read_ppms_RT(name, cell=False, abs=True, sortB=False, field_unit='Oe'):
    '''
    - **index 1:**\t time stamp (s)\n
    - **index 3:**\t system temperature (K)\n
    - **index 4:**\t magnetic field (Oe)\n
    - **index 6,8,10,12:**\t Resistivity Ch1, Ch2, Ch3, Ch4 (units depend on data file settings during measurement. check header)\n
    - **index 19,20,21,22:**\t Resistance Ch1, Ch2, Ch3, Ch4 (Ohms)
    
    **cell:** if True, resistance values from Puk sensor in Channel 1 are converted into temperature using calibration file 'X91774.txt' in package folder  
    '''
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
    if abs:
        data[:,19:23]=np.abs(data[:,19:23])
        data[:,[6,8,10,12]]=np.abs(data[:,[6,8,10,12]])
    else:
        pass
    if cell:
        spl=UnivariateSpline(T_puk_calibration_data[::-1,1],T_puk_calibration_data[::-1,0],k=3,s=0.01)
        T=spl(data[:,19]).reshape(-1,1)
        data=np.hstack((data,T))
    else:
        pass
    if sortB:
        s=np.mean(np.diff(data[:,4]))
        if s<0:
            data=data[::-1,:]
        else:
            pass
    if field_unit == 'T':
        data[:,4] = data[:,4]/10000
    elif field_unit == 'Oe':
        pass
    else:
        pass
    return data,header   




def read_ppms_red_HC(name, m_mol, mass=None, factor=1, del_corrupt_data=False, threshold=100, del_points=False, sortT=False):
    '''
    assumes that HC is measured in units µJ/K \n
    - **index 1:**\t time stamp (s)\n
    - **index 4:**\t system temperature (K)\n
    - **index 5:**\t magnetic field (Oe)\n
    - **index 7:**\t sample temperature\n
    - **index 9:**\t Sample HC
    - **index -1:**\t Molar specific heat in J/(K*mol) if sample Hc was measured in standard µJ/K
    - index 15:\t fit deviation (Chi square)
    - index 11:\t Addenda HC (usually µJ/K)
    
    **m_mol** (g/mol)   **Na4Ir3O8:** 796.612    |    **Cd2Ru2O7:** 538.953\n
    
    '''
    info={'NAME':'', 'WEIGHT':'', 'AREA':'', 'LENGTH':'', 'SHAPE':'', 'COMMENT':'', 'SEQUENCE FILE':'','FILEOPENTIME':''}
    with open(name,'r') as f:
        ind=10000000000
        for i, line in enumerate(f):
            if line.find('[Data]')>=0:
                ind=i
            elif i==ind+1:
                header=np.array(line.split(',')[:-1])
            elif line.find(',MASS:')>=0:
                info["WEIGHT"] = line[line.find(",INFO")+6:line.find(",MASS")]
    ind=ind+2
    if not mass:
        mass=float(info['WEIGHT'])/1000.
    else:
        mass=mass/1000.
    data=np.genfromtxt(name, delimiter=',', skip_header=ind, missing_values=('',))
    if sortT:
        data = sort_after_column(data, col=7)
    if del_points:
        data = np.delete(data, list(range(0, data.shape[0], 3)), axis=0)
        data = np.delete(data, list(range(0, data.shape[0], 2)), axis=0)
    else:
        pass
    if del_corrupt_data:
            data=data[data[:,15]<threshold,:]
    else:
        pass
    l=len(data)
    C_mol=data[:,9]*m_mol/(mass*factor)/1e6
    data=np.hstack((data,C_mol.reshape(l,1)))
    
    print 'mass = {:.2f} mg'.format(mass*1000.)
    return data,header   


def read_ppms_blue_HC(name, m_mol, mass=None, factor=1, del_corrupt_data=False, threshold=100):
    '''
    assumes that HC is measured in units µJ/K \n
    - **index 1:**\t time stamp (s)\n
    - **index 3:**\t system temperature (K)\n
    - **index 4:**\t magnetic field (Oe)\n
    - **index 6:**\t sample temperature\n
    - **index 8:**\t Sample HC
    - **index -1:**\t Molar specific heat J/(K*mol) (total mol, not mol per magnetic ion)
    
    **m_mol** (g/mol)   **Na4Ir3O8:** 796.612    |    **Cd2Ru2O7:** 538.953\n
    
    '''
    info={'NAME':'', 'WEIGHT':'', 'AREA':'', 'LENGTH':'', 'SHAPE':'', 'COMMENT':'', 'SEQUENCE FILE':'','FILEOPENTIME':''}
    with open(name,'r') as f:
        ind=10000000000
        for i, line in enumerate(f):
            if line.find('[Data]')>=0:
                ind=i
            elif i==ind+1:
                header=np.array(line.split(',')[:-1])
            elif line.find(',MASS:')>=0:
                info["WEIGHT"] = line[line.find(",INFO")+6:line.find(",MASS")]
    ind=ind+2
    if not mass:
        mass=float(info['WEIGHT'])/1000
    else:
        mass=mass/1000
    data=np.genfromtxt(name, delimiter=',', skip_header=ind, missing_values=('',))
    if del_corrupt_data:
            data=data[data[:,14]<threshold,:]
    else:
        pass
    l=len(data)
    C_mol=data[:,8]*m_mol/(mass*factor)
    data=np.hstack((data,C_mol.reshape(l,1)))
    
    print 'mass = {:.2f} mg'.format(mass*1000)
    return data,header


def read_ppms_VSM(name,m_mol=796.612,mass=None,factor=2,cols=(1,2,3,4,5),NAN=False):
    '''
    read mpms M(T) .dat file (columns are seperated by komma). functions looks for the keyword '[Data]' and starts reading data in the next line.\n
    **returns:** data,header,info\n
    **columns in data:**\n
    * **ind 0:** time (s)
    * **ind 1:** temperature (K)
    * **ind 2:** Field (Oe)
    * **ind 3:** Moment (emu)
    * **ind 4:** Moment Std. Err. (emu)
    * **ind 5:** moment per gram (emu/g). 
    * **ind 6:** moment per mol (emu/mol). 
    * **ind 8:** moment per mol (per atom sort)   
    
    **mass** (miligram):  if mass=None weight given in the data file is taken. \n
    **m_mol** (g/mol)   **Na4Ir3O8:** 796.612    |    **Cd2Ru2O7:** 538.953\n
    **factor:** factor is used to calculate chi per mol atom sort. e.g. for Na4Ir3O8 its factor=3 to get chi per Ir-mol\n
    **NAN:** if true, read in rows containing NAN values (which is empty entry or entry starting with #) get deleted
    '''
    info={'NAME':'', 'WEIGHT':'', 'AREA':'', 'LENGTH':'', 'SHAPE':'', 'COMMENT':'', 'SEQUENCE FILE':'','FILEOPENTIME':''}
    with open(name,'r') as f:
        ind=10000000000
        for i, line in enumerate(f):
            if line.find('[Data]')>=0:
                ind=i
            elif i==ind+1:
                header=np.array(line.split(',')[:-1])[list(cols)]
            elif line.find('SAMPLE_MASS')>=0:
                info["WEIGHT"] = line.split(',')[1]

        ind=ind+2
        if not mass:
            mass=float(info['WEIGHT'])/1000
        else:
            mass=mass/1000
        data=np.genfromtxt(name,delimiter=',',skip_header=ind,missing_values=('',),usecols=cols)
        l=len(data)
        m_emu_g=data[:,4]/mass
        m_emu_mol=data[:,4]*m_mol/mass
        m_emu_mol_factor=data[:,4]*m_mol/(mass*factor)
        data=np.hstack((data,m_emu_g.reshape(l,1),m_emu_mol.reshape(l,1),m_emu_mol_factor.reshape(l,1)))
        print(str(mass))     
#        header='Temperature [K]\t long magnetic moment [emu]\t magnetic field [Oe]\t timestamp [s]\t magnetic moment [emu/g]\t magnetic moment [emu/mol]\t Chi [emu/mol]\t Chi [emu/factor-mol]\t 1/chi [factor-mol/emu]'
        
        if NAN:
            data=data[~np.isnan(data).any(axis=1)]
       
        return data,header,info



#####################################################################################################################################################
'''MPMS'''

MPMS3_header_original=['Comment','Time Stamp (sec)','Temperature (K)','Magnetic Field (Oe)','Moment (emu)','M. Std. Err. (emu)'
                       ,'Transport Action,Averaging Time (sec)','Frequency (Hz)'
                       ,'Peak Amplitude (mm)','Center Position (mm)','Lockin Signal (V)','Lockin Signal (V)','Range','M. Quad. Signal (emu)'
                       ,'Min. Temperature (K)','Max. Temperature (K)','Min. Field (Oe)','Max. Field (Oe)','Mass (grams)','Motor Lag (deg)'
                       ,'Pressure (Torr)','Measure Count','Measurement Number','SQUID Status (code)','Motor Status (code)','Measure Status (code)'
                       ,'Motor Current (amps)','Motor Temp. (C)','Temp. Status (code)','Field Status (code)','Chamber Status (code)'
                       ,'Chamber Temp (K)','Redirection State','Average Temp (K)','Rotation Angle (deg)','Rotator state','DC Moment Fixed Ctr (emu)'
                       ,'DC Moment Err Fixed Ctr (emu)','DC Moment Free Ctr (emu)','DC Moment Err Free Ctr (emu)','DC Fixed Fit','DC Free Fit'
                       ,'DC Calculated Center (mm)','DC Calculated Center Err (mm)','DC Scan Length (mm)','DC Scan Time (s)','DC Number of Points'
                       ,'DC Squid Drift','DC Min V (V)','DC Max V (V)','DC Scans per Measure','Map 01','Map 02','Map 03','Map 04','Map 05','Map 06','Map 07','Map 08','Map 09','Map 10','Map 11','Map 12','Map 13','Map 14','Map 15','Map 16']


MPMS3_header=['Comment', 'Time', 'Temperature', 'Field', 'ACMoment', 'ACMoment_err', 'Transport_Action', 'Averaging_Time', 'Frequency'
              ,'Peak_Amplitude', 'Center_Position', 'Lockin_Signal_1', 'Lockin_Signal_2', 'Range', 'M_Quad_Signal'
              ,'Min_Temperature', 'Max_Temperature', 'Min_Field', 'Max_Field', 'Mass', 'Motor_Lag'
              ,'Pressure', 'Measure_Count', 'Measurement_Number', 'SQUID_Status', 'Motor_Status', 'Measure_Status'
              ,'Motor_Current', 'Motor_Temp', 'Temp_Status', 'Field_Status', 'Chamber_Status'
              ,'Chamber_Temp', 'Redirection_State', 'Average_Temp', 'Rotation_Angle', 'Rotator_state', 'DCMoment_Fixed_Ctr'
              ,'DCMoment_Err_Fixed_Ctr', 'DCMoment_Free_Ctr', 'DCMoment_Err_Free_Ctr', 'DC_Fixed_Fit', 'DC_Free_Fit'
              ,'DC_Calculated_Center', 'DC_Calculated_Center_Err', 'DC_Scan_Length', 'DC_Scan_Time', 'DC_Number_of_Points'
              ,'DC_Squid_Drift', 'DC_Min_V', 'DC_Max_V', 'DC_Scans_per_Measure'
              ,'Map01','Map02','Map03','Map04','Map05','Map06','Map07','Map08','Map09','Map10','Map11','Map12','Map13','Map14','Map15','Map16']





def read_MPMS3(name):
    ''' 
    * Comment
    * Time
    * **Temperature** 
    * **Field**
    * **ACMoment**
    * **ACMoment_err** 
    * Transport_Action
    * Averaging_Time
    * Frequency
    * **Peak_Amplitude**
    * **Center_Position** 
    * Lockin_Signal_1 
    * Lockin_Signal_2 
    * Range
    * M_Quad_Signal
    * Min_Temperature 
    * Max_Temperature 
    * Min_Field
    * Max_Field (Oe)
    * **Mass**
    * Motor_Lag
    * Pressure
    * Measure_Count
    * Measurement_Number 
    * SQUID_Status
    * Motor_Status
    * Measure_Status
    * Motor_Current
    * Motor_Temp
    * Temp_Status
    * Field_Status 
    * Chamber_Status
    * Chamber_Temp
    * Redirection_State
    * Average_Temp
    * Rotation_Angle
    * Rotator_state
    * **DCMoment_Fixed_Ctr**       
    * **DCMoment_Err_Fixed_Ctr** 
    * **DCMoment_Free_Ctr**
    * **DCMoment_Err_Free_Ctr**
    * DC_Fixed_Fit
    * DC_Free_Fit
    * **DC_Calculated_Center**
    * DC_Calculated_Center_Err 
    * DC_Scan_Length
    * DC_Scan_Time
    * DC_Number_of_Points
    * DC_Squid_Drift
    * DC_Min_V
    * DC_Max_V
    * DC_Scans_per_Measure \n\n
    noise_limit: value between 0 and 1: filters according to DC_Free_fit
    '''
    d=pd.read_csv(name,skiprows=27
               ,names=MPMS3_header)
    return d

def read_MPMS3_evaluate(name, mass, m_mol, meas='DC', factor=1, noise_limit_DC=0.8, noise_limit_AC=1e-5, 
                        rod = 'straw', diamagn_corr = False, m_foil = False, sweep_mode='both', sortB=False, sortT=False):
    ''' 
    * **Temperature**
    * **Chi**
    * **Moment**
    * **Moment error**
    mass in mg \n
    m_mol: string of chemical firmula\n
    noise_limit_DC: value between 0 and 1: filters according to DC_Free_fit \n
    noise_limit_AC: filtering according to ACMoment_err\n
    rod: 'straw', 'brass', 'quartz'\n
    **diamagn_corr:** e.g. diamagn_corr = 'Li1+2Ir4+1O2-3\n
    **foil:** mass in mg
    '''
    d=pd.read_csv(name,skiprows=27
               ,names=MPMS3_header)
    if sweep_mode=='up':
        Tmin, ind_min = np.min(d.Temperature), np.argmin(d.Temperature)
        Tmax, ind_max = np.max(d.Temperature),np.argmax(d.Temperature)
        if np.abs(d.Temperature[0]-Tmin) > np.abs(d.Temperature[0]-Tmax):
            ind_sweep = ind_min
        else:
            ind_sweep = ind_max
    elif sweep_mode=='down':
        pass
    elif sweep_mode=='both':
        pass
    
    if sortB:
        d = d.sort_values(by='Field')

    if sortT:
        d = d.sort_values(by='Temperature')
    
    if meas=='DC':
        M_err=d.DC_Free_Fit
        d=d[d.DC_Free_Fit>noise_limit_DC]
        M=d.DCMoment_Free_Ctr
    elif meas=='AC':   
        M_err=d.ACMoment_err
        d = d[d.ACMoment_err < noise_limit_AC]
        M=d.ACMoment
    else:
        print('wrong meas keyword')
        
    T = d.Temperature
    B = d.Field[0]
    mass=mass/1000
    
    if rod == 'quartz':
        M = M - quartz_rod(T,B)
    elif rod == 'brass':
        pass
    else:
        pass
    
    if m_foil:
        M = M + 8.0e-10 * B * m_foil
    else:
        pass
    
    if type(m_mol)==str:
        m_mol=get_molar_mass(m_mol)
    else:
        m_mol = m_mol
    chi=M*m_mol/(mass*B*factor)
    if diamagn_corr:
        chi = chi - get_diamag_corr(diamagn_corr)/factor
    else:
        pass
    return T.values, chi.values, M.values, M_err.values, d.Field.values

    
def read_mpms_MT(name, m_mol, mass=None, factor=1, cols=(3,4,2,0,5), NAN=False, save=False, del_corrupt_data=False, threshold=0.1, diamagn_corr = False, m_foil = False):
    '''
    read mpms M(T) .dat file (columns are seperated by komma). functions looks for the keyword '[Data]' and starts reading data in the next line.\n
    **returns:** data,header,info\n
    **columns in data:**\n
    * **ind 0:** Temperature (K)
    * **ind 1:** Long Moment (emu)
    * **ind 2:** Field (Oe)
    * **ind 3:** Time (s)
    * **ind 4:** Long scan standard deviation
    * **ind 5:** moment per gram (emu/g). 
    * **ind 6:** moment per mol (emu/mol). 
    * **ind 7:** chi per mol atom sort (emu/mol)
    * **ind 8:** 1/chi (mol/emu) (per atom sort)   
    
    **mass** (miligram):  if mass=None weight given in the data file is taken. \n
    **m_mol** (g/mol)   **Na4Ir3O8:** 796.612    |    **Cd2Ru2O7:** 538.953\n
    **factor:** factor is used to calculate chi per mol atom sort. e.g. for Na4Ir3O8 its factor=3 to get chi per Ir-mol\n
    **save:** if save=True, the calculated and original data (cols 3,4,2,0,5) gets saved as .txt file with the same name as the original .dat file\n
    **NAN:** if true, read in rows containing NAN values (which is empty entry or entry starting with #) get deleted\n
    **diamagn_corr:** e.g. diamagn_corr = 'Li1+2Ir4+1O2-3
    '''
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
        M = data[:,1]
        if m_foil:
            M = M + 8.0e-10 * B * m_foil
        else:
            pass
        m_emu_g = M / mass
        m_emu_mol = M * m_mol / mass
        chi_emu_factor_mol = M * m_mol / (mass * B * factor)
        if diamagn_corr:
            chi_emu_factor_mol = chi_emu_factor_mol - get_diamag_corr(diamagn_corr)/factor
        else:
            pass
        one_over_chi=1/chi_emu_factor_mol
        data=np.hstack((data,m_emu_g.reshape(l,1),m_emu_mol.reshape(l,1),chi_emu_factor_mol.reshape(l,1),one_over_chi.reshape(l,1)))
        print(str(mass)+' g',str(B)+' Oe',str(m_mol)+' g/mol')     
#        header='Temperature [K]\t long magnetic moment [emu]\t magnetic field [Oe]\t timestamp [s]\t magnetic moment [emu/g]\t magnetic moment [emu/mol]\t Chi [emu/mol]\t Chi [emu/factor-mol]\t 1/chi [factor-mol/emu]'
        
        if NAN:
            data=data[~np.isnan(data).any(axis=1)]
        if del_corrupt_data:
            data=data[np.abs(data[:,4]/data[:,1])<threshold,:]
        if save:
            name=name[:-3]+'txt'                
            np.savetxt(name,data,delimiter='\t',header='Temperature [K]\t long magnetic moment [emu]\t magnetic field [Oe]\t timestamp [s]\t magnetic moment [emu/g]\t magnetic moment [emu/mol]\t Chi [emu/mol]\t Chi [emu/factor-mol]\t 1/chi [factor-mol/emu]')
        
        return data,header,info


def read_mpms_raw(name,splitnumber=False):
    '''
    **Temp start/stop:** ind 3 / ind 4
    **scan number (for each temperature):** ind 5\n    
    **position:** ind 7\n
    **Long voltage:** ind 9\n
    **Lomg Average voltage** ind 10\n
    **Long Detrended Voltage:** ind 11
    **long scaled response:** ind 16
    **Long Avg. Scaled Response:** ind 17
    '''
    with open(name,'r') as f:
        ind=np.nan
        for i, line in enumerate(f):
            if line.find('[Data]')>=0:
                ind=i
            elif i==ind+1:
                header=np.array(line.split(',')[:-1])
        ind=ind+2
    data=np.genfromtxt(name,delimiter=',', skip_header=ind, missing_values=('',))
    number_of_scans=np.max(data[:,5])
#    print(number_of_scans)
    data=data[data[:,5]==number_of_scans]
    if splitnumber==False:
        splitnumber=len(np.where(data[:,7]==np.min(data[:,7]))[0])
        data=np.split(data,splitnumber)
    else:
        length,y=data.shape
#        print int(length)
        splitnumber=int(length)/splitnumber
        data=np.split(data,splitnumber)
    return data,header
    





#####################################################################################################################################################
'''SPECTRA'''


def read_spectra(name,lower_limit=693,upper_limit=700, normalize=False):
    with open(name, 'r+') as f:
        text = f.read()
        f.seek(0)
        f.truncate()
        f.write(text.replace(',', '.'))
        f.close()
    info={'average':'', 'box':'', 'integration':'','date':''}
    with open(name,'r') as f:
        for i, line in enumerate(f):
            if line.find('Date:')>=0:
                info["date"] = line[line.find("Date:")+6:-1]
            elif line.find('Integration Time')>=0:
                info["integration"] = line[line.find("Integration Time")+17:-1]
            elif line.find('Scans to average:')>=0:
                info["average"] = line[line.find("Scans to average:")+18:-1]
            elif line.find('Boxcar width:')>=0:
                info["box"] = line[line.find("Boxcar width:")+14:-1]
    data=np.genfromtxt(name,skip_header=14)
    x,y=data[:,0],data[:,1]
    if lower_limit:
        ind=find_nearest(x,lower_limit)
        x,y=x[ind:],y[ind:]
    if upper_limit:
        ind=find_nearest(x,upper_limit)
        x,y=x[:ind],y[:ind]
    if normalize:
        y=100*y/np.max(y)
    return x,y,info



    
#####################################################################################################################################################
'''SEVERAL FUNCTIONS'''  


def dsin(x):
    return np.sin(np.deg2rad(x))

def dcos(x):
    return np.cos(np.deg2rad(x))

def dtan(x):
    return np.tan(np.deg2rad(x))

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

def cluster_HC_data(x,y,threshold=0.1):
    b=(np.diff(x)>threshold).nonzero()
    x_group=np.split(x,b[0]+1)
    y_group=np.split(y,b[0]+1)
    x_mean = np.array([np.mean(el) for el in x_group]) 
    y_mean = np.array([np.mean(el) for el in y_group]) 
    return x_mean,y_mean
    
    
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
    
    
def find_point_data_distance(x,y,x0,y0):
    '''calculates the shortest distance of a point (x0,y0) to a data curve (x,y)\n
    output: distance d'''
    x,y=np.array(x),np.array(y)
    d=np.min(np.sqrt((x-x0)**2+(y-y0)**2))
    return d

def capacitance_parallel_wires(l=1,d=0.03,s=0.1,epsr=1):
    '''capacitance (pF) of two parallel wires of diameter d (mm), seperation s (mm,) between wire centers and length l in mm\n 
    epsr: relative permittivity'''
    C=np.pi*epsr*l*constants.eps0/np.arccosh(s/d)/(10**(-12))/1000
    return C
    
def capacitance_parallel_plates(d=1,A=1,epsr=1):
    '''capacitance (pF) of two parallel plates with area A (mm^2) and seperation s (mm)\n 
    epsr: relative permittivity'''
    C=epsr*constants.eps0*A/d/1000/(10**(-12))
    return C

def eps_r_parallel_plates(C,d=1,A=1):
    '''C in Farad,
    area A (mm^2) and seperation s (mm)'''
    eps_r=1000*C*d/(A*constants.eps0)
    return eps_r



def average_WK_data(name,n_steps):
    '''
    build average of WK data when many points are taken at each temperature\n
    returns: temperature, capacitance, cap std, loss D, loss std
    '''
    d=np.genfromtxt(name)
    a=np.split(d,n_steps)
    temp=[]
    cap=[]
    loss=[]
    cap_std=[]
    loss_std=[]
    
    for i in a:
        T_mean = np.mean(i[:,3])
        C_mean = np.mean(i[:,1])
        C_std = np.std(i[:,1])
        D_mean = np.mean(i[:,2])
        D_std = np.std(i[:,2])
        temp.append(T_mean)
        cap.append(C_mean)
        loss.append(D_mean)
        cap_std.append(C_std)
        loss_std.append(D_std)
    
    return temp, cap, cap_std, loss, loss_std


#eff=em*(2*di*(ei-em)+ei+2*em)/(2*em+ei+di*(em-ei))


def sort_after_column(a,col=0):
    '''sort array a after column col'''
    return a[a[:,col].argsort()]



def symmetrising(x, y, mode=0):
    '''mode=o: linear part (uneven contribution x,x**3,...)\n
    mode=1: quadratic part (even contribution X**2, x**4 ...)\n
    x data must be roughly symmetric around zero'''
    d=np.array([x,y]).T    
    d=sort_after_column(d,0)
    x,y=d[:,0],d[:,1]
    spl=interp1d(x,y,fill_value=(y[0],y[-1]),bounds_error=False)
    a=y[x>0]
    b=spl(-x[x>0])
    if mode==0:
        y=(a-b)/2
        return x[x>0],y
    elif mode==1:
        y=(a+b)/2-(a[0]+b[0])/2
        return x[x>0],y


def delete_spikes_in_data(a,col=[0,1],s=1,threshold=3):
    '''a: 2d-array
    col: tuple with first index for x-data and second index for y-data'''
    x,y=a[:,col[0]],a[:,col[1]]
    spl=UnivariateSpline(x,y,k=5,s=s/len(x))
    diff=np.abs(y-spl(x))
    mean=np.mean(np.abs(diff))
    a=a[diff<threshold*mean,:]
    return a



'''get molar mass from chemical formula'''
'''############################################################################################################################################'''

atomic_mass = {
    "H": 1.0079, "He": 4.0026, "Li": 6.941, "Be": 9.0122, 
    "B": 10.811, "C": 12.011, "N": 14.007, "O": 15.999, "F": 18.998,
    "Ne": 20.180, "Na": 22.990, "Mg": 24.305, "Al": 26.982,
    "Si": 28.086, "P": 30.974, "S": 32.065, "Cl": 35.453,
    "Ar": 39.948, "K": 39.098, "Ca": 40.078, "Sc": 44.956,
    "Ti": 47.867, "V": 50.942, "Cr": 51.996, "Mn": 54.938,
    "Fe": 55.845, "Co": 58.933, "Ni": 58.693, "Cu": 63.546,
    "Zn": 65.39, "Ga": 69.723, "Ge": 72.61, "As": 74.922,
    "Se":78.96, "Br": 79.904, "Kr": 83.80, "Rb": 85.468, "Sr": 87.62,
    "Y": 88.906, "Zr": 91.224, "Nb": 92.906, "Mo": 95.94,
    "Tc": 97.61, "Ru": 101.07, "Rh": 102.91, "Pd": 106.42,
    "Ag": 107.87, "Cd": 112.41, "In": 114.82, "Sn": 118.71,
    "Sb": 121.76, "Te": 127.60, "I": 126.90, "Xe": 131.29,
    "Cs": 132.91, "Ba": 137.33, "La": 138.91, "Ce": 140.12,
    "Pr": 140.91, "Nd": 144.24, "Pm": 145.0, "Sm": 150.36, "Eu": 151.96,
    "Gd": 157.25, "Tb": 158.93, "Dy": 162.50, "Ho": 164.93, "Er": 167.26,
    "Tm": 168.93, "Yb": 173.04, "Lu": 174.97, "Hf": 178.49, "Ta": 180.95,
    "W": 183.84, "Re": 186.21, "Os": 190.23, "Ir": 192.22, "Pt": 196.08,
    "Au": 196.08, "Hg": 200.59, "Tl": 204.38, "Pb": 207.2, "Bi": 208.98,
    "Po": 209.0, "At": 210.0, "Rn": 222.0, "Fr": 223.0, "Ra": 226.0,
    "Ac": 227.0, "Th": 232.04, "Pa": 231.04, "U": 238.03, "Np": 237.0,
    "Pu": 244.0, "Am": 243.0, "Cm": 247.0, "Bk": 247.0, "Cf": 251.0, "Es": 252.0,
    "Fm": 257.0, "Md": 258.0, "No": 259.0, "Lr": 262.0, "Rf": 261.0, "Db": 262.0,
    "Sg": 266.0, "Bh": 264.0, "Hs": 269.0, "Mt": 268.0
}

def find_closing_paren(tokens):
    count = 0
    for index, tok in enumerate(tokens):
        if tok == ')':
            count -= 1
            if count == 0:
                return index
        elif tok == '(':
            count += 1
    raise ValueError('unmatched parentheses')

def parse(tokens, stack):
    if len(tokens) == 0:
        return sum(stack)
    tok = tokens[0]
    if tok == '(':
        end = find_closing_paren(tokens)
        stack.append(parse(tokens[1:end], []))
        return parse(tokens[end + 1:], stack)
    elif tok.isdigit():
        stack[-1] *= int(tok)
    else:
        stack.append(atomic_mass[tok])
    return parse(tokens[1:], stack)

def get_molar_mass(formula):
    tokens = re.findall(r'[A-Z][a-z]*|\d+|\(|\)', formula)
    return(parse(tokens, []))



'''############################################################################################################################################'''

def temp_sep(data,col=3,turn=1):
    '''seperates the cooling up and cooling down curve of a temperaure sweep.\n
    col: is the column of temperature (start counting with 0, col=3 for PPMS resistivity option)\n
    turn=1: turningpoint is at lowest temperature) or\n
    turn=2: turning point is at highest temperature\n
    output: two data sets down and up (in this order)'''
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
        if (Tmax-data[imax+1,col])<(Tmax-data[imax-1,col]):
            up=data[:imax+1,:]
            down=data[imax+1:,:]
        else:
            up=data[:imax,:]
            down=data[imax:,:] 
    else:
        print('unvalid entry for variable "turn"')
    
    return down,up


def find_cooling_rate(data,col_T=3,col_t=1,_range=2,min_num_of_points=10):
    rate=np.diff(data[:,col_T])/np.diff(data[:,col_t]/60)
    _range=np.round(_range, decimals=1)+0.15
    bins=np.round(2*_range/0.1, decimals=0)
    hist=np.histogram(rate,bins,range=(-_range,_range))
    return hist[1][hist[0]>min_num_of_points]+0.05

    
def move_figure(newX=2000, newY=100):
    mngr=pl.get_current_fig_manager()
    geom = mngr.window.geometry()
    x,y,dx,dy = geom.getRect()
    mngr.window.setGeometry(newX, newY, dx, dy)


'''################################################################################################################################'''   
'''Matplotlib parameters (and plotting function)'''

def matplotlib_parameters2(figsize = 'golden ratio', size=1, textsize=1.5, xmargin=False, ymargin=False, style=1):
    '''
    **size=1** gives a figwidth of 427.21597 points which is the columnwidth of a latex DIN A4 standard document.\n 
    If your figure in your latex document is 0.8 wide you should choose size=0.8 and the fontsizes specified in this function correspond to the fontsizes in the latex document\n     
    
    **textsize:** float, changes fontsize of x,y-axis labels, legend, text, title, x,y-tick-labels at once but keeps the fontsize ratio between them\n    
    
    **style=1:** is for powerpoint presentations, set size=3    
    '''
    #calculate golden ratio
    if figsize=='golden ratio':
        fig_width_pt = size*427.21597  # Get this from LaTeX using \showthe\columnwidth
        inches_per_pt = 1.0/72.27               # Convert pt to inch
        golden_mean = (np.sqrt(5)-1.0)/2.0         # Aesthetic ratio
        fig_width = fig_width_pt*inches_per_pt  # width in inches
        fig_height = fig_width*golden_mean      # height in inches
        fig_size =  [fig_width,fig_height]
    elif figsize=='slides':
        fig_size=(size*4,size*3)
    elif figsize=='square':
        fig_size=(size*3,size*3)
    else:
        fig_size = figsize

    params1 = {'backend': 'ps',
          'axes.labelsize': textsize*30,
          'font.size': textsize*30,
          'legend.fontsize': textsize*27,
          'legend.labelspacing':0.25,
          'xtick.labelsize': textsize*25,
          'ytick.labelsize': textsize*25,
          'figure.figsize': fig_size,
          'figure.dpi': 300,
          'axes.labelpad': 8,
          'axes.grid': True,
          'axes.linewidth':2,
          'legend.numpoints': 1,
          'legend.handlelength': 0.8,
          'legend.handleheight': 0.8
#          'axes.labelweight': 'normal',
#          'axes.titlepad':20,
#          'figure.subplot.bottom': 0.1, # just if subplots are used
#          'figure.subplot.hspace': 0.2,
#          'figure.subplot.left': 0.125,
#          'figure.subplot.right': 0.9,
#          'figure.subplot.top': 0.9,
#          'figure.subplot.wspace': 0.2,
          }
    if xmargin:
        params1['axes.xmargin'] = xmargin
    if ymargin:
        params1['axes.ymargin'] = ymargin
    if style==1:
        return params1
    else:
        pass

font_serif = ['DejaVu Serif',
              'Bitstream Vera Serif',
              'Computer Modern Roman',
              'New Century Schoolbook',
              'Century Schoolbook L',
              'Utopia',
              'ITC Bookman',
              'Bookman',
              'Nimbus Roman No9 L',
              'Times New Roman',
              'Times',
              'Palatino',
              'Charter',
              'serif']

font_sans_serif = ['DejaVu Sans',
                   'Bitstream Vera Sans',
                   'Computer Modern Sans Serif',
                   'Lucida Grande',
                   'Verdana',
                   'Geneva',
                   'Lucid',
                   'Arial',
                   'Helvetica',
                   'Avant Garde',
                   'sans-serif']







'''################################################################################################################################'''   
'''spherical harmonics, p-orbitals and d-orbitals (sympy/mpmath)'''

def y_lm(l,m):
    '''returns absolute value of real part of spherical harmonics so that it can be plotted as \n
    mpmath.splot(Y_lm(l,m), [0,pi], [0,2*pi], points=100, keep_aspect=True, axes=False)'''
    def g(theta,phi):
        R = abs(re(spherharm(l,m,theta,phi)))
        x = R*cos(phi)*sin(theta)
        y = R*sin(phi)*sin(theta)
        z = R*cos(theta)
        return [x,y,z]
    return g
    
    
def combine_ylm(l,m1,m2,a,b):
    def g(theta,phi):
        R = abs(re(a*spherharm(l,m1,theta,phi)+b*spherharm(l,m2,theta,phi)))**2
        x = R*cos(phi)*sin(theta)
        y = R*sin(phi)*sin(theta)
        z = R*cos(theta)
        return [x,y,z]
    return g

def p_z():
    '''plot with mpmath.splot(p_z(), [0,pi], [0,2*pi])'''
    return y_lm(1,0)
    
def p_x():
    '''plot with mpmath.splot(p_z(), [0,pi], [0,2*pi])'''
    return combine_ylm(1,-1,1,1,-1)

def p_y():
    '''plot with mpmath.splot(p_z(), [0,pi], [0,2*pi])'''
    return combine_ylm(1,-1,1,1j,1j)

def d_xy():
    '''plot with mpmath.splot(p_z(), [0,pi], [0,2*pi])'''
    return combine_ylm(2,-2,2,1j,-1j)

def d_xz():
    '''plot with mpmath.splot(p_z(), [0,pi], [0,2*pi])'''
    return combine_ylm(2,-1,1,1,-1)
    
def d_yz():
    '''plot with mpmath.splot(p_z(), [0,pi], [0,2*pi])'''
    return combine_ylm(2,-1,1,1j,1j)
    
def d_x2y2():
    return combine_ylm(2,-2,2,1,1)
    
def d_z2():
    '''plot with mpmath.splot(p_z(), [0,pi], [0,2*pi])'''
    return y_lm(2,0)

    
'''################################################################################################################################'''
'''numerical spherical harmonics (numpy/scipy)'''    

def y_lm_num(l,m,phimax=np.pi,thetamax=2*np.pi,num_points=100):    
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
    
   
def combine_ylm_num(l,m1,m2,a,b,phimax=np.pi,thetamax=2*np.pi,num_points=100):    
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
    s=1/np.sqrt(2)*np.real(a*sph_harm(m1,l,theta,phi)+b*sph_harm(m2,l,theta,phi))
    r=np.abs(s)**2
    x = r*np.sin(phi) * np.cos(theta)
    y = r*np.sin(phi) * np.sin(theta)
    z = r*np.cos(phi)
    return [x,y,z],s

def p_z_num():
    '''plot with mpmath.splot(p_z(), [0,pi], [0,2*pi])'''
    return y_lm_num(1,0)
    
def p_x_num():
    '''plot with mpmath.splot(p_z(), [0,pi], [0,2*pi])'''
    return combine_ylm_num(1,-1,1,1,-1)

def p_y_num():
    '''plot with mpmath.splot(p_z(), [0,pi], [0,2*pi])'''
    return combine_ylm_num(1,-1,1,1j,1j)

def d_xy_num():
    '''plot with mpmath.splot(p_z(), [0,pi], [0,2*pi])'''
    return combine_ylm_num(2,-2,2,1j,-1j)

def d_xz_num():
    '''plot with mpmath.splot(p_z(), [0,pi], [0,2*pi])'''
    return combine_ylm_num(2,-1,1,1,-1)
    
def d_yz_num():
    '''plot with mpmath.splot(p_z(), [0,pi], [0,2*pi])'''
    return combine_ylm_num(2,-1,1,1j,1j)
    
def d_x2y2_num():
    return combine_ylm_num(2,-2,2,1,1)
    
def d_z2_num():
    '''plot with mpmath.splot(p_z(), [0,pi], [0,2*pi])'''
    return y_lm_num(2,0)
    
    
    
'''################################################################################################################################'''
'''Spin matrices (sympy)'''    
    

'''Spin 1/2'''    
Sz_05=0.5*Matrix([[1,0], [0,-1]])
Sx_05=0.5*Matrix([[0,1], [1,0]])
Sy_05=-0.5*I*Matrix([[0,1], [-1,0]])
Splus_05=Matrix([[0,1], [0,0]])
Sminus_05=Matrix([[0,0], [0,1]])
    
    
'''Spin 1'''    
Sz_1=Matrix([[1,0,0], [0,0,0], [0,0,-1]])
Sx_1=1/sqrt(2)*Matrix([[0,1,0], [1,0,1], [0,1,0]]) 
Sy_1=-1/sqrt(2)*I*Matrix([[0,1,0], [-1,0,1], [0,-1,0]]) 
Splus_1=sqrt(2)*Matrix([[0,1,0], [0,0,1], [0,0,0]]) 
Sminus_1=sqrt(2)*Matrix([[0,0,0], [1,0,0], [0,1,0]])     
    

    

    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
