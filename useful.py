# -*- coding: utf-8 -*-
"""
Created on Mon Jan 25 11:47:40 2016

@author: admin
"""

import numpy as np
from scipy.interpolate import UnivariateSpline
from mpmath import re,pi,cos,sin,spherharm
from scipy.special import sph_harm
from sympy.matrices import Matrix, eye, zeros, ones, diag
from sympy import I,sqrt
from matplotlib import rc,cm
from matplotlib import pyplot as pl



import os
dir = os.path.dirname(__file__)
filename = os.path.join(dir, 'data','X91774.txt')
filename2 = os.path.join(dir, 'data','161117-cernox-R07-C12.txt')

'''Puk sensor calibration spline function'''
T_puk_calibration_data=np.genfromtxt(filename,skip_header=3,usecols=(0,1)) 

'''Puk sensor calibration spline function'''
def cernox_R2Temp(x):
    '''
    **input:** Resistance of cernox R07-C12
    **output:** Temperature value according to calibration file '161117-cernoc-R07-C12'
    '''
    cernox_R07_C12_calibration_data=np.genfromtxt(filename2,skip_header=1) 
    spl=UnivariateSpline(cernox_R07_C12_calibration_data[:,1],cernox_R07_C12_calibration_data[:,0],k=3,s=0.01)
    return spl(x)




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


    

def read_ppms_RT(name,cell=False):
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
    data[:,19:23]=np.abs(data[:,19:23])
    if cell:
        spl=UnivariateSpline(T_puk_calibration_data[::-1,1],T_puk_calibration_data[::-1,0],k=3,s=0.01)
        T=spl(data[:,19]).reshape(-1,1)
        data=np.hstack((data,T))
    else:
        pass
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


def read_mpms_raw(name):
    '''
    **Temp start/stop:** ind 3 / ind 4
    **scan number (for each temperature):** ind 5\n    
    **position:** ind 7\n
    **Long voltage:** ind 9\n
    **Lomg Average voltage** ind 10\n
    **Long Detrended Voltage:** ind 11
    '''
    with open(name,'r') as f:
        ind=np.nan
        for i, line in enumerate(f):
            if line.find('[Data]')>=0:
                ind=i
            elif i==ind+1:
                header=np.array(line.split(',')[:-1])
        ind=ind+2
    data=np.genfromtxt(name,delimiter=',',skip_header=ind,missing_values=('',))
    number_of_scans=np.max(data[:,5])
    data=data[data[:,5]==number_of_scans]
    splitnumber=len(np.where(data[:,7]==np.min(data[:,7]))[0])
    data=np.split(data,splitnumber)
    return data,header
    


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
    
    
def find_point_data_distance(x,y,x0,y0):
    '''calculates the shortest distance of a point (x0,y0) to a data curve (x,y)\n
    output: distance d'''
    x,y=np.array(x),np.array(y)
    d=np.min(np.sqrt((x-x0)**2+(y-y0)**2))
    return d
    


'''############################################################################################################################################'''


def gauss_1(x,a,b,c,d):
    '''
    FWHM = 2*sqrt(2*ln(2))*c = 2.35482*c\n
    a*np.exp(-(x-b)**2/(2*c))+d    
    '''
    return 100*a*np.exp(-(x-100*b)**2/(2*c*0.1))+d

def gauss_2(x,a1,b1,c1,a2,b2,c2,d):
    '''FWHM = 2*sqrt(2*ln(2))*c = 2.35482*c'''
    return 100*a1*np.exp(-(x-100*b1)**2/(2*c1*0.1))+100*a2*np.exp(-(x-100*b2)**2/(2*c2*0.1))+d
        
def lorentz_1(x,a,b,c,d):
    '''FWHM = 2*c'''
    return a*c/((x-b)**2+c**2)+d
    
def lorentz_2(x,a1,b1,c1,a2,b2,c2,d):
    '''FWHM = 2*c'''
    return a1*0.1*c1/((x-100*b1)**2+(0.1*c1)**2)+a2*0.1*c2/((x-100*b2)**2+(0.1*c2)**2)+d
    


def spectrum_peak_parameters(x,y):
    peaks,dips=peakdetect(y,lookahead=20)
    b1,b2=x[peaks[0][0]],x[peaks[1][0]]
    return b1,b2
    



def _datacheck_peakdetect(x_axis, y_axis):
    if x_axis is None:
        x_axis = range(len(y_axis))
    
    if len(y_axis) != len(x_axis):
        raise (ValueError, 
                'Input vectors y_axis and x_axis must have same length')
    
    #needs to be a numpy array
    y_axis = np.array(y_axis)
    x_axis = np.array(x_axis)
    return x_axis, y_axis

    
def peakdetect(y_axis, x_axis = None, lookahead = 40, delta=20):
    """
    Converted from/based on a MATLAB script at: 
    http://billauer.co.il/peakdet.html
    
    function for detecting local maximas and minmias in a signal.
    Discovers peaks by searching for values which are surrounded by lower
    or larger values for maximas and minimas respectively
    
    keyword arguments:
    y_axis -- A list containg the signal over which to find peaks
    x_axis -- (optional) A x-axis whose values correspond to the y_axis list
        and is used in the return to specify the postion of the peaks. If
        omitted an index of the y_axis is used. (default: None)
    lookahead -- (optional) distance to look ahead from a peak candidate to
        determine if it is the actual peak (default: 200) 
        '(sample / period) / f' where '4 >= f >= 1.25' might be a good value
    delta -- (optional) this specifies a minimum difference between a peak and
        the following points, before a peak may be considered a peak. Useful
        to hinder the function from picking up false peaks towards to end of
        the signal. To work well delta should be set to delta >= RMSnoise * 5.
        (default: 0)
            delta function causes a 20% decrease in speed, when omitted
            Correctly used it can double the speed of the function
    
    return -- two lists [max_peaks, min_peaks] containing the positive and
        negative peaks respectively. Each cell of the lists contains a tupple
        of: (position, peak_value) 
        to get the average peak value do: np.mean(max_peaks, 0)[1] on the
        results to unpack one of the lists into x, y coordinates do: 
        x, y = zip(*tab)
    """
    max_peaks = []
    min_peaks = []
    dump = []   #Used to pop the first hit which almost always is false
       
    # check input data
    x_axis, y_axis = _datacheck_peakdetect(x_axis, y_axis)
    # store data length for later use
    length = len(y_axis)
    
    
    #perform some checks
    if lookahead < 1:
        raise ValueError, "Lookahead must be '1' or above in value"
    if not (np.isscalar(delta) and delta >= 0):
        raise ValueError, "delta must be a positive number"
    
    #maxima and minima candidates are temporarily stored in
    #mx and mn respectively
    mn, mx = np.Inf, -np.Inf
    
    #Only detect peak if there is 'lookahead' amount of points after it
    for index, (x, y) in enumerate(zip(x_axis[:-lookahead], 
                                        y_axis[:-lookahead])):
        if y > mx:
            mx = y
            mxpos = x
        if y < mn:
            mn = y
            mnpos = x
        
        ####look for max####
        if y < mx-delta and mx != np.Inf:
            #Maxima peak candidate found
            #look ahead in signal to ensure that this is a peak and not jitter
            if y_axis[index:index+lookahead].max() < mx:
                max_peaks.append([mxpos, mx])
                dump.append(True)
                #set algorithm to only find minima now
                mx = np.Inf
                mn = np.Inf
                if index+lookahead >= length:
                    #end is within lookahead no more peaks can be found
                    break
                continue
            #else:  #slows shit down this does
            #    mx = ahead
            #    mxpos = x_axis[np.where(y_axis[index:index+lookahead]==mx)]
        
        ####look for min####
        if y > mn+delta and mn != -np.Inf:
            #Minima peak candidate found 
            #look ahead in signal to ensure that this is a peak and not jitter
            if y_axis[index:index+lookahead].min() > mn:
                min_peaks.append([mnpos, mn])
                dump.append(False)
                #set algorithm to only find maxima now
                mn = -np.Inf
                mx = -np.Inf
                if index+lookahead >= length:
                    #end is within lookahead no more peaks can be found
                    break
            #else:  #slows shit down this does
            #    mn = ahead
            #    mnpos = x_axis[np.where(y_axis[index:index+lookahead]==mn)]
    
    
    #Remove the false hit on the first value of the y_axis
    try:
        if dump[0]:
            max_peaks.pop(0)
        else:
            min_peaks.pop(0)
        del dump
    except IndexError:
        #no peaks were found, should the function return empty lists?
        pass
        
    return [max_peaks, min_peaks]


def temp_sep(data,col=3,turn=1):
    '''seperates the cooling up and cooling down curve of a temperaure sweep.\n
    col: is the column of temperature (start counting with 0, col=3 for PPMS resistivity option)\n
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


def find_cooling_rate(data,col_T=3,col_t=1,_range=2,min_num_of_points=10):
    rate=np.diff(data[:,col_T])/np.diff(data[:,col_t]/60)
    _range=np.round(_range, decimals=1)+0.15
    bins=np.round(2*_range/0.1, decimals=0)
    hist=np.histogram(rate,bins,range=(-_range,_range))
    return hist[1][hist[0]>min_num_of_points]+0.05
    




'''################################################################################################################################'''   
'''Matplotlib parameters (and plotting function)'''

def matplotlib_parameters(size=1, textsize=1, style=1):
    '''
    **size=1** gives a figwidth of 427.21597 points which is the columnwidth of a latex DIN A4 standard document.\n 
    If your figure in your latex document is 0.8 wide you should choose size=0.8 and the fontsizes specified in this function correspond to the fontsizes in the latex document\n     
    
    **textsize:** float, changes fontsize of x,y-axis labels, legend, text, title, x,y-tick-labels at once but keeps the fontsize ratio between them\n    
    
    **style=1:** is for powerpoint presentations, set size=3    
    '''
    fig_width_pt = size*427.21597  # Get this from LaTeX using \showthe\columnwidth
    inches_per_pt = 1.0/72.27               # Convert pt to inch
    golden_mean = (np.sqrt(5)-1.0)/2.0         # Aesthetic ratio
    fig_width = fig_width_pt*inches_per_pt  # width in inches
    fig_height = fig_width*golden_mean      # height in inches
    fig_size =  [fig_width,fig_height]

    params1 = {'backend': 'ps',
          'axes.labelsize': textsize*30,
          'font.size': textsize*30,
          'legend.fontsize': textsize*26,
          'legend.labelspacing':0.3,
          'xtick.labelsize': textsize*24,
          'ytick.labelsize': textsize*24,
          'figure.figsize': fig_size,
          'figure.dpi': 600,
          'axes.labelpad': 6,
          'axes.grid': True,
          'axes.linewidth':4
#          'axes.titlepad':20,
#          'figure.subplot.bottom': 0.1, # just if subplots are used
#          'figure.subplot.hspace': 0.2,
#          'figure.subplot.left': 0.125,
#          'figure.subplot.right': 0.9,
#          'figure.subplot.top': 0.9,
#          'figure.subplot.wspace': 0.2,
#          'axes.ymargin': 0 #sets yrange with a certain margin to top and botton
          }    
    if style==1:
        return params1
    else:
        pass


def figure(n,x,y,xlabel='',ylabel='', labels=[''], legend=True, xlim=None,ylim=None,title='', 
           xlog=False, ylog=False, linestyles=[], color=[], colormap=[]):
    fig=pl.figure(n)
    ax=pl.axes()
    
    if isinstance(x,list):
        pass
    else:
        print('x and y must must be a list')
    
    lines=[]
    for i in range(len(x)):
        line,=pl.plot(x[i],y[i])
        lines.append(line)
    
    ax.set_xlabel(xlabel)
    ax.set_ylabel(ylabel)
    ax.set_title(title)
    
    if xlim:
        ax.set_xlim(xlim[0],xlim[1])
    else:
        pass
    if ylim:
        ax.set_ylim(ylim[0],ylim[1])
    else:
        pass
    
    
    if ylog:
        ax.set_yscale('log')
    else:
        ax.set_yscale('linear')
#    if len(labels)<len(lines):
#        l=['']*(len(lines)-len(labels))
#        labels.extend(l)        
        
    if legend:
        ax.legend(lines,labels)
    
    return fig,ax,lines



# These are the "Tableau 20" colors as RGB.    
tableau20 = [(31/255., 119/255., 180/255.), (174/255., 199/255., 232/255.), (255/255., 127/255., 14/255.), (255/255., 187/255., 120/255.),    
             (44/255., 160/255., 44/255.), (152/255., 223/255., 138/255.), (214/255., 39/255., 40/255.), (255/255., 152/255., 150/255.),    
             (148/255., 103/255., 189/255.), (197/255., 176/255., 213/255.), (140/255., 86/255., 75/255.), (196/255., 156/255., 148/255.),    
             (227/255., 119/255., 194/255.), (247/255., 182/255., 210/255.), (127/255., 127/255., 127/255.), (199/255., 199/255., 199/255.),    
             (188/255., 189/255., 34/255.), (219/255., 219/255., 141/255.), (23/255., 190/255., 207/255.), (158/255., 218/255., 229/255.)]    
  
hexcols = ['#332288', '#88CCEE', '#44AA99', '#117733', '#999933', '#DDCC77', 
           '#CC6677', '#882255', '#AA4499', '#661100', '#6699CC', '#AA4466',
           '#4477AA']


#fig1=pl.figure(1)
#ax1=pl.axes() #[0.15,0.15,0.95-0.15,0.95-0.15]

#ax1.plot()

#ax1.set_xlabel(r'')
#ax1.set_ylabel(r'')
#ax1.set_title(r'')

#ax1.set_xlim(0,20)
#ax1.set_ylim(0,20)
#ax1.set_xscale('log')
#ax1.set_yscale('log')
#ax1.legend()
#pl.close()



'''################################################################################################################################'''   
'''spherical harmonics, p-orbitals and d-orbitals (sympy/mpmath)'''

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
    s=1/np.sqrt(2)*np.real(a*sph_harm(m1,l,theta,phi)+b*sph_harm(m2,l,theta,phi))
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
    

    

    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
