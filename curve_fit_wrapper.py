# -*- coding: utf-8 -*-
"""
Created on Sat Jan 26 18:15:59 2019

@author: FMQ3_1
"""
import numpy as np
from scipy.optimize import curve_fit

def insert_in_list(lst, inds, values):
    for ind,value in zip(inds,values):
        lst.insert(ind,value)
    return lst

def wrf(func, inds, values):
    '''fixes arguments of function 'func' specified by index in 'inds' to value of 'values[index]'
    inds: list
    values: list
    '''
    def fn(x, *params):
        params=list(params)
        params_new = insert_in_list(params, inds, values)
        return func(x, *params_new)
    return fn 

def curve_fit_wrapper(func, x, y, fix=([],[]), fitrange = [], **kwargs):
    '''
    wrapper for optimise.curve_fit to allow fixing parameters to certain value 
    where only other parameters are varied for fitting\n
    fix = ([ind1, ..., indn],[value1, ..., valuen])
    parameters of number 'indn' is fixed to value 'valuen'\n\n
    fitrange: specifies fitting range of x data.
    kwargs are normal arguments for curve_fit like:
    maxfev = int, bounds = ([lower1,...],[upper1,...]), p0 = [start1,...]
    '''
    if fitrange:
        x_min, x_max = fitrange[0], fitrange[1]
        x, y = x[x > x_min], y[x > x_min]
        x, y = x[x < x_max], y[x < x_max]
    inds, values = fix[0], fix[1]
    fn = wrf(func, inds, values)
    popt, perr = curve_fit(fn, x, y, **kwargs)
    popt = np.array(insert_in_list(list(popt), inds, values))
    return popt, perr

