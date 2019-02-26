# -*- coding: utf-8 -*-
"""
Created on %(date)s

@author: %(username)s
"""

import numpy as np
import re

diamagnetic_correction = {
        "Ag1+": -28, "Ir4+": -29, "Rh4+": -18, "Ag2+": -24, "Ir5+": -20, "Ru3+": -23, "Al3+": -2, 
        "K1+": -14.9, "Ru4+": -18, "As3+": -9, "La3+": -20, "S4+": -3,"As5+":-6, "Li1+":-1.0, 
        "S6+":-1, "Au1+":-40, "Lu3+":-17, "Sb3+":-17, "Au3+":-32, "Mg2+":-5.0, "Sb5+":-14, "B3+":-0.2, 
        "Mn2+":-14, "Sc3+":-6, "Ba2+":-26.5, "Mn3+":-10, "Se4+":-8, "Be2+":-0.4, "Mn4+":-8, "Se6+":-5, 
        "Bi3+":-25, "Mn6+":-4, "Si4+":-1, "Bi5+":-23, "Mn7+":-3, "Sm2+":-23, "Br5+":-6, "Mo2+":-31, "Sm3+":-20, 
        "C4+":-0.1, "Mo3+":-23, "Sn2+":-20, "Ca2+":-10.4, "Mo4+":-17, "Sn4+":-16, "Cd2+":-24, "Mo5+":-12, "Sr2+":-19.0, 
        "Ce3+":-20, "Mo6+":-7, "Ta5+":-14, "Ce4+":-17, "N5+":-0.1, "Tb3+":-19, "Cl5+":-2, "NH4+":-13.3, "Tb4+":-17, 
        "Co2+":-12, "Te4+":-14, "Co3+":-10, "Te6+":-12, "Cr2+":-15, "Na1+":-6.8, "Th4+":-23, "Cr3+":-11, "Nb5+":-9, 
        "Ti3+":-9, "Cr4+":-8, "Nd3+":-20, "Ti4+":-5, "Cr5+":-5, "Ni2+":-12, "Tl1+":-35.7, "Cr6+":-3, "Os2+":-44, "Tl3+":-31, 
        "Cs1+":-35.0, "Os3+":-36, "Tm3+":-18, "Cu1+":-12, "Os4+":-29, "U3+":-46, "Cu2+":-11, "Os6+":-18, "U4+":-35, 
        "Dy3+":-19, "Os8+":-11, "U5+":-26, "Er3+":-18, "P3+":-4, "U6+":-19, "Eu2+":-22, "P5+":-1, "V2+":-15, "Eu3+":-20, 
        "Pb2+":-32.0, "V3+":-10, "Fe2+":-13, "Pb4+":-26, "V4+":-7, "Fe3+":-10, "Pd2+":-25, "V5+":-4, "Ga3+":-8, "Pd4+":-18, 
        "VO2+":-12.5, "Ge4+":-7, "Pm3+":-27, "W2+":-41, "Gd3+":-20, "Pr3+":-20, "W3+":-36, "H1+":0, "Pr4+":-18, "W4+":-23, 
        "Hf4+":-16, "Pt2+":-40, "W5+":-19, "Hg2+":-40.0, "Pt3+":-33, "W6+":-13, "Ho3+":-19, "Pt4+":-28, "Y3+":-12, "I5+":-12, 
        "Rb1+":-22.5, "Yb2+":-20, "I7+":-10, "Re3+":-36, "Yb3+":-18, "In3+ ":-19, "Re4+":-28, "Zn2+":-15.0, "Ir1+":-50, 
        "Re6+":-16, "Zr4+":-10, "Ir2+":-42, "Re7+":-12, "Ir3+":-35, "Rh3+":-22,
        
        "O2-": -12, "S2-": -30, "I1-": -50.6
        }



def get_diamag_corr(s):
    '''
    return diamganetic correction for atom cores in emu/mol.
    chemical formula must include valency of atom and number of atoms (1 must explicitely noted as well):\n 
    * **Cd2Ru2O7** must be writen as **'Cc2+2Ru4+2O2-7'**:\t Ru5+ is not included in table thats why Ru4+ needs to be taken.\n
    * **Li2IrO3** must be written as **'Li1+2Ir4+1O2-3'**
    at the moment this function only works when number of atom in chemical formula is smaller than 10 (e.g. Rb1+C60 wouldnt work)
    '''
    a=re.findall('[A-Z][^A-Z]*', s)
    corr_tot = 0
    for i in a:
        value = re.findall('[^A-Z][^1-9]', i)
        atom = re.findall('[A-Z][^1-9]*', i)
        c= atom[0] + value[0]
        n = float(re.findall('[\+-][1-9]', i)[0][-1])
        print c, n
        corr = n * diamagnetic_correction[c]
        print corr
        corr_tot += corr
    return 1e-6*corr_tot

