# code by Tim R. Reeder
import numpy as np
import scipy as sp
import struct
import glob
import matplotlib.pyplot as plt
import matplotlib.colors as colors
import time
# from numba import int64, int32, float32, boolean, jitclass, char, njit, jit
import numba
import sys
from astropy.table import Table
from astropy.table import Column
from astropy.io import ascii
import xarray as xr
import pandas as pd
import plotly.express as px


def cosd(x):
    return np.cos(np.deg2rad(x))
def sind(x):
    return np.sin(np.deg2rad(x))
def arcsind(x):
    return np.rad2deg(np.arcsin(x))
def arccosd(x):
    return np.rad2deg(np.arccos(x))
def f_Gstar(lat_pars):
    lengths = lat_pars['lengths']
    angles = lat_pars['angles']
    a=lengths[0]; b=lengths[1]; c=lengths[2]; 
    radangles = np.radians(angles)
    Alpha=radangles[0]; Beta=radangles[1]; Gamma=radangles[2]; #angles converted to radians
    vol = a*b*c*np.sqrt(1-(np.cos(Alpha))**2-(np.cos(Beta))**2-(np.cos(Gamma))**2+2*np.cos(Alpha)*np.cos(Beta)*np.cos(Gamma));
    Gstar = (2*np.pi)**2/(vol**2)*np.array([[b**2*c**2*np.sin(Alpha)**2, a*b*c**2*(np.cos(Alpha)*np.cos(Beta) - np.cos(Gamma)), a*b**2*c*(np.cos(Alpha)*np.cos(Gamma) - np.cos(Beta))], [a*b*c**2*(np.cos(Alpha)*np.cos(Beta) - np.cos(Gamma)), a**2*c**2*np.sin(Beta)**2, a**2*b*c*(np.cos(Beta)*np.cos(Gamma) - np.cos(Alpha))], [a*b**2*c*(np.cos(Alpha)*np.cos(Gamma) - np.cos(Beta)), a**2*b*c*(np.cos(Beta)*np.cos(Gamma) - np.cos(Alpha)), a**2*b**2*np.sin(Gamma)**2]]);
    #         # Gstar is the reciprocal space metric tensor
    recip = 2*np.pi*np.array([b*c*np.sin(Alpha)/vol,a*c*np.sin(Beta)/vol,a*b*np.sin(Gamma)/vol]) # in inverse angstroms
    return Gstar,recip,vol

def f_norm(u,lat_pars):
    Gstar,recip,vol = f_Gstar(lat_pars)
    return u*recip/np.sqrt(np.matmul(np.transpose(u),np.matmul(Gstar,u)))

def f_ki(Ei): # in inverse angstroms
    f = 8.06561E-5
    return Ei*f*2*np.pi
    
def f_full_tthth(step=1):
    tths=np.arange(90, 150+1,step)
    tth = []; th = []
    for i,t in enumerate(tths):
        newth = np.arange(0,t,step)
        th = np.append(th,newth)
        tth = np.append(tth,np.repeat(t,len(newth)))
    return tth,th
    
def f_coincident_tthth(Ei,tth_short,th_short,u,v,lat_pars,hkl='h'):
    d_hkl = {'h':0,'k':1,'l':2}
    ki = f_ki(Ei)
    th = []
    tth0 = np.repeat(tth_short[0],len(th_short))
    for i,t in enumerate(th_short):
        Q = f_Q(Ei,tth0,th_short,u,v,lat_pars)
    for ii,tt in enumerate(tth_short):
        tempth = np.sort(-(arcsind(Q[:,d_hkl[hkl]]/ki/(-2)/sind(-tt/2))-tt/2))
        th = np.append(th,tempth)
    tth = np.repeat(tth_short,len(th_short))
    return tth,th
    
def f_Q(Ei,tth,th,u,v,lat_pars): # u is in the sample holder plane, v is perpindicular. lengths and angles are the lattice parameters
    ki = f_ki(Ei)
    uhat = f_norm(u,lat_pars)
    vhat = f_norm(v,lat_pars)
    ki_recip = (np.expand_dims(uhat,1)*cosd(-th)+np.expand_dims(vhat,1)*sind(-th))*ki
    kf_recip = (np.expand_dims(uhat,1)*cosd(tth-th)+np.expand_dims(vhat,1)*sind(tth-th))*ki
    Q = np.transpose(ki_recip-kf_recip)
    return Q # in inverse angstrom
    
def f_th_tth(Ei,Q_rlu,recip): # this function isn't being used
    lQ = len(Q_rlu)
    ki = f_ki(Ei)
    print('Max |Q| = %.2f (inv ang) = %.2f[H,0,0] (r.l.u.)' % (2*ki, 2*ki/recip[0]))
    Q_ang = Q_rlu*2*np.pi*recip
    Q_norm = np.zeros(lQ)
    for i in range(lQ):
        Q_norm[i] = np.sqrt(np.matmul(np.transpose(Q_rlu[i]),np.matmul(Gstar,Q_rlu[i])))
    tth = np.arccos(1-Q_norm**2/(2*ki**2))
    # uhat = u*np.diag(Gstar) # in brillouin zone space
    th = -(np.arcsin(Q_ang[:,0]/ki/(-2)/np.sin(-tth/2))-tth/2) #potential bug here
    return th, tth

def f_C(lat_pars,u,v,Ei,tth='',th='',thstep = 10.,coinc=False,hkl='h'):
    # There are three ways to use this function: 
        # 1) leave kwd_args (tth,th,coinc) default: tth and th are chosen equal to the available ranges for SIX
        # 2) input user defined tth and th values. These need to be "long" meaning they need to repeat values to fill up the corresponding 2D tth/th phase-space. The coinc and hkl kwd_args are not used.
        # 3) if coinc = True, there needs to be user defined tth and th, and they should be "short" meaning they should only show unique values. The result: for each user defined tth value, th values are calculated to provide constant cuts along the chosen hkl axis
    C = xr.Dataset()
    if (len(tth)==0):
        tth,th = f_full_tthth(step=thstep)
    elif coinc:
        tth_short = tth; th_short = th;
        tth,th = f_coincident_tthth(Ei,tth_short,th_short,u,v,lat_pars,hkl=hkl)
    lth = len(th); ltth = len(tth);
    # th = np.repeat([th],ltth,axis=0).flatten()    
    # tth = np.repeat(tth,lth)
    C = C.assign_coords(index = np.arange(lth))
    Gstar,recip,vol = f_Gstar(lat_pars)
    C = C.assign({'Gstar': (('anginv','anginv'),Gstar),
                  'recip':  ('anginv',recip),
                  'vol':    ('Ei',[vol]),
                  'Ei':     ('Ei',[Ei])})
    uhat = f_norm(u,lat_pars)
    vhat = f_norm(v,lat_pars)
    C = C.assign({'u':(('HKO'),u),
                  'v':(('HKO'),v),
                  'uhat':(('anginv'),uhat),
                  'vhat':(('anginv'),vhat)})
    C = C.assign({'tth': ('index',tth),
                  'th':  ('index',th)})
    C['|ki|'] = f_ki(C['Ei'])
    C['ki'] = (cosd(-C['th'])*C['uhat']+sind(-C['th'])*C['vhat'])*C['|ki|']
    C['kf'] = (cosd(C['tth']-C['th'])*C['uhat']+sind(C['tth']-C['th'])*C['vhat'])*C['|ki|']
    C['Q'] = C['ki']-C['kf']
    # C = C.assign({'Q_rlu':(('index','HKO','Ei'),C['Q']/C['recip'])})
    C['Q_rlu'] = C['Q']/C['recip']
    C = C.assign({'|Q|':(('index','Ei'),np.linalg.norm(C['Q'],axis=1))}).round(3)
    Ca = C['Q'].groupby('anginv')
    Ch = C['Q_rlu'].groupby('anginv')
    qq = ['Qx','Qy','Qz']; hkl = ['H','K','L'];
    for a, (q,h,(caname, ca)) in enumerate(zip(qq,hkl,Ca)):
        C[q] = ca.round(3)
    for a, (q,h,(chname, ch)) in enumerate(zip(qq,hkl,Ch)):
        C[h] = ch.round(3)
    Cdf = C.squeeze('Ei').drop_vars(['|ki|','vol','Ei']).drop_dims(['anginv','HKO']).to_dataframe()
    return C,Cdf
def printeverythingdf(df):
    with pd.option_context('display.max_rows', None, 'display.max_columns', None, 'display.width', 1000):
        print(df)
def f_th_tth(Ei,Q_rlu,recip):
    lQ = len(Q_rlu)
    ki = f_ki(Ei)
    print('Max |Q| = %.2f (inv ang) = %.2f[H,0,0] (r.l.u.)' % (2*ki, 2*ki/recip[0]))
    Q_ang = Q_rlu*2*np.pi*recip
    Q_norm = np.zeros(lQ)
    for i in range(lQ):
        Q_norm[i] = np.sqrt(np.matmul(np.transpose(Q_rlu[i]),np.matmul(Gstar,Q_rlu[i])))
    
    tth = np.arccos(1-Q_norm**2/(2*ki**2))
    # uhat = u*np.diag(Gstar) # in brillouin zone space
    th = -(arcsind(Q_ang[:,0]/ki/(-2)/sind(-tth/2))-tth/2) #potential bug here
    return th, tth