#!/usr/bin/env python
#coding=gbk

import numpy as np
import os
import sys

def srt_col ( ar , nc , cp = 1 ):
    idx = ar [:, nc ].argsort ()
    if ( cp == 1 ):
        srt = ar [ idx ].copy ()
    else:
        srt = ar [ idx ]
    return srt ;


'''
fn='s'
with open(fn, 'w') as f:
    #data = f.read()
    f.write(data)

with open('outfile.txt', 'w') as f2, open('sample.txt', 'r') as f1:
    data = f1.read()
    f2.write(data)

print(f"Name of the script      : {sys.argv[0]=}")
print(f"Arguments of the script : {sys.argv[1:]=}")

fn=r'./.txt'
#a = np.loadtxt( fn , skiprows=0)
#a = np.loadtxt( fn ,delimiter= '\t' , skiprows=0)
#a = np.loadtxt( fn ,delimiter= '\t' , skiprows=0, dtype='int')

### if use the "space or tap as delimiter,  the use "delimiter=None" !!
np.loadtxt(fname, dtype=<class 'float'>,
comments='#', delimiter=None, converters=None,
skiprows=0, usecols=None, unpack=False, ndmin=0,
encoding='bytes', max_rows=None, *, like=None)

np.savetxt( fn , a , fmt='%.6f',  delimiter= '\t', header='', comments='#')
np.savetxt(fname, X, fmt='%.18e', delimiter=' ', newline='\n',
header='', footer='', comments='# ', encoding=None)

##sort by col
def srt_col(ar, nc, cp=1):
    idx=ar[:, nc].argsort()
    if(cp==1):
        srt=ar[idx].copy()
    else:                
        srt=ar[idx]
    return srt;


## Z is the spanwise(homogenous direction),so average in Z
## at last, the all'e elments is same Z, and various Y
def proc_dt(dt, nz=71, idx_z=2, idx_y=1):
    ##sort by z-coordinate
    ## idx_z is the col of Z
    dt_s=srt_col(dt, idx_z)
    all=np.vsplit(dt_s, nz)
    res=np.zeros(all[0].shape)
    b=[]
    for i in range(len(all)):
        ##sort by y-coordinate,  
        tmp=srt_col(all[i], idx_y)
        b.append(tmp)
        res+=tmp
    all=b
    res/=len(all)
    return all, res


## plot contourf
fig , ax = plt.subplots ( figsize =( 40 , 40 ), dpi = 80 )
cp=ax.contourf(Z,Y,ux,cmap='RdYlBu')
fig.colorbar(cp) # Add a colorbar to a plot
ax.set_title(' Contours ')
ax.set_xlabel('z (m)')
ax.set_ylabel('y (m)')


a = np.arange ( 10 )
np.set_printoptions ( precision = None , threshold = sys.maxsize , suppress = True )

x = np.linspace ( 0 , 2 * np.pi , 10 )
y = np.sin ( x )
xvals = np.linspace ( 0 , 2 * np.pi , 50 )
yinterp = np.interp ( xvals , x , y )

x = np.arange ( 10 )
y = 3 * x ** 2 + 2 * x
deg = 2
p1 = np.poly1d ( np.polyfit ( x , y , deg ))
print ( p1 )

a = sys.maxsize
print(f'{a:E}' )



a=np.fromstring( '1, 2, 3, 4 , 5' , dtype=int, sep= ',' )
a=np.c_[a, a]
print ( a )

x = np.linspace ( 0 , 2 * np.pi , 10 )
y = np.sin ( x )
xvals = np.linspace ( 0 , 2 * np.pi , 50 )
yinterp = np.interp ( xvals , x , y )


cmd=" ls Prf > r;sed -i -e's/Inlet_prf_//' -e 's/.csv//' r"
os.system(cmd)
print(f"Name of the script      : {sys.argv[0]=}")


import sys
import multiprocessing
def func1():
    print('func3')

def func2(s='oo'):
    print(s)

processes = [
    multiprocessing.Process(target=func1),
    multiprocessing.Process(target=func2, args=['helo']),
]
for p in processes:
    p.start()

for p in processes:
    p.join()

'''




'''
import numpy as np
import matplotlib.pyplot as plt
from scipy.optimize import curve_fit
xdata = [ -10.0, -9.0, -8.0, -7.0, -6.0, -5.0, -4.0, -3.0, -2.0, -1.0, 0.0, 1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0, 9.0, 10.0]
ydata = [1.2, 4.2, 6.7, 8.3, 10.6, 11.7, 13.5, 14.5, 15.7, 16.1, 16.6, 16.0, 15.4, 14.4, 14.2, 12.7, 10.3, 8.6, 6.1, 3.9, 2.1]
#Recast xdata and ydata into numpy arrays so we can use their handy features
xdata = np.asarray(xdata)
ydata = np.asarray(ydata)
plt.plot(xdata, ydata, 'o')
def Gauss(x, A, B):
    y = A*np.exp(-1*B*x**2)
    return y
parameters, covariance = curve_fit(Gauss, xdata, ydata)
fit_A = parameters[0]
fit_B = parameters[1]
print(fit_A)
print(fit_B)
fit_y = Gauss(xdata, fit_A, fit_B)
plt.plot(xdata, ydata, 'o', label='data')
plt.plot(xdata, fit_y, '-', label='fit')
plt.legend()
'''
