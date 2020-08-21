# -*- coding: utf-8 -*-
"""
Created on Thu Jun  4 16:48:27 2020

@author: Ming Cai
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt 
from Ridge import OLS 

#MA1 setup Test sequence 

dflen = 1000

np.random.seed(2020)
e = np.random.normal(scale = 81, size = dflen+1)
beta = 0.356
x = e[1:] + beta*e[0:dflen] 

df = pd.DataFrame(x, columns = ['Series 1'])

#df.to_csv('Series1.csv') 

def lag(series1, ind): 
    try: 
        intd = np.int(ind) 
        index = np.arange(1, intd+1)
    except: 
        index = np.sort(np.unique(np.array(ind, dtype = int)))
    #print(self.index)
    leni = index.shape[0]
    lags = series1.shift(-index[0]) 
    lagcol = ['L%d'%(index[0])]
    for j in range(1, leni):    
        lagj = series1.shift(-index[j])
        lags = pd.concat([lags, lagj], axis = 1)
        lagcol.append('L%d'%(index[j]))
    lags.columns = lagcol
    #print(lagcol)
    return lags 

class ARMA(object):
    def __init__(self, series, p, q):
        if series.shape[1] != 1:
            raise ValueError 
        else: 
            self.series = series 
            self.p = p 
            self.q = q
        #AR(P)
        if self.p>0 and self.q == 0:
            self.X = lag(self.series, self.p)
            self.out = OLS(self.series[series.columns[0]], self.X) 
        elif self.p<=0 and self.q <= 0:
            print('p or q must be greater than 0!')
        elif self.p>=series.shape[0] or self.q >= series.shape[0]:
            print('Lags cannot be larger than length of series!')
        else: #AR(infinity)+2step estimation 
            p1 = np.max([1, np.min([np.sqrt(self.series.shape[0]),20])]) 
            self.lagarp1x = lag(self.series, p1)
            self.estARP = OLS(self.series[df.columns[0]], self.lagarp1x) 
            self.res = pd.DataFrame(self.estARP.u_hat, columns = ['Residuals'])
            if self.p == 0: 
                self.lagmaqx = lag(self.res, self.q) 
                Xlen = self.lagmaqx.shape[0] 
                ylen = self.series.shape[0] 
                nans = np.ndarray([ylen-Xlen,q])*np.nan
                dfnan = pd.DataFrame(nans, columns = self.lagmaqx.columns)
                self.X = self.lagmaqx.append(dfnan, ignore_index = True)
                self.out = OLS(self.series[df.columns[0]], self.X) 
            else:
                self.lagarpx = lag(self.series, self.p) 
                self.lagmaqx = lag(self.res, self.q) 
                self.X = pd.concat([self.lagarpx, self.lagmaqx], axis = 1)
                self.out = OLS(self.series[df.columns[0]], self.X) 
        self.out.reg()

class VAR(object):
    def __init__(self,series,p):  
        self.len = series.shape[1] 
        self.series = series 
        self.p = p        


#Testing Model 
p = 0
q = 5
#a = np.arange(p+1)
#b = np.arange(q+1)

maxlag = np.max([np.max(p),np.max(q)])

est1 = ARMA(df, p, q) 
#Lag = lag(df, 1) 

#Bootstrap Standard Errors 
m = 2*df.shape[0] 
B = 1000 
SCoef = est1.out.b 
seriesb = np.zeros([m+df.shape[0]]) 
scalars = False 

### Scalar p, q only###
if np.isscalar(p): 
    if np.isscalar(q):
        BSCoef = np.zeros([B, p+q])
        scalars = True
        p1 = p 
        q1 = q 
    else:
        BSCoef = np.zeros([B,p+q.shape[0]]) 
        p1 = p
        p = np.arange(1,p1+1)
elif np.isscalar(q):
    BSCoef = np.zeros([B,p.shape[0]+q])
    q1 = q
    q = np.arange(1, q1+1) 
else:
    BSCoef = np.zeros([B,p.shape[0]+q.shape[0]]) 
    p1 = p
    p = np.arange(1,p1+1)
    q1 = q 
    q = np.arange(1, q1+1)

if scalars == True: 
    for b in range(B):
        if np.mod(b+1, 50) == 0:
            print("Bootstrap Sample # %d "%(b+1))
        resb = np.random.choice(est1.res.values.reshape(est1.res.size), size = m+df.shape[0]) 
        for t in range(maxlag, m+df.shape[0]): 
            ARpart = SCoef[:p1] @ seriesb[t-p1:t][::-1] 
            MApart = SCoef[p1:] @ resb[t-q1:t][::-1] 
            seriesb[t] = ARpart + MApart + resb[t] 
        dfy = pd.DataFrame(seriesb[m:m+df.shape[0]][::-1], columns = ['Bootstrap Series']) 
        dfres = pd.DataFrame(resb[m:m+df.shape[0]][::-1],columns = ['Bootstrap Residuals']) 
        if p1 == 0: 
            dfX = lag(dfres, q1) 
        elif q1 == 0: 
            dfX = lag(dfy, p1) 
        else: 
            dfX = pd.concat([lag(dfy,p1), lag(dfres, q1)], axis = 1) 
        BSCoef[b] = OLS(dfy[dfy.columns[0]], dfX).b 
else:
    print("Currently not supporting non-arange lag models") 


BSMean = np.mean(BSCoef, axis = 0) 
BSVar = (BSCoef-BSMean).transpose() @ (BSCoef-BSMean) / df.shape[0] 
BSSE = np.std(BSCoef, axis = 0) 
print("Bootstrap Estimate")
print(np.round(BSMean, decimals = 4)) 
print("Bootstrap Standard Errors")
print(np.round(BSSE, decimals = 4))
print("Bootstrap Estimator Variance-Covariance") 
print(np.round(BSVar, decimals = 4)) 
plt.title("Bootstrap Distribution of MA(1) Coefficient") 
plt.hist(BSCoef[:,0], bins = 50) 
