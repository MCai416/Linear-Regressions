# -*- coding: utf-8 -*-
"""
Created on Thu Jun  4 16:48:27 2020

@author: Ming Cai
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt 
from LinRegModule import OLS 
import scipy.stats as ss

#MA1 setup Test sequence 

dflen = 1000

np.random.seed(2020)
e = np.random.normal(scale = 81, size = dflen+1)
beta = 0.356
x = e[1:] + beta*e[0:dflen] 

#df = pd.DataFrame(x, columns = ['Series 1'])

#df.to_csv('Series1.csv') 

def lag(series1, ind): 
    try: 
        intd = np.int(ind) 
        index = np.arange(1, intd+1)
    except: 
        index = np.sort(np.unique(np.array(ind, dtype = int)))
    #print(self.index)
    leni = index.shape[0]
    lags = series1.shift(index[0]) 
    lagcol = ['L%d'%(index[0])]
    for j in range(1, leni):    
        lagj = series1.shift(index[j])
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
            self.out = OLS(self.series[series.columns[0]], self.X, nocons = True) 
        elif self.p<=0 and self.q <= 0:
            print('p or q must be greater than 0!')
        elif self.p>=series.shape[0] or self.q >= series.shape[0]:
            print('Lags cannot be larger than length of series!')
        else: #AR(infinity)+2step estimation 
            p1 = np.max([1, np.min([np.sqrt(self.series.shape[0]),20])]) 
            self.lagarp1x = lag(self.series, p1)
            y1 = pd.Series(self.series.values.transpose()[0], index = self.series.index, name = self.series.columns[0])
            self.estARP = OLS(y1, self.lagarp1x, nocons = True) 
            self.res = pd.DataFrame(self.estARP.u_hat, columns = ['Residuals'])
            if self.p == 0: 
                self.lagmaqx = lag(self.res, self.q) 
                Xlen = self.lagmaqx.shape[0] 
                ylen = self.series.shape[0] 
                nans = np.ndarray([ylen-Xlen,q])*np.nan
                dfnan = pd.DataFrame(nans, columns = self.lagmaqx.columns)
                self.X = self.lagmaqx.append(dfnan, ignore_index = True)
                self.out = OLS(y1, self.X, nocons = True) 
            else:
                temparpx = lag(self.series, self.p)
                self.lagarpx = pd.DataFrame(temparpx.values, columns = temparpx.columns, index = np.arange(0, temparpx.shape[0]))
                self.lagmaqx = lag(self.res, self.q)
                Xlen = self.lagmaqx.shape[0] 
                ylen = self.series.shape[0] 
                nans = np.ndarray([ylen-Xlen,q])*np.nan
                dfnan = pd.DataFrame(nans, columns = self.lagmaqx.columns)
                self.X = self.lagmaqx.append(dfnan, ignore_index = True)
                self.X = pd.concat([self.lagarpx, self.X], axis = 1)
                self.out = OLS(y1, self.X, nocons = True) 

class CORT(object): #Cochrane - Orcutt 
    def __init__(self, y, X, p = 1, n_iter = 1, vce = 'robust', nocons = False): 
        self.y = y 
        self.X = pd.DataFrame(X.values, index = X.index) 
        self.nocons = nocons
        self.out = OLS(self.y, self.X, vce = vce, nocons = nocons) 
        try:
            k = self.X.shape[1] 
        except: 
            k = 1 
        for j in range(n_iter): 
            self.res = pd.DataFrame(self.out.u_hat, columns = ['Residuals'])
            if p > 1:
                self.ARp = ARMA(self.res, p, 0) 
            else: 
                self.ARp = ARMA(self.res, 1, 0) 
            self.rho = self.ARp.out.b 
            if p > 1:
                self.y_lag = lag(self.y, p).values @ self.rho 
            else: 
                self.y_lag = lag(self.y, 1).values * self.rho[0]
            self.diffy = pd.Series(self.y.values - self.y_lag, index = self.y.index, name = self.y.name) 
            #print("Iteration: {0}, rho = {1:.4f}".format(j+1, self.rho[0]))
            # if t < 1.96 break 
            if p > 1: 
                X_lag = lag(self.X.iloc[:,0], p).values @ self.rho 
            else: 
                X_lag = lag(self.X.iloc[:,0], 1).values * self.rho[0]      
            self.diffX = pd.DataFrame(self.X.values[:,0] - X_lag, index = self.X.index, columns = [self.X.columns[0]]) 
            if k > 1: 
                for i in range(1, k): 
                    if p > 1:
                        X_lag = lag(self.X.iloc[:,i], p).values @ self.rho 
                    else: 
                        X_lag = lag(self.X.iloc[:,i], 1).values * self.rho[0]
                    self.diffX = pd.concat([self.diffX, pd.DataFrame(self.X.values[:,i] - X_lag, index = self.X.index, columns = [self.X.columns[i]])], axis = 1)
            self.out = OLS(self.diffy, self.diffX, vce = vce, nocons = nocons) 
    def sumstat(self): 
        self.u = self.y.values - self.X.values @ self.out.b[:self.out.l-1] - self.out.b[self.out.l-1]/(1-np.sum(self.rho))
        self.u2 = self.u**2 
        self.SSR = np.sum(self.u2) 
        self.SE = self.SSR/float(self.out.df) 
        if self.nocons == True: 
            self.c = self.out.b.reshape(self.out.l, 1)
            self.vartest = self.Varb 
            self.TSS = self.y @ self.y 
        if self.nocons == False: 
            self.c = self.out.b[:self.out.q].reshape(self.out.q, 1)
            self.vartest = self.out.Varb[:self.out.q, :self.out.q] 
            self.TSS = (self.y - np.mean(self.y)) @ (self.y - np.mean(self.y))
        self.R2 = 1 - self.SSR/self.TSS 
        self.AR2 = 1 - (1-self.R2)*float(self.out.n-1)/float(self.out.df)
        self.fstat = (self.c.transpose() @ np.linalg.inv(self.vartest) @ self.c)[0,0]
        self.pval = 1 - ss.f.cdf(self.fstat, self.out.q, self.out.df)
        print("-----------------------")
        print("Summary Statistics")
        print("-----------------------")
        print("Sum rho = {:.4f}".format(np.sum(self.rho)))
        print("SSR = {0:4.2f} \nSE = {1:.4f} \nR-sq = {2:.4g} \nAdj. R-sq = {3:.4g}".format(float(self.SSR), 
              float(self.SE), 
              float(self.R2), 
              float(self.AR2)))
        print("F-statistic = {0:.4f}".format(float(self.fstat)))
        if self.pval < 0.0001:
            print("F: P-value < 0.0001")
        else:
            print("F: P-value = %.4f"%(float(self.pval)))
        print("-----------------------")
        
class VAR(object):
    def __init__(self,series,p):  
        self.len = series.shape[1] 
        self.series = series 
        self.p = p        

"""
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
"""