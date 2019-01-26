# -*- coding: utf-8 -*-
"""
Created on Tue Jan 15 14:43:18 2019

@author: Ming Cai
"""

# -*- coding: utf-8 -*-
"""
Created on Sun Dec 23 11:14:15 2018

@author: Ming Cai
"""

import numpy as np
from numpy.linalg import inv
from numpy import matmul as m
from numpy import transpose as t
import matplotlib as mpl
import scipy.stats as ss
import pandas as pd

mpl.rcParams['font.size'] = 30
pd.options.display.float_format = '{:,.4g}'.format

#####################################################################
def clearNaN(DataFrameY, DataFrameX):
    l = len(DataFrameY)
    for i in range(l):
        drop = False
        if pd.isnull(np.any(DataFrameY.iloc[l-i-1])):
            DataFrameX = DataFrameX.drop([l-i-1])
            DataFrameY = DataFrameY.drop([l-i-1])
        for j in range(DataFrameX.shape[1]):
            if pd.isnull(DataFrameX.iloc[l-i-1,j]):
                drop = True
        if drop == True:
            DataFrameX = DataFrameX.drop([l-i-1])
            DataFrameY = DataFrameY.drop([l-i-1])
    return DataFrameY, DataFrameX

def lag(df, l, name = None):
    X = np.array(df.values, dtype = float)
    ldf = np.roll(X, l, 0)
    for i in range(l):
        ldf[i] = np.NaN
    return pd.DataFrame(ldf, columns = [name])

class OLS(object):
    def __init__(self, y, X, nocons = False, vce = None, cluster = None):
        y, X = clearNaN(y, X)
        self.depname = y.name
        self.nocons = nocons
        self.X0 = X
        self.n = len(y)
        self.l = X.shape[1]
        self.dep = np.array(y.values, dtype = float)
        self.X = X.values
        self.Xt = t(self.X)
        self.varlist = X.columns
        self.VarX = inv(self.Xt @ self.X)
        self.Px = self.X @ self.VarX @ self.Xt
        self.Mx = np.identity(self.n) - self.Px
        self.CovXy = self.Xt @ self.dep
        self.b = self.VarX @ self.CovXy
        self.df = np.trace(self.Mx)
        self.u_hat = self.dep - m(self.X, self.b)
        self.u1 = self.u_hat.reshape(len(self.u_hat), 1)
        self.u2 = self.u_hat**2
        self.SSR = t(self.u_hat) @ self.u_hat
        self.SE = self.SSR/float(self.df)
        self.Varb = self.SE * self.VarX #default
        if vce == None:
            vce = ""
        if vce.upper() == "ROBUST":
            self.u2 = self.u2*self.n/self.df
            self.ohm = np.zeros([self.n,self.n])
            for i in range(self.n):
               self.ohm[i][i] = self.u2[i]
            self.XOX = self.Xt @ self.ohm @ self.X
            self.Varb = self.VarX @ self.XOX @ self.VarX
        if vce.upper() == "HC2":
            self.u2 = self.u2/np.diag(self.Mx)
            self.ohm = np.zeros([self.n,self.n])
            for i in range(self.n):
               self.ohm[i][i] = self.u2[i]
            self.XOX = self.Xt @ self.ohm @ self.X
            if np.all(cluster) != None:
                for i in range(self.l):
                    for j in range(self.l):
                        if cluster.iloc[i] != cluster.iloc[j]:
                            self.XOX[i][j] = 0
            self.Varb = self.VarX @ self.XOX @ self.VarX
        if np.any(cluster) != None:
            ncl = len(np.unique(cluster))
            self.ohm = self.u1 @ t(self.u1)*ncl/(ncl-1)*(self.n-1)/self.df
            if np.all(cluster) != None:
                print("Cluster ID Retrieved!")
                for i in range(self.n):
                    for j in range(self.n):
                        if cluster.iloc[i] != cluster.iloc[j]:
                            self.ohm[i][j] = 0
            else:
                print("Incomplete Cluster ID!")
                print("HC1 Assumed")
                self.u2 = self.u2*self.n/self.df
                self.ohm = np.zeros([self.n,self.n])
                for i in range(self.n):
                    self.ohm[i][i] = self.u2[i]
            self.XOX = self.Xt @ self.ohm @ self.X
            self.Varb = self.VarX @ self.XOX @ self.VarX
        self.SEb = np.sqrt(np.diag(self.Varb))
        if nocons == False:
            self.ypred = m(self.X, self.b) - np.mean(self.dep)
            self.ESS = t(self.ypred) @ self.ypred
        if nocons == True:
            self.ESS = t(m(self.X, self.b)) @ m(self.X, self.b)
        self.TSS = self.SSR + self.ESS
        self.R2 = self.ESS/self.TSS
        self.AR2 = 1 - (1-self.R2)*float(self.n-1)/float(self.df)
        self.ts = np.zeros(len(self.b))
        self.pvalue = np.zeros(len(self.b))
        for j in range(len(self.b)):
            self.ts[j] = self.b[j]/self.SEb[j]
            self.pvalue[j] = 2*ss.t.cdf(-abs(self.ts[j]), self.df)
        self.Mx = (np.identity(self.n) - self.Px)
        self.SSR1 = t(self.dep) @ self.Mx @ self.dep
    def ftest(self, *args):
        #test under homoskedasticity
        self.p = len(args)
        self.dfX1 = self.X0
        self.uSSR = self.SSR
        if len(args) == 0:
            self.dfX1 = pd.DataFrame(np.ones(self.n))
        else:
            for i in args:
                self.dfX1 = self.dfX1.drop(columns = i)
        self.X1 = self.dfX1.values
        CovX1y = t(self.X1) @ self.dep
        VarX1 = inv(t(self.X1) @ self.X1)    
        self.b1 = VarX1 @ CovX1y
        self.u1_hat = self.dep - (self.X1 @ self.b1)
        self.q = float(self.l-len(args)-1)
        if len(args) == 0 and self.nocons == True:
            self.u1_hat = self.dep
            self.q = float(self.l)
        self.rSSR = t(self.u1_hat) @ self.u1_hat
        if len(args) == 0:
            fstat = ((self.rSSR - self.uSSR)/self.q)/(self.uSSR/float(self.df))
        else:
            fstat = ((self.rSSR - self.uSSR)/float(len(args)))/(self.uSSR/float(self.df))
        Pvalue = 1 - ss.f.cdf(fstat, self.l-len(args), self.df)
        return fstat, Pvalue
    def reg(self):
        fstat, pval = self.ftest()
        self.t95 = ss.t.ppf(0.975, self.df)
        self.mout = np.ndarray([len(self.b),5])
        for j in range(len(self.b)):
            self.mout[j, 0] = self.b[j]
            self.mout[j, 2] = self.ts[j]
            self.mout[j, 3] = self.pvalue[j]
            #self.mout[j, 4] = self.b[j] - self.t95*self.SEb[j]
            #self.mout[j, 5] = self.b[j] + self.t95*self.SEb[j]
            self.mout[j, 1] = self.SEb[j]
            if self.pvalue[j] < 0.001:
                self.mout[j, 4] = 0.001
            if self.pvalue[j] < 0.01:
                self.mout[j, 4] = 0.01
            if self.pvalue[j] < 0.05:
                self.mout[j, 4] = 0.05
            else: 
                self.mout[j, 4] = 0
        self.out = pd.DataFrame(data = self.mout, 
                                index = self.varlist,
                                columns = ['Coefficient', 'S.E', 't', 'p-value', 'Sig'], dtype = float)
        print("-----------------------")
        print("Dependent Variable: {0}".format(self.depname))
        print("-----------------------")
        print("Summary Statistics")
        print("-----------------------")
        print("SSR = {0:4.2f} \nSE = {1:.4g} \nR-sq = {2:.4g} \nAdj. R-sq = {3:.4g}".format(float(self.SSR), 
              float(self.SE), 
              float(self.R2), 
              float(self.AR2)))
        print("F-statistic = {0:.4f}".format(float(fstat)))
        if pval < 0.0001:
            print("F: P-value < 0.0001")
        else:
            print("F: P-value = %.4f"%(float(pval)))
        print("-----------------------")
        print("Coefficients")
        print("-----------------------")
        print(self.out)
        
        
class Ridge(object):
    def __init__(self, y, X, k = 0, nocons = False, vce = None, cluster = None):
        y, X = clearNaN(y, X)
        self.depname = y.name
        self.nocons = nocons
        self.X0 = X
        self.k = k
        self.n = len(y)
        self.l = X.shape[1]
        self.dep = np.array(y.values, dtype = float)
        self.X = X.values
        self.Xt = t(self.X)
        self.varlist = X.columns
        self.VarX = inv(self.Xt @ self.X + self.k * np.identity(self.l))
        self.VarXOLS = inv(self.Xt @ self.X)
        self.Px = self.X @ self.VarX @ self.Xt
        self.Mx = np.identity(self.n) - self.Px
        self.CovXy = self.Xt @ self.dep
        self.b = self.VarX @ self.CovXy
        self.bOLS = self.VarXOLS @ self.CovXy
        self.df = np.trace(self.Mx)
        self.u_hat = self.dep - m(self.X, self.b)
        self.u1 = self.u_hat.reshape(self.n, 1)
        self.u2 = self.u_hat**2
        self.SSR = t(self.u_hat) @ self.u_hat
        self.SE = self.SSR/float(self.df)
        self.Varb = self.SE * self.VarX #default
        if vce == None:
            vce = ""
        if vce.upper() == "ROBUST":
            self.u2 = self.u2*self.n/self.df
            self.ohm = np.zeros([self.n,self.n])
            for i in range(self.n):
               self.ohm[i][i] = self.u2[i]
            self.XOX = self.Xt @ self.ohm @ self.X
            self.Varb = self.VarX @ self.XOX @ self.VarX
        if vce.upper() == "HC2":
            self.u2 = self.u2/np.diag(self.Mx)
            self.ohm = np.zeros([self.n,self.n])
            for i in range(self.n):
               self.ohm[i][i] = self.u2[i]
            self.XOX = self.Xt @ self.ohm @ self.X
            if np.all(cluster) != None:
                for i in range(self.l):
                    for j in range(self.l):
                        if cluster.iloc[i] != cluster.iloc[j]:
                            self.XOX[i][j] = 0
            self.Varb = self.VarX @ self.XOX @ self.VarX
        if np.any(cluster) != None:
            ncl = len(np.unique(cluster))
            self.ohm = self.u1 @ t(self.u1)*ncl/(ncl-1)*(self.n-1)/self.df
            if np.all(cluster) != None:
                print("Cluster ID Retrieved!")
                for i in range(self.n):
                    for j in range(self.n):
                        if cluster.iloc[i] != cluster.iloc[j]:
                            self.ohm[i][j] = 0
            else:
                print("Incomplete Cluster ID!")
                print("HC1 Assumed")
                self.u2 = self.u2*self.n/self.df
                self.ohm = np.zeros([self.n,self.n])
                for i in range(self.n):
                    self.ohm[i][i] = self.u2[i]
            self.XOX = self.Xt @ self.ohm @ self.X
            self.Varb = self.VarX @ self.XOX @ self.VarX
        self.SEb = np.sqrt(np.diag(self.Varb))
        if nocons == False:
            self.ypred = m(self.X, self.b) - np.mean(self.dep)
            self.ESS = t(self.ypred) @ self.ypred
        if nocons == True:
            self.ESS = t(m(self.X, self.b)) @ m(self.X, self.b)
        self.TSS = self.SSR + self.ESS
        self.R2 = self.ESS/self.TSS
        self.AR2 = 1 - (1-self.R2)*float(self.n-1)/float(self.df)
        self.ts = np.zeros(len(self.b))
        self.pvalue = np.zeros(len(self.b))
        for j in range(len(self.b)):
            self.ts[j] = self.b[j]/self.SEb[j]
            self.pvalue[j] = 2*ss.t.cdf(-abs(self.ts[j]), self.df)
        self.Mx = (np.identity(self.n) - self.Px)
        self.SSR1 = t(self.dep) @ self.Mx @ self.dep
    def ftest(self, *args):
        #test under homoskedasticity
        self.p = len(args)
        self.dfX1 = self.X0
        self.uSSR = self.SSR
        if len(args) == 0:
            self.dfX1 = pd.DataFrame(np.ones(self.n))
        else:
            for i in args:
                self.dfX1 = self.dfX1.drop(columns = i)
        self.X1 = self.dfX1.values
        CovX1y = t(self.X1) @ self.dep
        VarX1 = inv(t(self.X1) @ self.X1)    
        self.b1 = VarX1 @ CovX1y
        self.u1_hat = self.dep - (self.X1 @ self.b1)
        self.q = float(self.l-len(args)-1)
        if len(args) == 0 and self.nocons == True:
            self.u1_hat = self.dep
            self.q = float(self.l)
        self.rSSR = t(self.u1_hat) @ self.u1_hat
        if len(args) == 0:
            fstat = ((self.rSSR - self.uSSR)/self.q)/(self.uSSR/float(self.df))
        else:
            fstat = ((self.rSSR - self.uSSR)/float(len(args)))/(self.uSSR/float(self.df))
        Pvalue = 1 - ss.f.cdf(fstat, self.l-len(args), self.df)
        return fstat, Pvalue
    def reg(self):
        fstat, pval = self.ftest()
        self.t95 = ss.t.ppf(0.975, self.df)
        self.mout = np.ndarray([len(self.b),5])
        for j in range(len(self.b)):
            self.mout[j, 0] = self.b[j]
            self.mout[j, 2] = self.ts[j]
            self.mout[j, 3] = self.pvalue[j]
            #self.mout[j, 4] = self.b[j] - self.t95*self.SEb[j]
            #self.mout[j, 5] = self.b[j] + self.t95*self.SEb[j]
            self.mout[j, 1] = self.SEb[j]
            if self.pvalue[j] < 0.001:
                self.mout[j, 4] = 0.001
            if self.pvalue[j] < 0.01:
                self.mout[j, 4] = 0.01
            if self.pvalue[j] < 0.05:
                self.mout[j, 4] = 0.05
            else: 
                self.mout[j, 4] = 0
        self.out = pd.DataFrame(data = self.mout, 
                                index = self.varlist,
                                columns = ['Coefficient', 'S.E', 't', 'p-value', 'Sig'], dtype = float)
        print("-----------------------")
        print("Dependent Variable: {0}".format(self.depname))
        print("-----------------------")
        print("Summary Statistics")
        print("-----------------------")
        print("SSR = {0:4.2f} \nSE = {1:.4g} \nR-sq = {2:.4g} \nAdj. R-sq = {3:.4g}".format(float(self.SSR), 
              float(self.SE), 
              float(self.R2), 
              float(self.AR2)))
        print("-----------------------")
        print("Coefficients")
        print("-----------------------")
        print(self.out)
        
def getK(est):
    #k = est.SE/np.power(est.b, 2)
    k = ss.stats.hmean(est.SE/np.power(est.b,2))
    return k 

def Test(dfy, dfX, dfnull, k=0):
    estu = Ridge(dfy, dfX, k, nocons = False)
    estr = Ridge(dfy, dfnull, k, nocons = False)
    q = estu.l - estr.l
    df = estu.df
    uSSR = estu.SSR
    rSSR = estr.SSR
    F = ((rSSR - uSSR)/q)/(uSSR/df)
    Pvalue = 1 - ss.f.cdf(F, q, df)
    print("F-Test Result: ")
    print("uSSR: {}".format(uSSR))
    print("rSSR: {}".format(rSSR))
    print("F-statistic: {}".format(F))
    print("P-value: {}".format(Pvalue))
