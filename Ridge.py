# -*- coding: utf-8 -*-
"""
Created on Tue Jan 15 14:43:18 2019

@author: Ming Cai
"""

import numpy as np
from numpy.linalg import inv
from numpy import matmul as m
from numpy import transpose as t
import matplotlib as mpl
import scipy.stats as ss
import pandas as pd
from numpy.linalg import eigh

mpl.rcParams['font.size'] = 30
pd.options.display.float_format = '{:,.4g}'.format

#####################################################################
def clearNaN(DataFrameY, DataFrameX): # to clear out NaN variables, which are invalid observations
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
        y, X = clearNaN(y, X) # get rid of null obs
        self.depname = y.name 
        self.nocons = nocons 
        self.n = len(y) 
        self.l = X.shape[1]
        self.dep = np.array(y.values, dtype = float)
        self.X = X.values
        self.Xt = t(self.X)
        self.varlist = X.columns
        self.VarX = inv(self.Xt @ self.X) # the main inverse
        self.Px = self.X @ self.VarX @ self.Xt # Px matrix from Davidson Mackinnon
        self.Mx = np.identity(self.n) - self.Px # Mx matrix ""
        self.CovXy = self.Xt @ self.dep # Xty
        self.b = self.VarX @ self.CovXy # combining the inverse and the Xty
        self.df = np.trace(self.Mx) # degrees of freedom = trace(Mx)
        self.u_hat = self.dep - m(self.X, self.b) # get residuals
        self.u1 = self.u_hat.reshape(len(self.u_hat), 1) # data organisation
        self.u2 = self.u_hat**2 # get squared residuals 
        self.SSR = t(self.u_hat) @ self.u_hat # get SSR
        self.SE = self.SSR/float(self.df) # get sigma_u estimate
        self.Varb = self.SE * self.VarX #default 
        #Heteroskedasticity/Clustered/ Serial Correlation works below 
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
            self.Varb = self.VarX @ self.XOX @ self.VarX
        if np.any(cluster) != None:
            if np.all(cluster) != None:
                print("Cluster ID Retrieved!")
                if cluster.shape[1] == 1:
                    ncl = len(np.unique(cluster))
                    self.ohm = self.u1 @ t(self.u1)*ncl/(ncl-1)*(self.n-1)/self.df
                    for i in range(self.n):
                        for j in range(self.n):
                            if np.all(cluster.iloc[i] != cluster.iloc[j]):
                                self.ohm[i][j] = 0
                elif cluster.shape[1] == 2:
                    print("Twoway clustering: ", cluster.columns)
                    ncl1 = len(np.unique(cluster.iloc[:,0]))
                    ncl2 = len(np.unique(cluster.iloc[:,1]))
                    ncl12 = len(np.unique(cluster, axis = 0))
                    self.ohm1 = self.u1 @ t(self.u1)*ncl1/(ncl1-1)*(self.n-1)/self.df
                    self.ohm2 = self.u1 @ t(self.u1)*ncl2/(ncl2-1)*(self.n-1)/self.df
                    self.ohm12 = self.u1 @ t(self.u1)*ncl12/(ncl12-1)*(self.n-1)/self.df
                    print("Retrieving First VCE")
                    for i in range(self.n):
                        for j in range(self.n):
                            if cluster.iloc[i,0] != cluster.iloc[j,0]:
                                self.ohm1[i][j] = 0
                    print("Retrieving Second VCE")
                    for i in range(self.n):
                        for j in range(self.n):
                            if cluster.iloc[i,1] != cluster.iloc[j,1]:
                                self.ohm2[i][j] = 0
                    print("Retrieving Third VCE")
                    if ncl12 == self.n:
                        d1 = np.diag(self.ohm12)
                        self.ohm12 = d1 * np.identity(self.n)
                    else:
                        for i in range(self.n):
                            for j in range(self.n):
                                if np.any(cluster.iloc[i] != cluster.iloc[j]):
                                    self.ohm12[i][j] = 0
                    self.ohm = self.ohm1 + self.ohm2 - self.ohm12
                else: 
                    print("Supports up to two way clustering only!")
            else:
                print("Incomplete Cluster ID!")
                print("HC1 Assumed")
                self.u2 = self.u2*self.n/self.df
                self.ohm = np.zeros([self.n,self.n])
                for i in range(self.n):
                    self.ohm[i][i] = self.u2[i]
            self.XOX = self.Xt @ self.ohm @ self.X
            self.Varb = self.VarX @ self.XOX @ self.VarX
            if np.any(np.diag(self.Varb) < 0):
                print("Non Positive Semi-Definite VCE Matrix! Cameron, Gelbach & Miller (2011) Transformation Used")
                lb, vb = eigh(self.Varb)
                idx = lb.argsort()[::-1]
                vb = vb[:,idx]
                lb = lb[idx]
                for i in range(len(lb)):
                    lb[i] = max(0, lb[i])
                diag = lb * np.identity(len(lb))
                self.Varb = vb @ diag @ t(vb)
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
        self.settest()
    def settest(self):
        self.cons = pd.DataFrame(np.ones(self.n))
        self.uSSR = self.SSR
        if self.nocons == False:
            self.depdemean = (self.dep - np.mean(self.dep))
            self.rSSR = np.dot(self.depdemean, self.depdemean)
            self.q = self.l-1
        if self.nocons == True:
            self.rSSR = np.dot(self.dep, self.dep)
            self.q = self.l
        self.fstat = ((self.rSSR - self.uSSR)/self.q)/(self.uSSR/self.df)
        self.pval = 1 - ss.f.cdf(self.fstat, self.q, self.df)
    def reg(self):
        self.t95 = ss.t.ppf(0.975, self.df)
        self.mout = np.ndarray([len(self.b),5])
        for j in range(len(self.b)):
            self.mout[j, 0] = self.b[j]
            self.mout[j, 2] = self.ts[j]
            self.mout[j, 3] = self.pvalue[j]
            #self.mout[j, 4] = self.b[j] - self.t95*self.SEb[j]
            #self.mout[j, 5] = self.b[j] + self.t95*self.SEb[j]
            self.mout[j, 1] = self.SEb[j]
            if self.pvalue[j] < 0.05:
                self.mout[j, 4] = 0.05
                if self.pvalue[j] < 0.01:
                    self.mout[j, 4] = 0.01
                    if self.pvalue[j] < 0.001:
                        self.mout[j, 4] = 0.001
            else: 
                self.mout[j, 4] = None
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
        print("F-statistic = {0:.4f}".format(float(self.fstat)))
        if self.pval < 0.0001:
            print("F: P-value < 0.0001")
        else:
            print("F: P-value = %.4f"%(float(self.pval)))
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
        self.Varb = self.SE * (self.VarX @ self.Xt @ self.X @ self.VarX) #default
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
            if np.all(cluster) != None:
                print("Cluster ID Retrieved!")
                if cluster.shape[1] == 1:
                    ncl = len(np.unique(cluster))
                    self.ohm = self.u1 @ t(self.u1)*ncl/(ncl-1)*(self.n-1)/self.df
                    for i in range(self.n):
                        for j in range(self.n):
                            if np.all(cluster.iloc[i] != cluster.iloc[j]):
                                self.ohm[i][j] = 0
                elif cluster.shape[1] == 2:
                    print("Twoway clustering: ", cluster.columns)
                    ncl1 = len(np.unique(cluster.iloc[:,0]))
                    ncl2 = len(np.unique(cluster.iloc[:,1]))
                    ncl12 = len(np.unique(cluster, axis = 0))
                    self.ohm1 = self.u1 @ t(self.u1)*ncl1/(ncl1-1)*(self.n-1)/self.df
                    self.ohm2 = self.u1 @ t(self.u1)*ncl2/(ncl2-1)*(self.n-1)/self.df
                    self.ohm12 = self.u1 @ t(self.u1)*ncl12/(ncl12-1)*(self.n-1)/self.df
                    print("Retrieving First VCE")
                    for i in range(self.n):
                        for j in range(self.n):
                            if cluster.iloc[i,0] != cluster.iloc[j,0]:
                                self.ohm1[i][j] = 0
                    print("Retrieving Second VCE")
                    for i in range(self.n):
                        for j in range(self.n):
                            if cluster.iloc[i,1] != cluster.iloc[j,1]:
                                self.ohm2[i][j] = 0
                    print("Retrieving Third VCE")
                    if ncl12 == self.n:
                        d1 = np.diag(self.ohm12)
                        self.ohm12 = d1 * np.identity(self.n)
                    else:
                        for i in range(self.n):
                            for j in range(self.n):
                                if np.any(cluster.iloc[i] != cluster.iloc[j]):
                                    self.ohm12[i][j] = 0
                    self.ohm = self.ohm1 + self.ohm2 - self.ohm12
                else: 
                    print("Supports up to two way clustering only!")
            else:
                print("Incomplete Cluster ID!")
                print("HC1 Assumed")
                self.u2 = self.u2*self.n/self.df
                self.ohm = np.zeros([self.n,self.n])
                for i in range(self.n):
                    self.ohm[i][i] = self.u2[i]
            self.XOX = self.Xt @ self.ohm @ self.X
            self.Varb = self.VarX @ self.XOX @ self.VarX
            if np.any(np.diag(self.Varb) < 0):
                print("Non Positive Semi-Definite VCE Matrix! Cameron, Gelbach & Miller (2011) Transformation Used")
                lb, vb = eigh(self.Varb)
                idx = lb.argsort()[::-1]
                vb = vb[:,idx]
                lb = lb[idx]
                for i in range(len(lb)):
                    lb[i] = max(0, lb[i])
                diag = lb * np.identity(len(lb))
                self.Varb = vb @ diag @ t(vb)
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
    def reg(self):
        self.t95 = ss.t.ppf(0.975, self.df)
        self.mout = np.ndarray([len(self.b),5])
        for j in range(len(self.b)):
            self.mout[j, 0] = self.b[j]
            self.mout[j, 2] = self.ts[j]
            self.mout[j, 3] = self.pvalue[j]
            #self.mout[j, 4] = self.b[j] - self.t95*self.SEb[j]
            #self.mout[j, 5] = self.b[j] + self.t95*self.SEb[j]
            self.mout[j, 1] = self.SEb[j]
            if self.pvalue[j] < 0.05:
                self.mout[j, 4] = 0.05
                if self.pvalue[j] < 0.01:
                    self.mout[j, 4] = 0.01
                    if self.pvalue[j] < 0.001:
                        self.mout[j, 4] = 0.001
            else: 
                self.mout[j, 4] = None
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
        try:
            len(self.k)
            print("Length k = {}, Mean k = {}, SD k = {}".format(len(self.k), np.nean(self.k), np.std(self.k)))
        except:
            print("k = {}".format(self.k))
        print("-----------------------")
        print("Coefficients")
        print("-----------------------")
        print(self.out)
        
def getK(est):
    #k = est.SE/np.power(est.b, 2)
    k = ss.stats.hmean(est.SE/np.power(est.b,2))
    return k 

def Test(dfy, dfX, dfnull, nocons = False, k0=0, k1=0, Test0 = False):
    estu = Ridge(dfy, dfX, k1, nocons = nocons)
    estr = Ridge(dfy, dfnull, k0, nocons = nocons)
    b0 = estr.b
    b1 = estu.b
    XtX = estu.Xt @ estu.X
    s2 = estu.SE
    df1 = estu.df
    if Test0 == True:
        q = len(dfy) - estu.df
        Foben = (t(b1) @ XtX @ b1)/(q*s2)
        print(q)
    else:
        q = estr.df - estu.df
        print(l, q)
        l = len(b1) - len(b0)
        if q < 0: 
            print("must have more or equal variables in dfx!")
            pass
        bdiff = np.zeros(len(b1))
        bres = np.zeros(len(b1))
        if q == 0:
            bdiff = b1 - b0
        if q > 0:
            for i in range(len(b1)):
                bdiff[i] = b1[i]
                bres[i] = b1[i]
                if i >= l:
                    bdiff[i] = b1[i] - b0[i-l]
                    bres[i] = 0
        Foben = (t(bdiff) @ XtX @ bdiff)/(q*s2)
    print("Obenchain 1977, F-statistic: {}".format(Foben))
    Pvoben = 1 - ss.f.cdf(Foben, q, df1)
    print("Obenchain 1977, P-value: {}".format(Pvoben))
    uSSR = estu.SSR
    rSSR = estr.SSR
    if nocons == True: 
        rSSR = np.dot(dfy.values, dfy.values)
    F = ((rSSR - uSSR)/q)/(uSSR/df1)
    Pvalue = 1 - ss.f.cdf(F, q, df1)
    print("F-Test Result: ")
    print("uSSR: {}".format(uSSR))
    print("rSSR: {}".format(rSSR))
    print("F-statistic: {}".format(F))
    print("P-value: {}".format(Pvalue))
    #estu.reg()
