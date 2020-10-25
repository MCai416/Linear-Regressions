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
from numpy.linalg import eigh

#mpl.rcParams['font.size'] = 30
#pd.options.display.float_format = '{:,.4g}'.format

#####################################################################
def clearNaN(DataFrameY, DataFrameX): # to clear out NaN variables for both X and y, which are invalid observations
    l = len(DataFrameY)
    for i in range(l):
        drop = False
        if pd.isnull(np.any(DataFrameY.iloc[l-i-1])):
            drop = True
        try: 
            for j in range(DataFrameX.shape[1]):
                if pd.isnull(DataFrameX.iloc[l-i-1,j]):
                    drop = True 
        except: 
            if pd.isnull(DataFrameX.iloc[l-i-1]):
                drop = True    
        if drop == True:
            DataFrameX = DataFrameX.drop(DataFrameX.index[l-i-1])
            DataFrameY = DataFrameY.drop(DataFrameY.index[l-i-1])
    return DataFrameY, DataFrameX 

def lag(df, l, name = None): 
    X = np.array(df.values, dtype = float)
    ldf = np.roll(X, l, 0)
    for i in range(l):
        ldf[i] = np.NaN
    return pd.DataFrame(ldf, columns = [name])

class OLS(object):
    def __init__(self, y, X, nocons = False, vce = "ROBUST", cluster = None, gls = False):
        y, X = clearNaN(y, X) # get rid of null obs
        self.gls = gls
        self.depname = y.name 
        self.nocons = nocons 
        self.n = len(y) 
        if nocons == False: 
            cons = pd.Series(np.ones(self.n), index = X.index, name = "Cons")
            X = pd.concat([X, cons], axis = 1)
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
        #Heteroskedasticity/Clustered/Serial Correlation works below 
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
        if np.all(np.any(cluster) != None):
            if np.all(np.all(cluster) != None):
                print("Cluster ID Retrieved!")
                try:
                    clsize = cluster.shape[1] 
                except: 
                    clsize = 1
                if clsize == 1:
                    ncl = len(np.unique(cluster))
                    self.o1 = self.u1 @ t(self.u1)*ncl/(ncl-1)*(self.n-1)/self.df
                    self.ohm = np.zeros(self.o1.shape)
                    for i in range(self.n):
                        for j in range(self.n):
                            if np.all(cluster.iloc[i] == cluster.iloc[j]):
                                self.ohm[i][j] = self.o1[i][j]
                elif clsize == 2:
                    print("Twoway clustering: ", cluster.columns)
                    ncl1 = len(np.unique(cluster.iloc[:,0]))
                    ncl2 = len(np.unique(cluster.iloc[:,1]))
                    ncl12 = len(np.unique(cluster, axis = 0))
                    self.o1 = self.u1 @ t(self.u1)*(self.n-1)/self.df*ncl1/(ncl1-1)
                    self.o2 = self.u1 @ t(self.u1)*(self.n-1)/self.df*ncl2/(ncl2-1)
                    self.o12 = self.u1 @ t(self.u1)*(self.n-1)/self.df*ncl12/(ncl12-1)
                    self.ohm1 = np.zeros(self.o1.shape)
                    self.ohm2 = np.zeros(self.o2.shape)
                    self.ohm12 = np.zeros(self.o12.shape)
                    print("Retrieving First VCE")
                    for i in range(self.n):
                        for j in range(self.n):
                            if cluster.iloc[i,0] == cluster.iloc[j,0]:
                                self.ohm1[i][j] = self.o1[i][j]
                    print("Retrieving Second VCE")
                    for i in range(self.n):
                        for j in range(self.n):
                            if cluster.iloc[i,1] == cluster.iloc[j,1]:
                                self.ohm2[i][j] = self.o2[i][j]
                    print("Retrieving Third VCE")
                    if ncl12 == self.n:
                        d1 = np.diag(self.o12)
                        self.ohm12 = d1 * np.identity(self.n)
                    else:
                        for i in range(self.n):
                            for j in range(self.n):
                                if np.any(cluster.iloc[i] == cluster.iloc[j]):
                                    self.ohm12[i][j] = self.o12[i][j]
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
            self.Varb1 = self.Varb
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
        if self.gls == True: 
            print("One step GLS")
            self.VarX = inv(self.XOX)
            self.CovXy = self.Xt @ self.ohm @ self.dep
            self.b = self.VarX @ self.CovXy
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
    def settest(self): #F-test against the constant as the restricted model  
        self.cons = pd.DataFrame(np.ones(self.n))
        self.uSSR = self.SSR        
        if self.nocons == False:
            self.q = self.l-1
            self.depdemean = (self.dep - np.mean(self.dep))
            self.rSSR = np.dot(self.depdemean, self.depdemean) 
            self.c = self.b[:self.q].reshape(self.q, 1)
            self.vartest = self.Varb[:self.q, :self.q]
        if self.nocons == True:
            self.q = self.l
            self.rSSR = np.dot(self.dep, self.dep) 
            self.c = self.b.reshape(self.l, 1)
            self.vartest = self.Varb
        self.fstat = (self.c.transpose() @ np.linalg.inv(self.vartest) @ self.c)[0,0]
        self.fstat1 = ((self.rSSR - self.uSSR)/self.q)/(self.uSSR/self.df) 
        #print("Old: {:.4f}, New {:.4f}".format(self.fstat, self.fstat1)) 
        self.pval = 1 - ss.f.cdf(self.fstat, self.q, self.df)
    def reg(self): #Output function, not going to output unless if est.reg()
        self.t95 = ss.t.ppf(0.975, self.df)
        self.mout = np.ndarray([len(self.b),5])
        for j in range(len(self.b)):
            self.mout[j, 0] = self.b[j]
            self.mout[j, 2] = self.ts[j]
            self.mout[j, 3] = self.pvalue[j]
            #self.mout[j, 4] = self.b[j] - self.t95*self.SEb[j]
            #self.mout[j, 5] = self.b[j] + self.t95*self.SEb[j]
            self.mout[j, 1] = self.SEb[j] #np.round(self.SEb[j], decimals = 3-np.int(np.floor(np.log10(np.abs(self.SEb[j])))))
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
        print("Coefficients")
        print("-----------------------")
        print(self.out)
        
class PCA(object):
    def __init__(self, dfy, dfX, numeigen = 0, minproportion = 0, vce = None, cluster = None):
        print("PCA Regression:")
        self.vce = vce
        self.cluster = cluster
        self.varlist = dfX.columns
        self.X = dfX.values
        self.Xt = self.X.transpose()
        self.XtX = self.Xt @ self.X
        self.lam, self.v = np.linalg.eigh(self.XtX)
        idx = self.lam.argsort()[::-1]
        self.v = self.v[:,idx]
        self.lam = self.lam[idx]
        self.lp = self.lam/sum(self.lam)
        self.dfA = pd.DataFrame(self.X @ self.v, columns = np.arange(len(self.lam))+1)
        self.l = len(self.lam)
        if numeigen > 0 and minproportion > 0:
            print("Error, only one is required")
            pass
        elif minproportion > 0:
            proportion = 0
            self.num = 0
            for i in range(len(self.lp)):
                self.num = self.num + 1
                proportion = proportion + self.lp[i]
                if proportion >= minproportion:
                    break
        elif numeigen > 0: 
            self.num = numeigen
        else:
            print("Selecting all components")
            self.num = len(self.lam)
        print("Number of components selected: {}".format(self.num))
        self.est = OLS(dfy, self.dfA.iloc[:,0:self.num], nocons = True, vce = self.vce, cluster = self.cluster)
        self.g = self.est.b
        self.SEg = self.est.SEb
        self.b = self.v[:,0:self.num] @ self.g
        self.Varb = self.v[:,0:self.num] @ self.est.Varb1 @ self.v[:,0:self.num].transpose()
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
        self.t = self.b/self.SEb
        self.pval = 2*ss.t.cdf(-abs(self.t), self.est.df)
    def reg(self):
        self.out1 = pd.DataFrame(np.transpose([self.g, self.SEg]), columns = ["Coefficient", "SE"])
        self.out2 = pd.DataFrame(np.transpose([self.b, self.SEb, self.t, self.pval]),columns = ["Coefficient", "SE", "t", "p-value"], index = self.varlist)
        print("PCA Regression:")
        print(self.out1)
        print("Beta coefficient values:")
        print(self.out2)

class Ridge(object):
    def __init__(self, y, X, k = 0, nocons = False, vce = "ROBUST", cluster = None):
        y, X = clearNaN(y, X)
        self.depname = y.name
        self.nocons = nocons
        self.X0 = X
        self.k = k
        try: 
            self.klength = len(self.k)
            print("Using separated Ridge paramters for each eigenvalue")
        except:
            self.klength = 1
            print("Using constant Ridge paramter for each eigenvalue")
        self.n = len(y)
        
        if nocons == False: 
            cons = pd.Series(np.ones(self.n), index = X.index, name = "Cons")
            X = pd.concat([X, cons], axis = 1)
        self.l = X.shape[1]
        self.dep = np.array(y.values, dtype = float)
        self.X = X.values
        self.Xt = t(self.X)
        self.varlist = X.columns
        if self.klength == 1:
            self.VarX = inv(self.Xt @ self.X + self.k * np.identity(self.l))
        elif self.klength > 1:
            self.lam, self.vec = eigh(self.Xt @ self.X)
            idx = self.lam.argsort()[::-1]
            self.vec = self.vec[:,idx]
            self.lam = self.lam[idx]
            self.lam1 = self.lam + self.k
            self.D = self.lam1 * np.identity(self.l)
            self.VarX = self.vec @ inv(self.D) @ self.vec.transpose()
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
        if np.all(np.any(cluster) != None):
            if np.all(np.all(cluster) != None):
                print("Cluster ID Retrieved!")
                try:
                    clsize = cluster.shape[1] 
                except: 
                    clsize = 1
                if clsize == 1:
                    ncl = len(np.unique(cluster))
                    self.o1 = self.u1 @ t(self.u1)*ncl/(ncl-1)*(self.n-1)/self.df
                    self.ohm = np.zeros(self.o1.shape) 
                    for i in range(self.n):
                        for j in range(self.n):
                            if np.all(cluster.iloc[i] == cluster.iloc[j]):
                                self.ohm[i][j] = self.o1[i][j]
                elif clsize == 2:
                    print("Twoway clustering: ", cluster.columns)
                    ncl1 = len(np.unique(cluster.iloc[:,0]))
                    ncl2 = len(np.unique(cluster.iloc[:,1]))
                    ncl12 = len(np.unique(cluster, axis = 0))
                    self.o1 = self.u1 @ t(self.u1)*(self.n-1)/self.df*ncl1/(ncl1-1)
                    self.o2 = self.u1 @ t(self.u1)*(self.n-1)/self.df*ncl2/(ncl2-1)
                    self.o12 = self.u1 @ t(self.u1)*(self.n-1)/self.df*ncl12/(ncl12-1)
                    #reghdfe in STATA did not adjust for cluster df properly
                    #This version is a better adjustment to cluster df
                    self.ohm1 = np.zeros(self.o1.shape)
                    self.ohm2 = np.zeros(self.o2.shape)
                    self.ohm12 = np.zeros(self.o12.shape)                    
                    print("Retrieving First VCE")
                    for i in range(self.n):
                        for j in range(self.n):
                            if cluster.iloc[i,0] == cluster.iloc[j,0]:
                                self.ohm1[i][j] = self.o1[i][j]
                    print("Retrieving Second VCE")
                    for i in range(self.n):
                        for j in range(self.n):
                            if cluster.iloc[i,1] == cluster.iloc[j,1]:
                                self.ohm2[i][j] = self.o2[i][j]
                    print("Retrieving Third VCE")
                    if ncl12 == self.n:
                        d1 = np.diag(self.o12)
                        self.ohm12 = d1 * np.identity(self.n)
                    else:
                        for i in range(self.n):
                            for j in range(self.n):
                                if np.any(cluster.iloc[i] == cluster.iloc[j]):
                                    self.ohm12[i][j] = self.o12[i][j]
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
            self.Varb1 = self.Varb
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
        
def getK(est, separate = False):
    """
    The following separated k is not recommended but still usable 
    """
    XtX = est.Xt @ est.X
    SE = est.SE
    l, v = np.linalg.eigh(XtX)
    idx = l.argsort()[::-1]
    v = v[:,idx]
    l = l[idx]
    gamma = v.transpose() @ est.b
    ks = SE / gamma**2
    if separate == True:
        k = ks
    if separate == False: 
        k = ss.stats.hmean(ks)
    return k 

def Bartlett(X):
    corrX = np.corrcoef(X.values, rowvar = False)
    detX = np.linalg.det(corrX)
    lndetX = np.log(detX)
    n, p = X.values.shape
    bstat = -(n-1/6*(2*p+5))*lndetX
    df = 1/2*p*(p-1)
    Pval = 1 - ss.chi2.cdf(bstat, df)
    print("Stat: {}, df: {}, Pval: {}".format(bstat, df, Pval))
