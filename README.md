# Linear Regressions 

This folder contains linear regression programmes I have written using Python 3. 

Each estimator is a python class. The calculation is done in __init__, and the results will be printed using estmator.reg(), for time series estimations: estimator.out.reg() 

Variable inputs require the following data types:

  Dependent Variable: Pandas Series
  
  Independent Variable: Pandas DataFrame 

# Description of regression classes 

Inside LinRegModule: 

1) OLS with twoway standard errors. Located in Ridge.py

This programme is used to replace STATA's twoway standard error robust regression, where I have encountered a twoway correlation in a panel data for my undergraduate dissertation. 

For OLS, constants are included in X automatically unless if nocons option is True 

2) Ridge regression based on Hoerl and Kennard (1970) and Hoerl, Kennard, Baldwin (1975). Located in Ridge.py

This includes the feature of adding a Ridge regression bias parameter into the regression. Main method to reduce effects of imperfect multicollinearity. 

3) PCA, by Massy (1965). Located in Ridge.py 

This is used to analyse the degree of imperfect multicollinearity. Also a method to classify components and analyse coefficients qualitatively. 

Inside TSModule: 

4) ARMA(p,q) + Bootstrap By Franke & Kreiss. 

This class only takes a one dimensional Pandas DataFrame sequence. 

It relies on the OLS class estimator in Ridge.py 

This file contains a 1000 element MA(1) test sequence with a fixed seed. 

This file is separated into two parts, the first part is a two step least squares estimation (AR(infinity)+ARMA(p,q)), avoiding nonlinear optimisation. Requires stationarity assumptions. The second part is Bootstrapping ARMA(p,q) coefficients. At first I thought that the two step LS cannot estiamte standard errors correctly. However the Bootstrap distribution indicates that the standard errors are similar to the ones estimated using LS Asymptotics. This suggests that LS Asymptotics still works even if there is dependence. However Bootstrapping ARMA(p,q) using method in Franke and Kreiss (1992) is highly demanding in calculation. Especially when one wants 10000 bootstrap calculations. 

For ARMA(p, q), I have used AR(20) in the first step to estimate AR(inf), for highly dependent series, more lags are required. However, if more lags are used, there will be fewer estimated MA(q) observations, as a result, lower precision. But please make sure that there are enough observations to carry an MA(q) twostep regression. 

5) Two-step Cochrane-Orcutt 

Apparently CORT is only an AR(1) GLS transformation, so I kept the AR(1), if the model requires AR(p) transformation, it can be done with little effort. 

Not completed: 
Vector Autoregression 
