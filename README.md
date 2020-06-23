# Linear Regressions 

This folder contains 4 linear regression programmes I have written using Python 3. 

Each estimator is a python class. The calculation is done in __init__, and the display is printed using estmator.reg(). 

Variable inputs require the following data types:

  Dependent Variable: Pandas Series
  
  Independent Variable: Pandas DataFrame 
  

1) OLS with panel/twoway standard errors. Located in Ridge.py

This programme is used to replace the STATA twoway panel standard error regression, where I have encountered a twoway correlation in a panel data for my undergraduate dissertation. 

2) Ridge regression based on Hoerl and Kennard (1970) and Hoerl, Kennard, Baldwin (1975). Located in Ridge.py

This includes the feature of selecting a Ridge regression bias parameter. 

3) PCA, by Massy (1965). Located in Ridge.py 

This is used to analyse the degree of imperfect multicollinearity. Also a method to classify components and analyse coefficients qualitatively. 

4) ARMA(p,q) + Bootstrap By Franke & Kreiss. Located in Bootstrap ARMA.py 

This file relies on the OLS class estimator in Ridge.py 

This file contains a 1000 element MA(1) test sequence with a fixed seed. 

This file is separated into two parts, the first part is a two step least squares estimation (AR(infinity)+ARMA(p,q)), avoiding nonlinear optimisation or maximum likelihood assumptions. Requires stationarity assumptions. 

The second part is Bootstrapping ARMA(p,q) coefficients. At first I thought that the two step LS cannot estiamte standard errors correctly. However the Bootstrap distribution indicates that the standard errors are similar to the ones estimated using LS Asymptotics. This suggests that LS Asymptotics still works even if there is dependence. 

However Bootstrapping ARMA(p,q) using method in Franke and Kreiss (1992) is highly demanding in calculation. Especially when one wants 10000 bootstrap calculations. 

Not completed: 
Vector Autoregression 
