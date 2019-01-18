# Ridge Regression

OLS and Ridge Regression based on Hoerl and Kennard (1970) and Hoerl, Kennard, Baldwin (1975). 


Please install the preliminary library required. I used anaconda 3 - spyder 3 as IDE, which included all of them. 

copy the files to your directory and then type: 
import Ridge

First create an OLS or Ridge Type object, then use object.reg() for output on your console. 

OLS takes 3 arguments: y, X, nocons. 

X and y uses dataframe as input. This is easier to manage if you read from excel. And dataframes include columns, which looks nicer at regression.

After you include a constant, please set "nocons = False", it is just a way to calculate R-squred, nothing changes really. If you want to ignore R-squred, leave it as it is, except that R-squred will be closer to 1. This is because of the way TSS is calculated. With constant, TSS = var(y), without constant, TSS = y'y (which is larger). This is also how nocons in STATA works. 

If you standardise the variables, they don't make a difference. 

Default OLS and Ridge has no constant, please add in a column of constant in your dataframe. 

The reason for "nocons = True" is that it is also prepared for principal component analysis. You can regress the principal components with "nocons = True". Because while regressing PCA, 

To use ridge, please use getK first, this is a function to retrieve the best bias/Minimum MSE estimated K value from the two papers. 
Ridge takes four arguments. The third argument is the k value you want to input. Then the last one set "nocons = False"
Standard ridge regression does not include F-test, because it is meaningless. 

Test function is there to do F-test under Homoskedasticity. Takes 4 arguments: y, X, null, and k.

"null" is the restricted model that you want to test. Please input constant if you want the standard test. Otherwise please specify your restricted model. And X would be your unrestricted model. K is the bias you would like to include. 

lag function is basically there to create lag values for time series analysis returns the dataframe of the lagged value. 

Please contact me by 
cai.ming@live.com 
if you have any questions. 
