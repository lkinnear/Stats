####Louis Kinnear, University of Edinburgh, 6/11/2018
import numpy as np
import matplotlib.pyplot as plt
from scipy.optimize import curve_fit
import scipy as sy
from scipy.stats.distributions import  t
#Import your data and set up your x and y
data = np.genfromtxt('/file path', delimiter=',', names=['x1', 'y1'])
x = data['x1']
y = data['y1']
#Create a linespace to use to calculate the best fit line
x2 = np.linspace(0,1,100)
#Define your non-linear function, there is a range of ones here I've used in the past, make sure the func arguments contains the right variables!
def func(x, a, c, b):
    #return a+(b-a)*sy.exp(-sy.exp(c)*x)
    #return a-sy.exp(-x/c)
    #return a*(1-sy.exp(-x*c))
    return a*x**c+b


#Set up an array for the initial guessed variables, number of variables should match number of function variables
p0 = sy.array([0.5,0.5,0.5])

#Fit the curve. coeffs will give out the optimal parameters for the best fit line using non-linear least squares regression, matcov gives the stimated covariance of coeffs
coeffs, matcov = curve_fit(func, x, y, p0, maxfev=1000)
#Calulate the y values using the best fit parameters
yaj = func(x2, coeffs[0], coeffs[1], coeffs[2])
#print(coeffs)
#print(matcov)
#Compute one standard deviation errors on the parameters use
perr = np.sqrt(np.diag(matcov))
#print(perr)
# 95% confidence interval = 100*(1-alpha)    
alpha = 0.05 
# number of data points
n = len(x)  
# number of parameters 
p = len(p0)   
# number of degrees of freedom 
dof = max(0, n - p) 
    
#Caluclate Student-t value for the dof and confidence level to account for possiblysmall sample size
tval = t.ppf(1.0-alpha/2., dof) 
#print tval
#Takes iterables and finds the upper and lower C.I for each parameter and prints it out. Sigma is the square root of the variance
for i, p,var in zip(range(n), coeffs, np.diag(matcov)):
    sigma = var**0.5
    print 'p{0}: {1} [{2}  {3}]'.format(i, p,
                                  p - sigma*tval,
                                  p + sigma*tval)
#Create arrays for the upper and lower bounds                                  
lower = []
upper = []
#Basically does same as abovce but instead of printing it appends values to get the C.I. parameters
for p,var in zip(coeffs, np.diag(matcov)):
    sigma = var**0.5    
    lower.append(p - sigma*tval)
    upper.append(p + sigma*tval)
#print lower 
#print upper   
#Sets up and draws the upper and lower CIs
xfit = np.linspace(0,1)
yfit = func(xfit, *lower)
plt.plot(xfit,yfit,'--', color='k', label='CI 5%')
yfit = func(xfit, *upper)
plt.plot(xfit,yfit,'--', color='k', label='CI 95%')

#Now plot the rest of the data
plt.scatter(x,y)
#Best fit line using the best fit parameters
plt.plot(x2, yaj, c='k')
plt.xlabel('')
plt.ylabel('', color='k')
plt.show()  
plt.savefig('figure_name', dpi=800, bbox_inches='tight')
 