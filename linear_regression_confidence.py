

from scipy import stats
import numpy as np
import matplotlib.pyplot as plt
import statsmodels.api as sm
from statsmodels.sandbox.regression.predstd import wls_prediction_std
import pandas as pd
#Input your data
data = pd.read_csv('F:/Gael/gael_analysis/June_analysis/basin_file_gael_northutah.csv', delimiter=',' ,names=['mbs', 'erosion'])
y = data['mbs'].values
x = data['erosion'].values

#Set up your line
x2 = np.linspace(0,max(x)+50)

#This runs a simple linear regression model and then spits out a stats summary followed by an array of the confidence intervals values
x=sm.add_constant(x)
model = sm.OLS(y,x)
fitted = model.fit()
print(fitted.summary2())
print(fitted.conf_int())
conf_int = fitted.conf_int()

lwr_c = conf_int[0,0]
lwr_x = conf_int[1,0]
upp_c = conf_int[0,1]
upp_x = conf_int[1,1]

lwr = lwr_x*(x2)+lwr_c
upp = upp_x*(x2)+upp_c
#Create the figure
x = data['erosion'].values

p=[]

slope, intercept, r_value, p_value, std_err = stats.linregress(x,y)
r = r_value**2
if p_value <0.001:
    p.append('<0.001')
elif p_value <0.01 and p_value >0.001:
    p.append('<0.01')
elif p_value <0.05 and p_value >0.01:
    p.append('<0.05')    
else: 
    p.append('>0.05')
    
mean = slope*x2+intercept
    
fig, ax = plt.subplots()
ax.set_ylim(0, 1)
ax.set_xlim(0,max(x)+50)

ax.scatter(x, y, color='k')
ax.annotate('$R^2=$'+'%0.2f' %r + '\n$p=$'+ p[0],xy=(0.95,0.05), xycoords='axes fraction', color='k', fontsize=14, bbox=dict(facecolor='white', edgecolor='white', boxstyle='square, pad=0.3'),ha='right',va='bottom')



#Thi runs a simple linear regression model and then spits out a stats summary followed by an array of the confidence intervals values

#PLot the confidence intervals and regreszion line
#ax.plot(x2,mean,'k-')
ax.plot(x2,lwr,'k--')
ax.plot(x2,upp,'k--')
ax.plot(x2,mean,'k')
plt.show()  
plt.savefig('figure_name', dpi=800, bbox_inches='tight')
