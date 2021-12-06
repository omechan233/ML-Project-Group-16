#!/usr/bin/env python
# coding: utf-8

# In[1]:


import numpy as np
import pandas
import numpy
import matplotlib.pyplot as plt
from sklearn.metrics import mean_squared_error, r2_score
from sklearn.preprocessing import PolynomialFeatures
from sklearn.linear_model import LinearRegression


# In[2]:


def get_plot(prediction):
        fig=plt.figure(figsize=(20,20))
        ax = fig.add_subplot(221,projection='3d')
        ax.scatter(cleaned_data.iloc[:,0],cleaned_data.iloc[:,1],y,c='black',label='Actual Data')
        surface = ax.plot_trisurf(cleaned_data.iloc[:,0],cleaned_data.iloc[:,1],prediction, cmap='coolwarm',alpha=0.75)
        ax.set_title("Actual Data vs predicted")
        ax.set_xlabel('X1')
        ax.set_ylabel('X2')
        ax.legend()
        plt.show()
def get_error(predictions):
    rmse_err = np.sqrt(mean_squared_error(y,predictions))
    r2_err = r2_score(y,predictions)
    print('rmse:',rmse_err)
    print('r2',r2_err)
    rmse.append(rmse_err)
    r2.append(r2_err)
def get_polynomial_features(a,deg):
    p = PolynomialFeatures(degree=deg)
    a_poly = p.fit_transform(a)
    return a_poly


# In[3]:


cols = ['Duration', 'NumSubscribers', 'ViewCount']
cleaned_data = pandas.read_csv("C:\\Users\\vyshn\\Desktop\\cleaned_data_youtube.csv")
cleaned_data.boxplot(cols)
rmse,r2 = [],[]


# In[4]:


# creating X and y features
X = np.column_stack((cleaned_data.iloc[:,0],cleaned_data.iloc[:,1]))
y = cleaned_data.iloc[:,2]
cleaned_data.head()


# In[5]:


for deg in range(2,11):
    poly_X = get_polynomial_features(X,deg)
    print("degree: ",deg)
    model = LinearRegression().fit(poly_X,y)
    predictions = model.predict(poly_X)
    get_plot(predictions) # plotting data and predictions
    get_error(predictions)
    #print(rmse)
plt.errorbar(range(2,11), rmse, yerr= r2,fmt='.',capsize=5) #plotting error plots
plt.plot(range(2,11), rmse, linestyle=':')
plt.ylabel('RMSE')
plt.ylim(39000,49000)
plt.title('Plotting error with varying degree')
plt.xlabel('Degree')  
plt.legend(bbox_to_anchor=(1.5,1))
plt.show()
  


# In[ ]:




