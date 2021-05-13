#!/usr/bin/env python
# coding: utf-8

# ### Import libraries 
import matplotlib
import numpy as np
import matplotlib.pyplot as plt
import csv
import datetime
from matplotlib.dates import DateFormatter
from statsmodels.tsa.api import ExponentialSmoothing, SimpleExpSmoothing, Holt
import statsmodels.api as sm
import pandas as pd
import matplotlib.dates as mdate



# ===========================================================================================
# ### Import dataset 


df0=pd.read_csv('./data/2577743.csv')
df = df0[['DATE', 'TAVG']].set_index(['DATE'])


# ===========================================================================================
# ### User-defined functions necessary for plots 

# In[3]:


def transform_df_timeplotseries_TAVG(some_df):
    
    
    import datetime
    import matplotlib.dates as mdates
    
    x = []
    y = []
    for i,item in enumerate(some_df.index):
        item_stp = item.split("/")
        time_value = datetime.date(int('20'+item_stp[2]), int(item_stp[1]), int(item_stp[0]))
        x.append(time_value)
        y.append(some_df.TAVG[i])


    x = np.array(x)
    y = np.array(y)
    
    return (x, y)



def transform_df_timeplotseries_Holt(some_df):
    
    
    import datetime
    import matplotlib.dates as mdates
    
    x = []
    y = []
    for i,item in enumerate(some_df.index):
        item_stp = item.split("/")
        time_value = datetime.date(int('20'+item_stp[2]), int(item_stp[1]), int(item_stp[0]))
        x.append(time_value)
        y.append(some_df.Holt_Winter[i])


    x = np.array(x)
    y = np.array(y)
    
    return (x, y)



# ===========================================================================================
# ## Forecasting values 

start_index = 20
train = df[start_index:start_index+2000]
test= df[start_index+2000:start_index+2000+50]


y_hat_avg = test.copy()
print ('computation started')
fit1 = ExponentialSmoothing(np.asarray(train['TAVG']) ,seasonal_periods=365 ,trend='add', seasonal='add',).fit()
y_hat_avg['Holt_Winter'] = fit1.forecast(len(test))
print ('computation completed')





train_df = train[-100:]

(x_tr, y_tr) = transform_df_timeplotseries_TAVG(train_df)
(x_test, y_test) = transform_df_timeplotseries_TAVG(test)

(x_pred, y_pred) = transform_df_timeplotseries_Holt(y_hat_avg)



fig = plt.figure(1)
ax = fig.add_subplot(1,1,1)  
plt.plot(x_tr, y_tr, 'b:', label='Train')
plt.plot(x_test, y_test, 'r--', label='Test')
plt.plot(x_pred, y_pred, 'k-', label='Predicted')
plt.legend(loc='best', fontsize=18)
plt.grid(True)
plt.xlabel('Date (YY-MM)', fontsize=14)
plt.ylabel('Temperature (in K)', fontsize=14)
fig.tight_layout()
save_name = './results/prediction-index20.PDF' 
fig.savefig(save_name)
plt.show()

