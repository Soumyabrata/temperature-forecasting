# ===============================================================================

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


# ===============================================================================
# ### Import dataset 

# In[2]:


df0=pd.read_csv('./data/2577743.csv')
df = df0[['DATE', 'TAVG']].set_index(['DATE'])

print (len(df))


# ===============================================================================
# ### Benchmarking 


import random
end_index_of_df = len(df)


lead_time_array = np.arange(1,5,1)
no_of_experiments = 5


previous_observations = 2000

text_file = open("./results/comparison.txt", "w")
text_file.write("time, our, naive, average \n")





for item1 in lead_time_array:
    #lead_time = item1
    lead_observations = item1

    rmse_array = []
    persist_array = []
    average_array = []

    kot = 0
    while(kot<no_of_experiments):
    
        last_possible_index = end_index_of_df - (previous_observations + lead_observations)
        start_index = random.randint(0, last_possible_index)
        print('From start index of ', str(start_index))

        print(['computing for lead time = ', str(lead_observations), ' day with history of ', str(previous_observations), ' mins'])
        train = df[start_index:start_index + previous_observations]
        test = df[start_index + previous_observations:start_index + previous_observations + lead_observations]
        
        
        # holt winter
        y_hat_avg = test.copy()
        print('computation started')
        fit1 = ExponentialSmoothing(np.asarray(train['TAVG']), seasonal_periods=365, trend='add', seasonal='add', ).fit()
        y_hat_avg['Holt_Winter'] = fit1.forecast(len(test))

        # persistence
        last_value = train['TAVG'][-1]
        y_hat_avg['naive'] = last_value * np.ones(len(test))
        
        
        ## average
        mean_training_value = np.mean(train['TAVG'])
        y_hat_avg['aver'] = mean_training_value*np.ones(len(test))

        print('computation completed')

        # computing the error for exponential smoothing
        a = y_hat_avg['TAVG']
        b = y_hat_avg['Holt_Winter']
        rmse_value1 = np.sqrt(np.mean((b - a) ** 2))
        

        ## computing the error for persistence model
        a = y_hat_avg['TAVG']
        b = y_hat_avg['naive']
        rmse_value2 = np.sqrt(np.mean((b - a) ** 2))
        

        # computing the error for average model
        a = y_hat_avg['TAVG']
        b = y_hat_avg['aver']
        rmse_value3 = np.sqrt(np.mean((b - a) ** 2))
        

        rmse_array.append(rmse_value1)
        persist_array.append(rmse_value2)
        average_array.append(rmse_value3)
            
        print (rmse_value1, rmse_value2, rmse_value3)
            
        kot = kot+1
            
            

    rmse_array = np.array(rmse_array)
    persist_array = np.array(persist_array)
    average_array = np.array(average_array)
    
    text_file.write("%s, %s, %s, %s \n" % (item1, np.mean(rmse_array), np.mean(persist_array), np.mean(average_array)))

    
text_file.close()
print ('Loop complete.')

 