import pandas as pd
import ml_model
import arima_model
from metrics import *
from utils import * 

df = pd.read_csv("co2.txt", delim_whitespace=True)
target_col = 'CO2'
size_of_gap = 18
r = 3 # set higher if large data set
method = ['ml', 'arima', 'arima_ml']

df = pd.DataFrame(df[target_col], columns=[target_col])
df, y_truth = create_continuous_missing_values(df, target_col, size_of_gap)
    
ml = []
arima = []
arima_ml = []

if 'ml' in method: 
    for i in range(20):
        ml.append(ml_model.run(df, target_col, size_of_gap, r, y_truth))

if 'arima' in method: 
    for i in range(20): 
        arima.append(arima_model.run(df, target_col, size_of_gap, r, y_truth))
        
if 'arima_ml' in method: 
    for i in range(20): 
        arima_ml.append(arima_model.run(df, target_col, size_of_gap, r, y_truth, ml=True))
        
print('ML:', average_performance_metrics(ml))
print('Arima:', average_performance_metrics(arima))
print('ML+Arima', average_performance_metrics(arima_ml))

