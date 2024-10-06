import pandas as pd
import numpy as np
from pmdarima import auto_arima
from sklearn.preprocessing import Normalizer
from sklearn.svm import SVR

from utils import *
from metrics import evaluate

import warnings
warnings.filterwarnings('ignore')
pd.set_option('display.float_format', '{:.0f}'.format)

scaler = Normalizer()

def one_direction(data, size_of_gap, ml): 
    arima_model = auto_arima(data)
    arima_model.fit(data)
    prediction = arima_model.predict(n_periods = int(size_of_gap))
    
    if ml: 
        res = arima_model.resid()
        data_transformed = transform_to_multivariate(res, size_of_gap)
        _df = to_df(data_transformed)
        
        X_train = scaler.fit_transform(np.array(_df.iloc[:, :-1]))
        y_train = np.array(_df.iloc[:, -1])
        
        ml_model = SVR(kernel='linear').fit(X_train, y_train)
        res = res[-size_of_gap:]
        lst_nan = np.full_like(res, np.nan)
        merged_res = np.concatenate([res, lst_nan])
        
        result = [] 
        for i in range(int(len(merged_res)//2)):
            record = merged_res[i:i+size_of_gap+1]
            
            record[size_of_gap] = ml_model.predict(scaler.transform(np.array(record[:size_of_gap]).reshape(1,-1)))
            
            result.append(record[size_of_gap])
        final_result = prediction + np.array(result)
        return final_result
    else: 
        return prediction 

def run(df, target_col, size_of_gap, r, y_truth, ml=False): 
    data = df[target_col].values
    nan_index = np.where(np.isnan(data))[0][0]

    min_miss_index = r*size_of_gap
    max_miss_index = len(data)-r*size_of_gap-1
    
    # define case
    if nan_index < min_miss_index: # case 1: before 
        data_nan = data[nan_index+size_of_gap:][::-1]
        result = one_direction(data = data_nan
                                  , size_of_gap=size_of_gap
                                  , ml=ml)
    elif nan_index > max_miss_index: # case 2: behind
        data_nan = data[:nan_index]
        result = one_direction(data = data_nan
                                  , size_of_gap=size_of_gap
                                  , ml=ml)
    else: # case: between
        data_before_nan = data[:nan_index]
        before_result = one_direction(data = data_before_nan
                                  , size_of_gap=size_of_gap
                                  , ml=ml)
        
        data_nan_after = data[nan_index+size_of_gap:][::-1]
        after_result = one_direction(data = data_nan_after
                                  , size_of_gap=size_of_gap
                                  , ml=ml)
        result = (before_result + after_result)/2
    return evaluate(y_truth, result)