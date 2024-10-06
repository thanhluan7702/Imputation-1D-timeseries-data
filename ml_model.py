import pandas as pd
import numpy as np
from sklearn.ensemble import RandomForestRegressor

from utils import *
from metrics import evaluate

import warnings
warnings.filterwarnings('ignore')
pd.set_option('display.float_format', '{:.0f}'.format)
    
def one_direction(data, test_data, size_of_gap): 
    _df = to_df(data)

    # train process
    X_train = np.array(_df.iloc[:, :-1])
    y_train = np.array(_df.iloc[:, -1])
    model = RandomForestRegressor(n_estimators=200
                                  , random_state=42).fit(X_train, y_train)

    # evaluate process 
    test_data = np.concatenate(test_data).ravel()
    results = []
    for i in range(int(len(test_data)//2)):
        record = test_data[i:i+size_of_gap+1]
        
        # record is sub array, so if it is changed then main array is changed
        record[size_of_gap] = model.predict(np.array(record[:size_of_gap]).reshape(1, -1))
        results.append(record[size_of_gap])
    return results

def run(df, target_col, size_of_gap, r, y_truth):
    data = df[target_col].values
    nan_index = np.where(np.isnan(data))[0][0]

    min_miss_index = r*size_of_gap
    max_miss_index = len(data)-r*size_of_gap-1

    first_data = data[:nan_index]
    last_data  = data[nan_index+size_of_gap:][::-1] # inverse for get data feature

    # define case
    if nan_index < min_miss_index: # case 1: before 
        data_transformed = transform_to_multivariate(last_data, size_of_gap)
        data_test = df.values.tolist()[nan_index : nan_index + 2 * size_of_gap][::-1]
        result = one_direction(data=data_transformed
                               , test_data = data_test
                               , size_of_gap=size_of_gap)
        
    elif nan_index > max_miss_index: # case 2: behind
        data_transformed = transform_to_multivariate(first_data, size_of_gap)
        data_test = df.values.tolist()[nan_index - size_of_gap : nan_index + size_of_gap]
        result = one_direction(data=data_transformed
                               , test_data = data_test
                               , size_of_gap=size_of_gap)
    
    else: # case: between
        ''' a: before 
            b: after
        '''
        Da = data[nan_index + size_of_gap:][::-1]
        Db = data[:nan_index]

        MDa = transform_to_multivariate(Da, size_of_gap)
        data_test_before = df.values.tolist()[nan_index - size_of_gap : nan_index + size_of_gap]
        a_result = one_direction(data=MDa
                                    , test_data=data_test_before
                                    , size_of_gap=size_of_gap)
        
        MDb = transform_to_multivariate(Db, size_of_gap)  
        data_test_after = df.values.tolist()[nan_index:nan_index + 2 * size_of_gap ][::-1]    
        b_result = one_direction(data=MDb
                                    , test_data=data_test_after
                                    , size_of_gap=size_of_gap)

        final_result = [(x + y)/2 for x,y in zip(a_result, b_result)]
        result = final_result
    
    return evaluate(y_truth, result)
