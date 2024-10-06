import random
import pandas as pd 
import numpy as np 

def to_df(array):
    X = [i[:-1] for i in array]
    y = [i[-1] for i in array]
    transpose = [list(x) for x in zip(*X)]

    dataframe = pd.DataFrame({f'Column{i+1}': lst for i, lst in enumerate(transpose)})
    dataframe['Target'] = y
    return dataframe

def transform_to_multivariate(dataframe, gap_size): ## build features vector
    lst_multi_var = []
    for i in range(len(dataframe) - gap_size):
        row = dataframe[i : i+gap_size+1]
        lst_multi_var.append(row)
    return np.array(lst_multi_var)

def create_continuous_missing_values(dataframe, column_name, num_missing_values):
    modified_df = dataframe.copy()
    random_index = random.randint(0, len(dataframe) - num_missing_values - 1)
    y_truth = modified_df.loc[random_index:random_index + num_missing_values - 1, column_name].values.copy()
    modified_df.loc[random_index:random_index + num_missing_values - 1, column_name] = np.nan
    return modified_df, y_truth