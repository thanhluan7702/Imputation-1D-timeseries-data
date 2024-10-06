import numpy as np
from sklearn.metrics import mean_absolute_error, mean_squared_error
from collections import defaultdict

def fractional_bias(lst_y_truth, lst_y_pred):
        return 2 * abs((np.mean(lst_y_pred) - np.mean(lst_y_truth)) / (np.mean(lst_y_pred) + np.mean(lst_y_truth)))

def fractional_std(lst_y_truth, lst_y_pred):
    std_dev_Y = np.std(lst_y_pred)
    std_dev_X = np.std(lst_y_truth)

    if std_dev_X == 0:
        return None
    
    fsd = 2 * abs((std_dev_Y - std_dev_X) / (std_dev_X + std_dev_Y))
    return fsd

def similarity(lst_y_truth, lst_y_pred):
        T = len(lst_y_truth)  # Number of missing values
        similarity_sum = 0

        for i in range(T):
            yi = lst_y_truth[i]
            xi = lst_y_pred[i]
            similarity_sum += 1 / (1 + abs(yi - xi) / (max(lst_y_pred) - min(lst_y_pred)))

        similarity = similarity_sum / T
        return similarity
    
def mae(lst_y_truth, lst_y_pred): 
    return mean_absolute_error(lst_y_truth, lst_y_pred)

def rmse(lst_y_truth, lst_y_pred):
    return np.sqrt(mean_squared_error(lst_y_truth, lst_y_pred))

def evaluate(lst_y_truth, lst_y_pred):
    return {
        'Similarity' : similarity(lst_y_truth, lst_y_pred),
        'MAE' : mae(lst_y_truth, lst_y_pred),
        'RMSE' : rmse(lst_y_truth, lst_y_pred),
        'Fractional Bias' : fractional_bias(lst_y_truth, lst_y_pred),
        'Fractional Standard Deviation' : fractional_std(lst_y_truth, lst_y_pred)
    }
    
def average_performance_metrics(list_of_dicts):
    totals = defaultdict(float)
    counts = defaultdict(int)
    
    for d in list_of_dicts:
        for key, value in d.items():
            totals[key] += value
            counts[key] += 1

    averages = {key: totals[key] / counts[key] for key in totals}
    return averages