import numpy as np
import pandas as pd

def score(ground_truth_path, pred_path):
    ground_truth = np.array(pd.read_csv(ground_truth_path))
    pred = np.array(pd.read_csv(pred_path))

    accuracy = np.mean(ground_truth == pred)
    
    return accuracy