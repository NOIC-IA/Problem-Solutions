import numpy as np

def score(submission_path, ground_truth_path):
    submission = np.load(submission_path)
    ground_truth = np.load(ground_truth_path)
    
    correct_matches = np.sum(submission == ground_truth)
    total_queries = len(ground_truth)

    return correct_matches / total_queries if total_queries > 0 else 0