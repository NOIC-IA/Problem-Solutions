import numpy as np

def score(submission_path, ground_truth_path):
    submission = np.load(submission_path)
    ground_truth = np.load(ground_truth_path)

    if submission.shape != ground_truth.shape:
        raise ValueError("Submission and ground truth arrays must have the same shape.")
    
    if submission.ndim != 1 or ground_truth.ndim != 1:
        raise ValueError("Both submission and ground truth must be 1-dimensional arrays.")
    
    correct_matches = np.sum(submission == ground_truth)
    total_queries = len(ground_truth)

    return correct_matches / total_queries if total_queries > 0 else 0