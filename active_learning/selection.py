import numpy as np

def select_top_k(scores, k):
    idx = np.argsort(scores)[::-1]
    return idx[:k]
