import os
import numpy as np
import Settings as S
from sklearn.linear_model import LinearRegression
from sklearn.metrics import r2_score

# Create the directory if it doesn't exist
if not os.path.exists('Threshold'):
    os.makedirs('Threshold')

def find_linear_region(vgs, ids_sqrt, window_size=S.fittingWindow):
    best_r2 = -np.inf
    best_fit = None

    # Slide over the data with the given window size
    for i in range(len(vgs) - window_size + 1):
        # Extract window
        vgs_window = vgs[i:i + window_size]
        ids_sqrt_window = ids_sqrt[i:i + window_size]
        
        # Convert to NumPy arrays and reshape
        vgs_window_np = vgs_window.to_numpy().reshape(-1, 1)
        ids_sqrt_window_np = ids_sqrt_window.to_numpy()
        
        # Fit linear model
        model = LinearRegression()
        model.fit(vgs_window_np, ids_sqrt_window_np)
        r2 = r2_score(ids_sqrt_window_np, model.predict(vgs_window_np))
 
        # Keep track of the best fit
        if r2 > best_r2:
            best_r2 = r2
            best_fit = (model, vgs_window_np, ids_sqrt_window_np)
    
    return best_fit