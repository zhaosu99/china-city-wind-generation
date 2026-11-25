import numpy as np



def smape(y_true, y_pred):
    """
    Calculate the Symmetric Mean Absolute Percentage Error (SMAPE).
    
    Parameters:
    y_true (array-like): Ground truth values.
    y_pred (array-like): Predicted values.
    
    Returns:
    float: SMAPE value in percentage.
    """
    denominator = (np.abs(y_true) + np.abs(y_pred))
    mask = denominator != 0
    # Handle the case where all denominators are zero to avoid division by zero in mean
    if not np.any(mask):
        return 0.0
    return np.mean(np.abs(y_true[mask] - y_pred[mask]) / denominator[mask]) * 100