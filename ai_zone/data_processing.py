from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, MinMaxScaler
import numpy as np

def split_train_test(X: np.ndarray, y: np.ndarray, test_size: float, scaling: str = None):
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=test_size)
    
    # Perform scaling if necessary
    if scaling is not None:
        if scaling == "standard":
            scaler = StandardScaler()
        elif scaling == "minmax":
            scaler = MinMaxScaler()
        else:
            raise Exception(f"Data scaler {scaling} is not valid.")
        
        # Fit on X_train and then transform X_test as well
        X_train = scaler.fit_transform(X_train)
        X_test = scaler.transform(X_test)
        
    return X_train, X_test, y_train, y_test