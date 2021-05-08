import numpy as np
from pathlib import Path
from typing import Tuple
from glob import glob

def read_HAR(
    dataset_path:Path,
    timesteps=128
) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    
    """Load the Human Activity Recognition Dataset
    This is a multi-feature multiclass time series dataset
    
    Download Link: https://machinelearningmastery.com/how-to-model-human-activity-from-smartphone-data/

    Raises:
        Exception: If the dataset filepath specified does not match a valid "HARDataset/"

    Returns:
        X_train: np.ndarray
        y_train: np.ndarray
        X_test: np.ndarray
        y_test: np.ndarray
    """
    
    # Find the file
    if not dataset_path.exists():
        raise Exception("Cannot load HAR dataset, make sure you are pointing to a directory called HARDataset/.")

    # Get the X_train features
    X_train_fps = sorted(glob(str(dataset_path / "train" / "Inertial Signals" / "*.txt")))
    X_train = np.dstack([np.loadtxt(fp) for fp in X_train_fps])
    
    # Get the X_test features
    X_test_fps = sorted(glob(str(dataset_path / "test" / "Inertial Signals" / "*.txt")))
    X_test = np.dstack([np.loadtxt(fp) for fp in X_test_fps])
    
    # Get the train labels
    y_train_fp = str(dataset_path / "train" / "y_train.txt")
    y_train = np.loadtxt(y_train_fp, dtype=int)
    
    # Get the test labels
    y_test_fp = str(dataset_path / "test" / "y_test.txt")
    y_test = np.loadtxt(y_test_fp, dtype=int)
    
    # Reshape X and y to meet timestamps requirement
    y_train = y_train.repeat(X_train.shape[1] / timesteps)
    y_test = y_test.repeat(X_test.shape[1] / timesteps)
    X_train = np.reshape((-1, timesteps, X_train.shape[2]))
    X_test = np.reshape((-1, timesteps, X_test.shape[2]))
    
    return X_train, y_train, X_test, y_test