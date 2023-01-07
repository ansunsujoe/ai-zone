import numpy as np

def read_wine_quality_three_classes(dataset_path: str) -> tuple[np.ndarray, np.ndarray]:
    """Read Wine Quality dataset with 3 balanced classes
    
    Download the dataset here: 
    https://www.kaggle.com/datasets/uciml/red-wine-quality-cortez-et-al-2009

    Args:
        dataset_path (str): _description_

    Returns:
        Tuple[np.ndarray, np.ndarray]: X (features) and y (labels), in that order
    """
    data = np.genfromtxt(dataset_path, delimiter=",", skip_header=1)
    X_train = data[:, :-1]
    y_train = data[:, -1]
    return X_train, y_train