import numpy as np
from pathlib import Path

def read_HAR():
    """
    Human Activity Recognition Dataset
    - Multi-feature time series data
    - Download at https://machinelearningmastery.com/how-to-model-human-activity-from-smartphone-data/
    """
    dataset_path = Path('datasets') / "HARDataset"
    if not dataset_path.exists():
        print("Cannot load HAR dataset, make sure there is a directory called HARDataset in datasets")