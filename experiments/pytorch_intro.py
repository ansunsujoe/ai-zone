from ai_zone.datasets.human_activity import read_HAR
from ai_zone.config import Config
import torch
from torch.utils.data import Dataset, DataLoader

# Hyperparameters
c = Config()
c.add("timesteps", 8)
c.add("latent_dim", 8)
c.add("dataset_path", "../datasets/HARDataset")

# Load dataset
X_train, y_train, X_test, y_test = read_HAR(
    dataset_path=c.get("dataset_path"),
    timesteps=c.get("timesteps")
)

# Add more configs
c.add("features", X_train.shape[2])

# Train and test data loaders
train_data = Dataset(X_train, y_train)
test_data = Dataset(X_test, y_test)
train_loader = DataLoader(train_data, batch_size=32, shuffle=True)
test_loader = DataLoader(test_data, batch_size=32, shuffle=False)

