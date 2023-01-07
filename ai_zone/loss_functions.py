from torch import nn

def cross_entropy_loss(model: nn.Module, X, y):
    y_hat = model.forward(X)
    return nn.CrossEntropyLoss()(y_hat, y)
    