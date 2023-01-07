from torch import nn

def get_activation_from_string(activation: str) -> nn.Module:
    act_lower = activation.lower()
    if act_lower == "relu":
        return nn.ReLU()
    elif act_lower == "leaky_relu":
        return nn.LeakyReLU()
    elif act_lower == "sigmoid":
        return nn.Sigmoid()
    elif act_lower == "tanh":
        return nn.Tanh()
    elif act_lower == "softmax":
        return nn.Softmax()
    else:
        raise Exception(f"Activation function {activation} not supported.")