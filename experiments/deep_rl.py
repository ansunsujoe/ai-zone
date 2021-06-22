
class DQNAgent():
    def __init__(self, path, episodes, epsilon_decay, state_size, action_size, 
                 epsilon=1.0, epsilon_min=0.01, gamma=1, alpha=.01,
                 alpha_decay=.01, batch_size=16, prints=False):
    
        self.state_size = state_size
        self.action_size = action_size
        self.episodes = episodes
        self.epsilon = epsilon
        self.epsilon_decay = epsilon_decay
        self.epsilon_min = epsilon_min
        self.gamma = gamma
        self.alpha = alpha
        self.alpha_decay = alpha_decay
        self.batch_size = batch_size
        self.path = path                     #location where the model is saved to
        self.prints = prints                 #if true, the agent will print his scores