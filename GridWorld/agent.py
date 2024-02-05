import torch as t
import torch.nn.functional as F
from grid_world import GridWorld
from grid_world import STATES, ACTIONS
from utils import sampleFromDistribution

class Agent:
    def __init__(self, n_states, n_actions, gamma=0.9, sigma=0.1):
        self.gamma = gamma
        self.sigma = sigma
        self.n_states = n_states
        self.n_actions = n_actions        
        self.theta = t.randn(n_states, n_actions) # is this correct initialization

    def getPolicy(self):
        pi = F.softmax(self.sigma * self.theta, dim=1)
        return pi

    def getAction(self, state):
        pi = self.getPolicy()
        action_probs = pi[state]
        a = sampleFromDistribution(action_probs)
        return a

    def run_episode(self, env:GridWorld):
        state, rew = env.reset()
        tot_return = rew
        done = False
        while not done:
            action = self.getAction(state)
            next_state, rew, done = env.step(action)
            tot_return += rew
            state = next_state
        return tot_return


if __name__ == '__main__':
    n_states, n_actions = len(STATES), len(ACTIONS)

    agent = Agent(n_states, n_actions)
    pi = agent.getPolicy()
    eps = 0.001
    assert abs(pi.sum().item() - n_states) <= eps 
    
    '''
    20  21  22  23  24
    15  16  17  18  19
    10  11  12  13  14
    5   6   7   8   9
    0   1   2   3   4
    '''
    