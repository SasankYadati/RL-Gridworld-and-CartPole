import torch as t
import torch.nn.functional as F
from grid_world import GridWorld
from grid_world import STATES, ACTIONS
from utils import sampleFromDistribution, EpisodeInformation

t.random.manual_seed(42)

class ParameterizedPolicyAgent:
    def __init__(self, n_states, n_actions, sigma, gamma=0.9):
        self.gamma = gamma
        self.sigma = sigma
        self.n_states = n_states
        self.n_actions = n_actions        
        self.theta = t.randn(n_states, n_actions) # is this correct initialization

    def getPolicy(self):
        pi = F.softmax(self.sigma * self.theta, dim=1)
        return pi
    
    def sampleTheta(self):
        return t.normal(self.theta, self.sigma * t.ones_like(self.theta))

    def updateTheta(self, theta):
        self.theta = theta

    def getAction(self, state):
        pi = self.getPolicy()
        action_probs = pi[state]
        a = sampleFromDistribution(action_probs)
        return a
    
    def hillSearch(self, n_episodes:int, n_trials:int) -> list[EpisodeInformation]: 
        results = []
        tot_g = 0
        for i in range(n_episodes):
            G = run_episode(self, GridWorld())
            results.append(EpisodeInformation(i, 1, G, self.sigma, n_episodes, n_trials))
            tot_g += G

        max_G_so_far = tot_g / n_episodes
        print(f"trial={0}, avg return={max_G_so_far}")

        for trial in range(1, n_trials):
            theta = self.theta
            theta_ = self.sampleTheta()
            self.theta = theta_
            tot_g = 0
            for i in range(n_episodes):
                G = run_episode(self, GridWorld())
                results.append(EpisodeInformation(i, trial+1, G, self.sigma, n_episodes, n_trials))
                tot_g += G
            self.theta = theta

            if tot_g / n_episodes > max_G_so_far:
                self.theta = theta_
                max_G_so_far = tot_g / n_episodes
                print(f"trial={trial}, avg return={max_G_so_far}")
        
        return results


def run_episode(agent, env:GridWorld):
    state, rew = env.reset()
    tot_return = rew
    done = False
    while not done:
        action = agent.getAction(state)
        next_state, rew, done = env.step(action)
        tot_return += rew
        state = next_state
    return tot_return


if __name__ == '__main__':
    n_states, n_actions = len(STATES), len(ACTIONS)

    agent = ParameterizedPolicyAgent(n_states, n_actions)
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
    