import einops
import torch as t
import torch.nn.functional as F
from grid_world import GridWorld
from grid_world import STATES, ACTIONS
from utils import sampleFromDistribution, EpisodeInformation

t.random.manual_seed(42)

class Agent:
    def __init__(self):
        pass

    def getAction(self, state):
        pass

class ParameterizedPolicyAgent(Agent):
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


class ValueIterationAgent(Agent):
    def __init__(self, n_states:int, n_actions:int, gamma:float, theta:float, terminal_states:[int], transition_fn, reward_fn):
        self.gamma = gamma
        self.n_states = n_states
        self.n_actions = n_actions        
        self.V = t.randn(n_states)
        self.V[terminal_states] = 0.0
        self.transition_fn = transition_fn
        self.reward_fn = reward_fn
        self.V = self.valueIteration(self.V, self.transition_fn, self.reward_fn, self.gamma, self.theta)
        self.pi = self.getPolicy(self.V, self.transition_fn, self.reward_fn, self.gamma)

    def valueIteration(self, V, T, R, gamma, theta):
        delta = 0
        while delta > theta:
            V_prev = V
            V = einops.einsum(T, R + gamma * V, "s a s1, s a s1 -> s a").max(axis=1)
            delta = max(delta, (V_prev - V).abs().max())
        return V

    def getPolicy(self, V, T, R, gamma):
        pi = einops.einsum(T, R + gamma * V, "s a s1, s a s1 -> s a").argmax(dim=1)
        return pi

    def getAction(self, state):
        return self.pi[state]

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
    
    transistion_fn = t.empty(n_states, n_actions, n_states)
    reward_fn = t.empty(n_states, n_actions, n_states)

    '''
    20  21  22  23  24
    15  16  17  18  19
    10  11  12  13  14
    5   6   7   8   9
    0   1   2   3   4
    '''
    