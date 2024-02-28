import torch as t
from cart_pole import State, LEFT, RIGHT
import torch.nn.functional as F
import einops
from dataclasses import dataclass

g = t.random.manual_seed(42)

def sampleFromDistribution(probs:t.Tensor):
    assert len(probs.shape) == 1
    probs = probs.cumsum(0)
    return (t.rand(1) > probs).sum(dim=-1).item()

@dataclass
class HillSearchResult:
    Episode: int
    Trial: int
    Return: float
    Sigma: float
    NumEpisodesPerTrial: int
    NumTrials: int
    
    def to_dict(self):
        return {
            'Episode': self.Episode,
            'Trial': self.Trial,
            'Return': self.Return,
            'Sigma': self.Sigma,
            'NumEpisodesPerTrial': self.NumEpisodesPerTrial,
            'NumTrials': self.NumTrials
        }

class Agent:
    def __init__(self):
        pass

    def getAction(self, state):
        pass

class ParameterizedPolicyAgent(Agent):
    def __init__(self, alpha=0.01, n_state_attributes=5, n_actions=2):
        self.sigma = alpha
        # self.w3 = t.randn(n_actions, n_state_attributes, dtype=t.float64)
        # self.w2 = t.randn(n_actions, n_state_attributes, dtype=t.float64)
        self.w1 = t.randn(n_state_attributes, dtype=t.float64)
        self.b = t.randn(1, dtype=t.float64)
    
    def getAction(self, state:State):
        s = state.getTensor()
        f_s = (self.w1 * (s)).sum() + self.b
        pi_s = F.sigmoid(f_s)
        return LEFT if pi_s < 0.5 else RIGHT
    
    def sampleParams(self, sigma):
        # w2_ = t.normal(self.w2, self.alpha*t.ones_like(self.w2))
        w1_ = t.normal(self.w1, sigma*t.ones_like(self.w1))
        b_ = t.normal(self.b, sigma*t.ones_like(self.b))
        return w1_, b_

    def hillSearch(self, env, n_episodes:int, n_trials:int) -> list[HillSearchResult]: 
        results = []
        sigma = self.sigma
        # for i in range(n_episodes):
        G = run_episode(self, env)
        results.append(HillSearchResult(1, 1, G, self.sigma, n_episodes, n_trials))

        max_G_so_far = G
        print(f"trial={0}, avg return={max_G_so_far}")

        for trial in range(1, n_trials):
            w1, b = self.w1, self.b
            w1_, b_ = self.sampleParams(sigma)
            self.w1, self.b = w1_, b_
            # for i in range(n_episodes):
            G = run_episode(self, env)
            self.w1, self.b = w1, b

            if G > max_G_so_far:
                results.append(HillSearchResult(1, trial+1, G, self.sigma, n_episodes, n_trials))
                # sigma *= 0.99 # reduce sd of sampling distribution for each improvement
                self.w1, self.b = w1_, b_
                max_G_so_far = G
                print(f"trial={trial}, avg return={max_G_so_far}")
            # else:
            #     hsr = HillSearchResult(1, trial+1, max_G_so_far, self.sigma, n_episodes, n_trials)
            #     results.append(hsr)
        
        return results


def run_episode(agent, env):
    state, rew = env.reset()
    tot_return = rew
    done = False
    while not done:
        action = agent.getAction(state)
        next_state, rew, done = env.step(action)
        tot_return += rew
        state = next_state
    return tot_return

def run_trial(agent, env, n_episodes):
    returns = []
    for _ in range(n_episodes):
        returns.append(run_episode(agent, env))
    return returns

def run_exp(agent, env, n_trials, n_episodes):
    avg_returns = []
    std_returns = []
    for _ in range(n_trials):
        returns = run_trial(agent, env, n_episodes)
        std, mean = t.std_mean(t.tensor(returns))
        avg_returns.append(mean.item())
        std_returns.append(std.item())
    return avg_returns, std_returns