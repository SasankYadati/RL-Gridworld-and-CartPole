import torch as t
from cart_pole import State
import torch.nn.functional as F
import einops

g = t.random.manual_seed(42)

def sampleFromDistribution(probs:t.Tensor):
    assert len(probs.shape) == 1
    probs = probs.cumsum(0)
    return (t.rand(1) > probs).sum(dim=-1).item()

class Agent:
    def __init__(self):
        pass

    def getAction(self, state):
        pass

class ParameterizedPolicyAgent(Agent):
    def __init__(self, alpha=0.01, eps=1e-3, n_state_attributes=5, n_actions=2):
        self.alpha = alpha
        self.eps = eps
        # self.w3 = t.randn(n_actions, n_state_attributes, dtype=t.float64)
        # self.w2 = t.randn(n_actions, n_state_attributes, dtype=t.float64)
        self.w1 = t.randn(n_actions, n_state_attributes, dtype=t.float64)
        self.b = t.randn(n_actions, dtype=t.float64)
    
    def getAction(self, state:State):
        s = state.getTensor()
        f_s = (self.w1 @ (s) + self.b)
        pi_s = F.softmax(f_s, dim=-1)
        return sampleFromDistribution(pi_s)
    
    def sampleParams(self):
        w1_ = t.normal(self.w1, self.alpha*t.ones_like(self.w1))
        b_ = t.normal(self.b, self.alpha*t.ones_like(self.b))
        return w1_, b_

    def hillSearch(self, env, n_episodes:int, n_trials:int): 
        results = []
        tot_g = 0
        for i in range(n_episodes):
            G = run_episode(self, env)
            tot_g += G

        max_G_so_far = tot_g / n_episodes
        print(f"trial={0}, avg return={max_G_so_far}")

        for trial in range(1, n_trials):
            w1, b = self.w1, self.b
            w1_, b_ = self.sampleParams()
            self.w1 = w1_
            self.b = b_
            tot_g = 0
            for i in range(n_episodes):
                G = run_episode(self, env)
                # results.append(HillSearchResult(i, trial+1, G, self.sigma, n_episodes, n_trials))
                tot_g += G
            self.w1 = w1
            self.b = b

            if tot_g / n_episodes > max_G_so_far:
                self.w1 = w1_
                self.b = b_
                max_G_so_far = tot_g / n_episodes
                print(f"trial={trial}, avg return={max_G_so_far}")
        
        return results
 
    def gradientAscentStep(self, env, n_episodes):
        eps = self.eps

        agent_tmp = ParameterizedPolicyAgent()
        agent_tmp.w1 = self.w1.clone() + eps
        
        return1 = sum(run_trial(self, env, n_episodes)) / n_episodes
        return2 = sum(run_trial(agent_tmp, env, n_episodes)) / n_episodes

        grad_w1 = (return2 - return1) / (eps)
        
        
        #####
        agent_tmp = ParameterizedPolicyAgent()
        agent_tmp.w2 = self.w2.clone() + eps
        
        return1 = sum(run_trial(self, env, n_episodes)) / n_episodes
        return2 = sum(run_trial(agent_tmp, env, n_episodes)) / n_episodes

        grad_w2 = (return2 - return1) / (eps)

        #####
        agent_tmp = ParameterizedPolicyAgent()
        agent_tmp.w3 = self.w3.clone() + eps
        
        return1 = sum(run_trial(self, env, n_episodes)) / n_episodes
        return2 = sum(run_trial(agent_tmp, env, n_episodes)) / n_episodes

        grad_w3 = (return2 - return1) / (agent_tmp.w3 - self.w3)
        

        #####
        agent_tmp = ParameterizedPolicyAgent()
        agent_tmp.b = self.b.clone() + eps
        
        return1 = sum(run_trial(self, env, n_episodes))
        return2 = sum(run_trial(agent_tmp, env, n_episodes))

        grad_b = (return2 - return1) / eps

        print(f"Grads {grad_w2} {grad_w1}  {grad_b}")# {grad_w2} {grad_w3}")

        self.w1 += self.alpha * (grad_w1)
        self.w2 += self.alpha * (grad_w2)
        # self.w3 += self.alpha * (grad_w3)
        self.b += self.alpha * (grad_b)

        print(f"Params {self.w1} {self.b}")
        
        ret = sum(run_trial(self, env, n_episodes)) / n_episodes
        print(f"Avg return per trial {ret}")
    
    def gradientAscent(self, env, n_iters, n_episodes):
        ret = sum(run_trial(self, env, n_episodes)) / n_episodes
        print(f"Avg return per trial {ret}")
        for i in range(n_iters):
            print(f"Iteration:{i+1}/{n_iters}")
            self.gradientAscentStep(env, n_episodes)


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