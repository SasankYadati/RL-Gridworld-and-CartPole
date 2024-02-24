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
        self.w3 = t.randn(n_actions,n_state_attributes)
        self.w2 = t.randn(n_actions,n_state_attributes)
        self.w1 = t.randn(n_actions,n_state_attributes)
        self.b = t.randn(n_actions)
    
    def getAction(self, state:State):
        s = state.getTensor()
        f_s = (self.w1 @ (s) + self.b)
        pi_s = F.softmax(f_s, dim=-1)
        return sampleFromDistribution(pi_s)
 
    def gradientAscentStep(self, env, n_episodes):
        eps = self.eps

        agent_tmp = ParameterizedPolicyAgent()
        agent_tmp.w1 = self.w1.clone() + eps
        
        return1 = sum(run_trial(self, env, n_episodes)) / n_episodes
        return2 = sum(run_trial(agent_tmp, env, n_episodes)) / n_episodes

        grad_w1 = (return2 - return1) / (agent_tmp.w1 - self.w1)
        
        '''
        #####
        agent_tmp = ParameterizedPolicyAgent()
        agent_tmp.w2 = self.w2.clone() + eps
        
        return1 = sum(run_trial(self, env, n_episodes))
        return2 = sum(run_trial(agent_tmp, env, n_episodes))

        grad_w2 = (return2 - return1) / (agent_tmp.w2 - self.w2)

        #####
        agent_tmp = ParameterizedPolicyAgent()
        agent_tmp.w3 = self.w3.clone() + eps
        
        return1 = sum(run_trial(self, env, n_episodes))
        return2 = sum(run_trial(agent_tmp, env, n_episodes))

        grad_w3 = (return2 - return1) / (agent_tmp.w3 - self.w3)
        '''

        #####
        agent_tmp = ParameterizedPolicyAgent()
        agent_tmp.b = self.b.clone() + eps
        
        return1 = sum(run_trial(self, env, n_episodes))
        return2 = sum(run_trial(agent_tmp, env, n_episodes))

        grad_b = (return2 - return1) / (agent_tmp.b - self.b)

        print(f"Grads {grad_w1} {grad_b}")# {grad_w2} {grad_w3}")

        self.w1 += self.alpha * (grad_w1)
        # self.w2 += self.alpha * (grad_w2)
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