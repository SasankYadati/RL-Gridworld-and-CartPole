import torch as t
from CartPole import State
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
    def __init__(self, n_state_attributes=4, n_actions=2):
        self.weights1 = t.randn(n_state_attributes,n_actions)
        self.weights2 = t.randn(n_state_attributes,n_actions)
        self.bias = t.randn(n_actions)
    
    def getAction(self, state:State):
        s = state.getList()
        f_s = (self.weights1 * (s ** 2) + self.weights2 * (s) + self.bias).sum()
        pi_s = F.softmax(f_s, dim=-1)
        return sampleFromDistribution(pi_s)
 
        
        
