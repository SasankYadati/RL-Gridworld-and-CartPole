from dataclasses import dataclass
import torch as t

t.random.manual_seed(42)

def sampleFromDistribution(probs:t.Tensor):
    assert len(probs.shape) == 1
    probs = probs.cumsum(0)
    return (t.rand(1) > probs).sum(dim=-1).item()

@dataclass
class EpisodeInformation:
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

if __name__ == '__main__':
    probs = t.tensor([1, 0, 0])
    assert sampleFromDistribution(probs) == 0

    probs = t.tensor([0.5, 0.5, 0])
    assert sampleFromDistribution(probs) != 2

    probs = t.tensor([0.0500, 0.1000, 0.0500, 0.0000, 0.0000, 0.0000, 0.8000])
    assert sampleFromDistribution(probs) in [0,1,2,6]