import torch as t
from utils import sampleFromDistribution
t.random.manual_seed(42)
GRID_SZ = 5

UP = 0
DOWN = 1
LEFT = 2
RIGHT = 3

ACTIONS = {UP:(0,1), DOWN:(0,-1), LEFT:(-1,0), RIGHT:(1,0)}

STATES = {i:(i%GRID_SZ, i//GRID_SZ) for i in range(GRID_SZ*GRID_SZ)}

def getStateID(i,j):
    return i + j * GRID_SZ

def intended_next_state(state:int, action:int):
    next_state = (STATES[state][0] + ACTIONS[action][0], STATES[state][1] + ACTIONS[action][1])
    return getStateID(*next_state)

def getNinetyDegRightAction(action):
    if action == UP:
        return RIGHT
    elif action == RIGHT:
        return DOWN
    elif action == DOWN:
        return LEFT
    else:
        return UP
    
def getNinetyDegLeftAction(action):
    g = getNinetyDegRightAction
    return g(g(g(action)))

def getNinetyDegActions(action):
    return getNinetyDegRightAction(action), getNinetyDegLeftAction(action)

class GridWorld:
    def __init__(self):
        self.start_state = getStateID(0,0)
        self.goal_state = getStateID(4,4)
        self.water_state = getStateID(4,2)
        self.blocked_state_1 = getStateID(2,2)
        self.blocked_state_2 = getStateID(3,2)

    def getRewardForEnteringState(self, state:int):
        if state == self.water_state:
            return -10.0
        elif state == self.goal_state:
            return 10.0
        else:
            return 0.0
        
    def getRewardFn(self) -> t.Tensor:
        R = t.zeros(len(STATES), len(ACTIONS), len(STATES))
        for state in STATES.keys():
            for action in ACTIONS.keys():
                for state_ in STATES.keys():
                    R[state, action, state_] = self.getRewardForEnteringState(state_)
        R[self.goal_state, :, self.goal_state] = 0
        return R
        
    def isBlockedState(self, state:int):
        return state == self.blocked_state_1 or state == self.blocked_state_2

    @staticmethod
    def isOutOfGrid(state:int):
        return state not in STATES.keys()

    def getTransitionProbs(self, state:int, action:int):
        p = t.zeros(len(STATES))
        if state == self.goal_state or self.isBlockedState(state):
            p[state] = 1.0
            return p

        p[state] = 0.1
        intended_next_s = intended_next_state(state, action)
        if GridWorld.isOutOfGrid(intended_next_s):
            p[state] += 0.8
        elif self.isBlockedState(intended_next_s):
            p[state] += 0.8
        else:
            p[intended_next_s] += 0.8
        
        a_right, a_left = getNinetyDegActions(action)

        s_right = intended_next_state(state, a_right)
        s_left = intended_next_state(state, a_left)

        if GridWorld.isOutOfGrid(s_right):
            p[state] += 0.05
        elif self.isBlockedState(s_right):
            p[state] += 0.05
        else:
            p[s_right] += 0.05

        if GridWorld.isOutOfGrid(s_left):
            p[state] += 0.05
        elif self.isBlockedState(s_left):
            p[state] += 0.05
        else:
            p[s_left] += 0.05

        return p

    def reset(self):
        self.current_state = 0
        return self.current_state, 0

    def step(self, action):
        trans_probs = self.getTransitionProbs(self.current_state, action)
        next_state = sampleFromDistribution(trans_probs)
        next_reward = self.getRewardForEnteringState(next_state)
        done = next_state == self.goal_state
        self.current_state = next_state
        return next_state, next_reward, done
    
    def getTransistionFn(self):
        T = t.empty(len(STATES), len(ACTIONS), len(STATES))
        for s in STATES.keys():
            for a in ACTIONS.keys():
                T[s][a] = self.getTransitionProbs(s, a)
        
        return T

if __name__ == '__main__':
    gw = GridWorld()

    # basic test from problem description
    probs = gw.getTransitionProbs(getStateID(1,1), RIGHT)
    assert probs.sum() == 1.0
    assert probs[getStateID(2,1)] == 0.8
    assert probs[getStateID(1,0)] == 0.05
    assert probs[getStateID(1,2)] == 0.05
    assert probs[getStateID(1,1)] == 0.1

    # test 2
    probs = gw.getTransitionProbs(getStateID(1,2), UP)
    assert probs.sum() == 1.0
    assert probs[getStateID(0,2)] == 0.05
    assert probs[getStateID(1,2)] == 0.15
    assert probs[getStateID(1,3)] == 0.8

    
    # test 3
    probs = gw.getTransitionProbs(getStateID(0,0), UP)
    assert probs.sum() == 1.0
    assert probs[getStateID(0,0)] == 0.15
    assert probs[getStateID(0,1)] == 0.8
    assert probs[getStateID(1,0)] == 0.05

    # test 4
    T = gw.getTransistionFn()
    assert T[getStateID(1,1)][RIGHT].sum() == 1.0
    assert T[getStateID(1,1)][RIGHT][getStateID(2,1)] == 0.8
    assert T[getStateID(1,1)][RIGHT][getStateID(1,0)] == 0.05
    assert T[getStateID(1,1)][RIGHT][getStateID(1,2)] == 0.05
    assert T[getStateID(1,1)][RIGHT][getStateID(1,1)] == 0.1

    # test 5
    probs = gw.getTransitionProbs(getStateID(4,4), UP)
    assert probs.sum() == 1.0
    print(probs)
    assert probs[getStateID(4,4)] == 1.0

    '''
    20  21  22  23  24
    15  16  17  18  19
    10  11  12  13  14
    5   6   7   8   9
    0   1   2   3   4
    '''