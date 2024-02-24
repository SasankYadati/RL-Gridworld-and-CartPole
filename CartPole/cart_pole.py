from math import pi as PI
import math
from dataclasses import dataclass
import torch as t

G = 9.8
STEP = 0.02

@dataclass
class State:
    x: float
    x_dot: float
    theta: float
    theta_dot: float
    t: float

    def getList(self):
        return [self.x, self.x_dot, self.theta, self.theta_dot, self.t]

    def getTensor(self):
        return t.tensor(self.getList(), dtype=t.float32)

INITIAL_STATE = State(0, 0, 0, 0, 0)

F = 10 #Newton
CART_MASS = 1 #kg
POLE_MASS = 0.1 #kg

POLE_LEN = 1

X_RANGE = (-3, 3)
V_RANGE = (-10, 10)
THETA_RANGE = (-5*PI/12, 5*PI/12)
THETA_V_RANGE = (-PI, PI)
MAX_TIME = 20.0

LEFT = 0
RIGHT = 1

ACTIONS = {LEFT:-1, RIGHT:1}

REWARD = 1.0

class CartPole:
    def __init__(self):
        self.g = G
        self.mc = CART_MASS
        self.mp = POLE_MASS
        self.l = POLE_LEN
        self.F = F
        self.state = INITIAL_STATE
        self.delta_t = STEP

    def isDone(self, state:State) -> bool:
        x, _, theta, _, t = state.getList()
        # if (x < X_RANGE[0] or x > X_RANGE[1]):
        #     print("x out of range")

        # if (theta < THETA_RANGE[0] or theta > THETA_RANGE[1]):
        #     print("theta out of range")

        # if (t >= 20.0):
        #     print("out of time")


        return (
            (x < X_RANGE[0] or x > X_RANGE[1]) or
            (theta < THETA_RANGE[0] or theta > THETA_RANGE[1]) or
            (t >= 20.0)
        )

    def clipVelocity(self, x_dot):
        clipped_v = min(max(x_dot, V_RANGE[0]), V_RANGE[1])
        # if x_dot != clipped_v: print(f"vel clipped from {x_dot} to {clipped_v}")
        return clipped_v
    
    def clipAngVelocity(self, theta_dot):
        clipped_theta_v = min(max(theta_dot, THETA_V_RANGE[0]), THETA_V_RANGE[1])
        # if theta_dot != clipped_theta_v: print(f"ang vel clipped from {theta_dot} to {clipped_theta_v}")
        return clipped_theta_v

    def transition_fn(self, state:State, action:int) -> State:
        assert action in ACTIONS.keys()
        x, x_dot, theta, theta_dot, t = state.getList()

        F = ACTIONS[action] * self.F
        costheta = math.cos(theta)
        sintheta = math.sin(theta)

        tmp = (F + self.mp * self.l * theta_dot ** 2 * sintheta) / (self.mp + self.mc)

        theta_dot_dot = (self.g * sintheta + costheta * (-tmp)) / (self.l * (4.0/3.0 - (self.mp * costheta ** 2)/(self.mc + self.mp)))
        x_dot_dot = tmp - (self.mp * self.l * theta_dot_dot * costheta) / (self.mc + self.mp)

        x = x + self.delta_t * x_dot
        x_dot = x_dot + self.delta_t * x_dot_dot
        x_dot = self.clipVelocity(x_dot)
        
        theta = theta + self.delta_t * theta_dot
        theta_dot = theta_dot + self.delta_t * theta_dot_dot
        theta_dot = self.clipAngVelocity(theta_dot)

        t += STEP

        # print(f"step {t}, s {State(x, x_dot, theta, theta_dot, t)}")

        return State(x, x_dot, theta, theta_dot, t)


    def step(self, action:int) -> tuple[State, float, bool]:
        self.state = self.transition_fn(self.state, action)
        is_done = self.isDone(self.state)
        return self.state, REWARD, is_done

    def reset(self) -> State:
        self.state = INITIAL_STATE
        return INITIAL_STATE, REWARD
        
if __name__ == '__main__':
    env = CartPole()
    
    tot_r = 0.0
    done = False
    s, r = env.reset()
    i = 0
    j = 0
    while not done:
        tot_r += r
        i = i ^ 1
        j = j ^ 1 if i == 0 else j
        action = LEFT if j == 0 else RIGHT
        s, r, done = env.step(action)
    
    print(f"tot_return {tot_r}")