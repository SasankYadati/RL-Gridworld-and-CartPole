from math import pi as PI
import math
from dataclasses import dataclass

G = 9.8
STEP = 1.0/50.0

@dataclass
class State:
    x: float
    x_dot: float
    theta: float
    theta_dot: float
    t: float

    def getTuple(self):
        return (self.x, self.x_dot, self.theta, self.theta_dot, self.t)

INITIAL_STATE = State(0, 0, 0, 0, 0)

F = 10 #Newton
cart_mass = 1 #kg
pole_mass = 0.1 #kg

pole_len = 1

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
        self.mc = cart_mass
        self.mp = pole_mass
        self.l = pole_len
        self.F = F
        self.state = INITIAL_STATE
        self.delta_t = STEP

    def isDone(self, state:State) -> bool:
        x, _, theta, _, t = state.getTuple()
        return (
            (x < X_RANGE[0] or x > X_RANGE[1]) or
            (theta < THETA_RANGE[0] or theta > THETA_RANGE[1]) or
            (t >= 20.0)
        )

    def clipVelocity(self, x_dot):
        return min(max(x_dot, V_RANGE[0]), V_RANGE[1])
    
    def clipAngVelocity(self, theta_dot):
        return min(max(theta_dot, THETA_V_RANGE[0]), THETA_V_RANGE[1])

    def transition_fn(self, state:State, action:int) -> State:
        assert action in ACTIONS.keys()
        x, x_dot, theta, theta_dot, t = state.getTuple()

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

        return State(x, x_dot, theta, theta_dot, t)


    def step(self, action:int) -> tuple[State, float, bool]:
        self.state = self.transition_fn(self.state, action)
        is_done = self.isDone(self.state)
        return self.state, 1.0, is_done
        
if __name__ == '__main__':
    pass