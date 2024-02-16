import math
from dataclasses import dataclass

PI = math.pi
G = 9.8
X_RANGE = (-3, 3)
THETA_RANGE = (-5*PI/12, 5*PI/12)

MAX_TIME = 20.0
STEP = 1.0/50.0

@dataclass
class State:
    X: float
    V: float
    Theta: float
    Theta_v: float
    T: float

INITIAL_STATE = State(0, 0, 0, 0, 0)

F = 10 #Newton
Mc = 1 #kg
Mp = 0.1 #kg

V_RANGE = (-10, 10)
THETA_V_RANGE = (-PI, PI)

ACTIONS 