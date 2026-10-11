from dataclasses import dataclass

@dataclass
class Parameters:
    gamma: float
    alpha_pi: float
    I_n: float
    g: float
    S: float

p = Parameters(
    gamma=4/3,
    alpha_pi=0.0,
    I_n=1.0,
    g=1.5,
    S=0.0
)

#x_in = 3.107
#lambda_in = 1.4