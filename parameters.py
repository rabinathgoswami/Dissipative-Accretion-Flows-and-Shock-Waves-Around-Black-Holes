from dataclasses import dataclass

@dataclass
class Parameters:
    gamma: float
    alpha_pi: float
    I_n: float
    g: float
    # S: float
    # B: float
    mu: float
    # m_p: float
    # k_B: float
    # T_p: float
    _beta: float
    m_dot: float
    m_e: float
    G: float
    M_dot: float
    c: float
    e: float

p = Parameters(
    gamma=4/3,
    alpha_pi=0.001,
    I_n=1.0,
    g=1.5,
    # S=0.0,
    # B=1.0,
    mu=0.6,
    # m_p=1.0,
    # k_B=1.0,
    # T_p=1.0,
    _beta=0.01,
    m_dot=0.233,
    m_e=9.10938356e-28, # electron mass in grams
    G=6.67430e-8, # gravitational constant in cm^3 g^-1 s^-2
    M_dot=1.989e33, # solar mass in grams
    c=2.99792e10, # speed of light in cm/sec
    e=4.8032e-10  # Elementary charge in esu

)

#x_in = 3.107
#lambda_in = 1.4