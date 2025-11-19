import matplotlib.pyplot as plt
import numpy as np

#define constants
c = 299792458 # speed of light in m/s
h = 6.62607015e-34 # Planck's constant in J*s

#define values
t=1 # in seconds
lambda_freq = 1550 # in nm
tao = 1 # in seconds
C = 1 # in m^2

def calculate_eta(P,P_env, N, N_env):
    # Calculate the efficiency
    eta = ((h * c) / (lambda_freq * t)) * ((N-N_env)/(tao * C * (P-P_env)))
    return eta
