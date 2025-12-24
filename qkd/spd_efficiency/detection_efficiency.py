import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from scipy.optimize import curve_fit


#Defining Fixed constants when changing the attenuation
h = 6.62607015e-34
c = 299792458
lambda_m = 1550e-9
lambda_m_std = 0.1e-9
P = (4.4045e-6)
P_std = 0.001715938357e-6
P_env = 0.014e-9
P_env_std = 0.006992058988e-9
C = 1
C_std = 0.0017 # 1.7%
t = 1
t_std = 0.001 #1 ms
N_env = 94.6
N_env_std = 10.21110507
tau_fixed = 46.5

#Defining the changable variables as dataframe arrays
df = pd.DataFrame({
    "attenuator_db": [38, 35, 32, 29, 26, 23],
    "N": [493.1, 901.7, 1665.7, 3151.1, 5950.2, 11022.1],
    "N_std": [25.31775837, 32.05221435, 42.84351371, 60.44179937, 72.6862359, 92.65881262],
})
df["tau_db"] = tau_fixed + df["attenuator_db"]
df["tau"] = 10 ** (-df["tau_db"] / 10)
df["tau_std"] = df["tau"] * 2 / 100 #defining the standard deviation of tau
def calculate_eta(row):
    return (h * c * (row["N"] - N_env)) / (
        lambda_m * t * row["tau"] * C * (P - P_env)
    )


def calculate_eta_std(row):
    # Differences uncertainties
    NminusN_std = np.sqrt(row["N_std"]**2 + N_env_std**2)
    PminusP_std = np.sqrt(P_std**2 + P_env_std**2)

    rel_var = (
        (NminusN_std / (row["N"] - N_env))**2
        + (PminusP_std / (P - P_env))**2
        + (lambda_m_std / lambda_m)**2
        + (t_std / t)**2
        + (row["tau_std"] / row["tau"])**2
        + (C_std / C)**2
    )

    eta = calculate_eta(row)
    eta_std = eta * np.sqrt(rel_var)

    return eta_std

eta = calculate_eta(df)
eta_std = calculate_eta_std(df)

print(df["tau"])
print(eta)
print(eta_std)

plt.errorbar(df["N"]/t, eta,
    yerr=eta_std,
    fmt="o-", capsize=5)
plt.xscale("log")   #log scale on x-axis
plt.xlabel("Counts/s (log scale)")
plt.ylabel("Detection Efficiency")
plt.title("Detection Efficiency as a function of Counts/s")
plt.show()
