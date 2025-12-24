import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from scipy.optimize import curve_fit


#Defining Fixed constants when changing the attenuation
h = 6.62607015e-34
c = 299792458
lambda_m = 1550e-9
lambda_m_std = 0.1e-9
P = 4.4045e-6
P_std = 0.001715938357e-6
P_env = 0.014e-9
P_env_std = 0.006992058988e-9
C = 1.0
C_std = 0.017  # 1.7%
t = 1.0
t_std = 0.001   # 1 ms
N_env = 94.6
N_env_std = 10.21110507
tau_fixed = 46.5  # dB

#Defining the changable variables as dataframe arrays
df = pd.DataFrame({
    "attenuator_db": [38, 35, 32, 29, 26, 23],
    "N": [493.1, 901.7, 1665.7, 3151.1, 5950.2, 11022.1],
    "N_std": [25.31775837, 32.05221435, 42.84351371, 60.44179937, 72.6862359, 92.65881262],
})

df["tau_db"] = tau_fixed + df["attenuator_db"]
df["tau"] = 10 ** (-df["tau_db"] / 10)
df["tau_std"] = df["tau"] * 2 / 100  # 2% on tau

# 3) Efficiency + uncertainty split (Type A vs Type B)
def eta_from_row(N, tau):
    return (h * c * (N - N_env)) / (lambda_m * t * tau * C * (P - P_env))

df["x"] = df["N"] / t  # x = N/t (counts/s)
df["eta"] = eta_from_row(df["N"], df["tau"])

# Type A (statistical): only N and N_env
NminusN_std = np.sqrt(df["N_std"]**2 + N_env_std**2)
df["eta_std_A"] = df["eta"] * (NminusN_std / (df["N"] - N_env))

# Type B (systematic): lambda, P, P_env, t, tau, C
PminusP_std = np.sqrt(P_std**2 + P_env_std**2)
rel_var_B = (
    (PminusP_std / (P - P_env))**2
    + (lambda_m_std / lambda_m)**2
    + (t_std / t)**2
    + (df["tau_std"] / df["tau"])**2
    + (C_std / C)**2
)
df["eta_std_B"] = df["eta"] * np.sqrt(rel_var_B)

# Total per-point uncertainty (for plotting)
df["eta_std_total"] = np.sqrt(df["eta_std_A"]**2 + df["eta_std_B"]**2)

#Fit model: eta(x) = eta0 * (1 - Dt * x)
#    Fit uses ONLY Type A uncertainties
def eta_model(x, eta0, Dt):
    return eta0 * (1.0 - Dt * x)

x = df["x"].to_numpy()
y = df["eta"].to_numpy()
sigma_A = df["eta_std_A"].to_numpy()

eta0_guess = float(df.loc[df["x"].idxmin(), "eta"])  # closest to x=0
Dt_guess = 1e-6

popt, pcov = curve_fit(
    eta_model,
    x, y,
    p0=[eta0_guess, Dt_guess],
    sigma=sigma_A,
    absolute_sigma=True,
    maxfev=100000,
)

eta0_fit, Dt_fit = popt
eta0_std_A, Dt_std_A = np.sqrt(np.diag(pcov))  # Type A uncertainties from the fit

#Add Type B to eta0 (two strategies)
# Strategy 1: max Type B (relative) across points
# Strategy 2: Type B (relative) at the point nearest to x=0

df["relB"] = df["eta_std_B"] / df["eta"]

relB_max = float(df["relB"].max())
idx_nearest_x0 = int(df["x"].argmin())
relB_nearest = float(df["relB"].iloc[idx_nearest_x0])

eta0_std_B_max = eta0_fit * relB_max
eta0_std_B_nearest = eta0_fit * relB_nearest

eta0_std_total_max = np.sqrt(eta0_std_A**2 + eta0_std_B_max**2)
eta0_std_total_nearest = np.sqrt(eta0_std_A**2 + eta0_std_B_nearest**2)

# results
print("\n=== Fit results (Type A only in fit) ===")
print(f"eta0 = {eta0_fit:.10g}  +/- {eta0_std_A:.3g}   (Type A from fit)")
print(f"Dt   = {Dt_fit:.10g}  +/- {Dt_std_A:.3g}   (Type A from fit)")

print("\n=== eta0 after adding Type B in quadrature ===")
print(f"Strategy 1 (max Type B over 6 points): eta0 +/- {eta0_std_total_max:.3g}")
print(f"Strategy 2 (Type B of nearest x≈0 point): eta0 +/- {eta0_std_total_nearest:.3g}")

print("\n=== Per-point values ===")
print(df[["attenuator_db", "x", "eta", "eta_std_A", "eta_std_B", "eta_std_total"]].to_string(index=False))

# 7) Plot: data with total errorbars + fitted curve
x_plot = np.logspace(np.log10(x.min()*0.8), np.log10(x.max()*1.2), 300)
y_fit_plot = eta_model(x_plot, eta0_fit, Dt_fit)

plt.figure()
plt.errorbar(
    x, y,
    yerr=df["eta_std_total"].to_numpy(),
    fmt="o", capsize=5,
    label="Data (Type A + Type B per point)"
)
plt.plot(x_plot, y_fit_plot, label="Fit: eta0 (1 - Dt N/t)")

plt.xscale("log")
plt.xlabel("Counts/s  (N/t)")
plt.ylabel("Detection efficiency η")
plt.title("η vs N/t with dead-time fit")
plt.legend()
plt.show()
