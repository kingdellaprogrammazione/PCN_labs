import csv
from pathlib import Path
from typing import Dict, List, Tuple

import matplotlib.pyplot as plt
import numpy as np
from scipy.optimize import curve_fit
from scipy.integrate import cumulative_trapezoid
import bisect
from scipy.signal import find_peaks, savgol_filter
from scipy.special import erfc



DATA_DIR = Path(__file__).resolve().parents[1] / "data"

SPECT_BLOCK_SIZE = 10

INTERFEROMETER_BLOCK_SIZE = 40  

PEAK_WINDOW_BLOCK_SIZE = 20

PEAK_THRESHOLD = 0.0025

SUB_DOPPLER_PEAK_THRESHOLD = 0.01

# Filtering defaults for extrema detection
RELATIVE_AMPLITUDE_THRESHOLD = 0.2  # fraction of full y-range: keep extrema close to global extrema
MAX_RELATIVE_DISTANCE = None  # fraction of x-range; None disables x-distance filtering
# Defaults for peak-finding sensitivity (for CH2 Gaussian fitting)
FIND_PEAKS_PROMINENCE_FACTOR = 0.15  # fraction of y-range to use as minimum prominence
FIND_PEAKS_DISTANCE_FRACTION = 0.005  # fraction of total samples as minimum separation

 
SPEED_OF_LIGHT = 299792458.0  # m/s

VACOUM_PERMITTIVITY = 8.8541878188e-12  # F/m

#CENTRAL_FREQ = 335.6 * 10**12
CENTRAL_FREQ = 335.116048807 * 10**12 # (41) Hz

CENTRAL_FREQ_UNC = 0.000000041 * 10**12  # (41) Hz

PATH_DIFFERENCE = 0.188

RED_PLANCK_CONSTANT = 1.054571817e-34  # J·s

DECAY_RATE = 28.743e6  # (75) Hz

DECAY_RATE_UNC = 0.075e6  # (40) Hz

CELL_LENGTH = 0.01

CELL_LENGTH_UNC = 0.001

PEAKS_DISTANCE_CORRECTION = True



PLOT_OPTIONS = {
    "iterferometer_raw": False,
    "interferometer_fringes": False,
    "peaks_spacings": False,
    "rescaled_interferometer_fringes": False,
    "absorption+interferometer": False,
    "sub_doppler_absorption": False,
    "fine_absorption_peaks": False,
    "gaussian_fits": True,
    "laser_power_current": False,
    "absorption_baseline_fit": False,
    "transmittance": False
}


PRINT_OPTIONS = {
    
    "allowed_extrema_dist": False,
    "working window": False,
    "peak_fitting": False,
    "fitting_attempts": False
}
    
def apply_window(x: np.ndarray, y: np.ndarray, min_time: float , max_time: float) -> Tuple[np.ndarray, np.ndarray]:
    start_idx = bisect.bisect_left(x, min_time)
    end_idx = bisect.bisect_right(x, max_time)
    return x[start_idx:end_idx], y[start_idx:end_idx]
    
def load_scope_csv(csv_path: Path) -> Tuple[str, List[float], Dict[str, List[float]]]:
    with csv_path.open() as handle:
        reader = csv.reader(handle)
        header = None
        units = None

        for row in reader:
            if row and row[0].strip() == "Source":
                header = [col.strip() for col in row]
                units = [col.strip() for col in next(reader, [])]
                break

        if header is None:
            raise ValueError(f"'Source' header not found in {csv_path}")

        data_rows = [row for row in reader if any(field.strip() for field in row)]

    source_label = header[0]
    source_values = [float(row[0]) for row in data_rows]

    channels: Dict[str, List[float]] = {}
    for idx, name in enumerate(header[1:], start=1):
        if not name.strip():
            continue
        channels[name] = [float(row[idx]) for row in data_rows]

    return source_label, source_values, channels
    
def load_channel(csv_path: Path, channel: str) -> Tuple[np.ndarray, np.ndarray]:
    _, source_values, channels = load_scope_csv(csv_path)
    if channel not in channels:
        raise KeyError(f"{channel} not present in {csv_path.name}")
    return np.array(source_values), np.array(channels[channel])

def detect_working_window(x,y) -> Tuple[float, float]:

    if y.size > 0:
        idx_min = int(np.argmin(y))
        idx_max = int(np.argmax(y))
        min_time = float(x[min(idx_min, idx_max)])
        max_time = float(x[max(idx_min, idx_max)])
    else:
        min_time = x[0]
        max_time = x[-1]

    return min_time, max_time

def exp_model(x, a, b, c, shift):
    return a * np.exp(b * (x-shift)) + c

def gaussian(x: np.ndarray, A: float, mu: float, sigma: float, c: float) -> np.ndarray:
    return A * np.exp(-0.5 * ((x - mu) / sigma) ** 2) + c


def asym_gaussian(x: np.ndarray, A: float, mu: float, sigma_l: float, sigma_r: float, c: float) -> np.ndarray:
    x = np.asarray(x)
    sigma = np.where(x <= mu, sigma_l, sigma_r)
    return A * np.exp(-0.5 * ((x - mu) / sigma) ** 2) + c

def second_order_peak(x: np.ndarray, A: float, shift: float, s: float, damp: float, offset: float) ->np.ndarray:
    return A * (1 - ((x - shift)**2) / ((s**2) + (2 * damp * (x - shift) * s) + ((x - shift)**2))) + offset

def emg(x, A, mu, sigma, tau, offset):
    sigma = max(sigma, 1e-12)
    tau = max(tau, 1e-12)

    lam = 1.0 / tau
    
    with np.errstate(invalid='ignore', over='ignore'):
        arg = lam * (sigma**2 / 2 - (x - mu))
    
        part = A * lam/2 * np.exp(arg)

        return offset + part * erfc((sigma**2 * lam - (x - mu)) / (np.sqrt(2)*sigma))


def poly_model(x, *coeffs):
    return np.polyval(coeffs, x)

def lin_model(x, m, q):
    return m * x + q


def rectify_exp(x: np.ndarray, *fit):
    #ref = x[-1]
    dx = np.diff(x)
    dvals = fit[0]*np.exp(fit[1]*(x-fit[3])) + fit[2]
    rect = cumulative_trapezoid(1/dvals, x, initial=0)
    # rect -= (rect[-1] - ref)
    return rect


    

def fit_exponential(x: np.ndarray, y: np.ndarray, unc: np.ndarray = None, unc_calc=False) -> Tuple[float, float, float, float]:
    c0 = y[-1]
    # a ≈ y0 - c0 (initial amplitude)
    a0 = y[0] - c0
    
    shift0 = x[0]
    
    # b small positive or negative slope
    b0 = (np.log(max(y[1], 1e-9)) - np.log(max(y[0], 1e-9))) / (x[1] - x[0])

    p0 = [a0, b0, c0, shift0]

    try:
        (a, b, c, shift), cov = curve_fit(
            exp_model,
            x,
            y,
            p0=p0,
            maxfev=20000,
            sigma=unc if unc is not None else None,
            absolute_sigma=unc_calc,
        )
    except RuntimeError:
        # fallback: return a simple constant model
        a, b, c, shift = 0.0, 0.0, float(np.mean(y)), shift0

    if unc_calc:
        return (a, b, c, shift), np.sqrt(np.diag(cov))
    else:
        return (a, b, c, shift)


'''
def find_peak_window(x: np.ndarray,y: np.ndarray,x_peak: float,slope_threshold: float = 0.02,consecutive: int = 5, bonus=1.1) -> Tuple[int, int]:
    dy = np.gradient(y, x)
    
    dx, dy = block_average(x, y, block_size=PEAK_WINDOW_BLOCK_SIZE)

    peak_idx = np.argmin(np.abs(dx - x_peak))

    # --- search left --------------------------------------------------------
    left_idx = peak_idx
    count = 0
    for i in range(peak_idx, -1, -1):
        if dy[i] < 0 + slope_threshold:
            count += 1
        else:
            count = 0
        if count >= consecutive:
            left_idx = i
            break

    # --- search right -------------------------------------------------------
    right_idx = peak_idx
    count = 0
    for i in range(peak_idx, len(x)):
        if dy[i] > 0 - slope_threshold:
            count += 1
        else:
            count = 0
        if count >= consecutive:
            right_idx = i
            break

    return np.searchsorted(x, ((dx[left_idx] - dx[peak_idx]) * (bonus)) + dx[peak_idx]), np.searchsorted(x, ((dx[right_idx] - dx[peak_idx]) * (bonus)) + dx[peak_idx])
'''

def find_peak_window(x: np.ndarray, y: np.ndarray, x_peak: float, th: float = 0.01, consecutive: int = 5, bonus: float = 1.0):
    
    start_idx = (np.abs(x - x_peak)).argmin()
    
    y = savgol_filter(y, window_length=30, polyorder=3)
    
    n = len(y)
    left_search = start_idx
    right_search = start_idx
    
    count = 0
    while left_search > 0:
        left_search -= 1
        if y[left_search] < th:
            count += 1
        else:
            count = 0
            
        if count >= consecutive:
            break
            
    count = 0
    while right_search < n - 1:
        right_search += 1
        if y[right_search] < th:
            count += 1
        else:
            count = 0

        if count >= consecutive:
            break
            
    current_width = right_search - left_search
    
    extra_width = int((bonus - 1) * current_width)
    
    padding = extra_width // 2
    
    final_left = max(0, left_search - padding)
    final_right = min(n - 1, right_search + padding)
    
    return final_left, final_right


def fit_peak(x, y, x_peak, y_sigma=None):
    mu0 = x_peak
    
    index_left, index_right = find_peak_window(x, y, x_peak, th=(y[np.searchsorted(x, x_peak)] - np.min(y)) * 0.2 + np.min(y), consecutive=30, bonus=1.5)
    sigma0 = (x[index_right] - x[index_left]) / 6.0
    
    # print(x_peak, index_left, index_right, x[index_left], x[index_right])
    
    # c0 = (y[index_left] + y[index_right]) / 2.0
    
    c0=0.0
    
    tau0 = (x[index_right] - 2*x_peak + x[index_left]) / 2.0
    
    A0 = np.max(y) - c0
    
    fit, cov = curve_fit(gaussian, x, y, p0=[A0, mu0, sigma0, c0], maxfev=10000, sigma=y_sigma, absolute_sigma=True if y_sigma is not None else False, bounds=([0, x[index_left], 0, -0.0001],[np.inf, x[index_right], np.inf, 0.0]))
    
    return fit, cov


def detect_peaks_and_spacings(
    x: np.ndarray,
    y: np.ndarray,
    *,
    prominence: float = None,
    height: float = None,
    distance: float = None,
    relative_amplitude_threshold: float = None,
    max_relative_distance: float = None,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    if relative_amplitude_threshold is None:
        relative_amplitude_threshold = RELATIVE_AMPLITUDE_THRESHOLD
    if max_relative_distance is None:
        max_relative_distance = MAX_RELATIVE_DISTANCE

    try:
        find_kwargs = {}
        if prominence is not None:
            find_kwargs["prominence"] = prominence
        if height is not None:
            find_kwargs["height"] = height
        if distance is not None:
            if not np.isscalar(distance):
                find_kwargs["distance"] = distance
            else:
                dx = np.mean(np.diff(x)) if len(x) > 1 else 1.0
                find_kwargs["distance"] = max(1, int(round(distance / dx)))

        peaks_idx, _ = find_peaks(y, **find_kwargs)
        mins_idx, _ = find_peaks(-y, **find_kwargs)
    except Exception:
        print("Warning: peak finding failed; returning empty extrema.")
        peaks_idx = np.array([], dtype=int)
        mins_idx = np.array([], dtype=int)

    peak_x = np.asarray(x)[peaks_idx]
    min_x = np.asarray(x)[mins_idx]
    
    peak_x.sort()
    min_x.sort()
    
    
    

    if (relative_amplitude_threshold != None and len(y) > 0):
        y_max = float(np.max(y))
        y_min = float(np.min(y))
        y_range = y_max - y_min if y_max != y_min else 0.0
        

        if y_range > 0.0:
            if peak_x.size > 0:
                y_peaks = np.interp(peak_x, x, y)
                keep_peak = y_peaks >= (y_max - relative_amplitude_threshold * y_range)
                peak_x = peak_x[keep_peak]
            if min_x.size > 0:
                    
                y_mins = np.array([y[np.where(x == mx)[0][0]] for mx in min_x])
                    
                keep_min = y_mins <= (y_min + relative_amplitude_threshold * y_range)
                min_x = min_x[keep_min]

    if max_relative_distance == None:
        allowed = max(np.diff(peak_x).max(),np.diff(min_x).max())/3.0
    else:
        allowed = float(max_relative_distance) * (float(np.max(x)) - float(np.min(x)))
    
    if PRINT_OPTIONS.get("allowed_extrema_dist", True):
        print("Min allowed extrema distance:", allowed)
    
    if (len(x) > 1):
        x_min = float(np.min(x))
        x_max = float(np.max(x))
        x_range = x_max - x_min if x_max != x_min else 0.0
        if x_range > 0.0:
            removed = True
            while removed:
                removed = False
                keep = [True] * len(peak_x)
                for i in range(len(peak_x)-1):
                    if abs(peak_x[i+1] - peak_x[i]) < allowed:
                        removed = True
                        
                        pos1 = np.where(x == peak_x[i])[0][0]
                        pos2 = np.where(x == peak_x[i+1])[0][0]
                        
                        if y[pos1] > y[pos2]: keep[i+1] = False
                        else: keep[i] = False
                peak_x = peak_x[keep]
                keep = [True] * len(min_x)
                for i in range(len(min_x)-1):
                    if abs(min_x[i+1] - min_x[i]) < allowed:
                        removed = True
                        
                        pos1 = np.where(x == min_x[i])[0][0]
                        pos2 = np.where(x == min_x[i+1])[0][0]
                        
                        if y[pos1] < y[pos2]: keep[i+1] = False
                        else: keep[i] = False
                min_x = min_x[keep]            

    # Combine extrema, sort by x, and compute spacings and midpoints between consecutive extrema
    if peak_x.size == 0 and min_x.size == 0:
        spacings = np.array([], dtype=float)
        midpoints = np.array([], dtype=float)
    else:
        all_extrema = np.sort(np.concatenate([peak_x, min_x]))
        if all_extrema.size > 1:
            spacings = np.diff(all_extrema)
            midpoints = (all_extrema[:-1] + all_extrema[1:]) / 2.0
        else:
            spacings = np.array([], dtype=float)
            midpoints = np.array([], dtype=float)

    return peak_x, min_x, spacings, midpoints

def block_average(x: np.ndarray, y: np.ndarray, block_size: int) -> np.ndarray:
    n_blocks = len(y) // block_size
    y_blocked = np.array([np.mean(y[i*block_size:(i+1)*block_size]) for i in range(n_blocks)])
    x_blocked = np.array([np.mean(x[i*block_size:(i+1)*block_size]) for i in range(n_blocks)])
    return x_blocked, y_blocked

def load(file, channel, block_size=1):
    x, y = load_channel(file, channel)
    
    diffs = np.abs(np.diff(np.sort(np.unique(y))))
    
    # print("minimum step:", np.min(diffs[diffs>0]))
    
    x, y = block_average(x, y, block_size=block_size)
    
    # x = apply_downsampling(x, downsample)
    # y = apply_downsampling(y, downsample)
    # y = apply_stride(y, stride=stride)
    return x, y

def conversion(x,pos1,pos2,gap_value):
        gap = pos2 - pos1
        conv = gap_value / gap
        return x * conv  
    
def centering_zero(x, pos1, pos2):
    center = (pos1 + pos2) / 2.0
    return x - center

def main():
    covlon = DATA_DIR / "CSVCOVLON.csv"
    covshort = DATA_DIR / "CSVCOVSHORT.csv"
    csv_final = DATA_DIR / "CSVFINAL.csv"
    final_doppler = DATA_DIR / "FINAL_DOPPLER.csv"
    
#---------------------
# laser power-current analysis
#---------------------
    
    I_mA = [
        8, 16, 24, 32, 40, 48, 56, 56.5, 56.6, 56.7, 56.8, 56.9,
        57, 58, 59, 60, 61, 62, 63,
        64, 72, 80,
        88, 96, 104, 112, 120, 128, 136, 144,
        152, 160, 168, 176, 184, 192, 200]

    P_mW = [
        0.00059, 0.00060, 0.00080, 0.00101, 0.00139, 0.00200, 0.00299, 0.0041, 0.0051, 0.0080, 0.0084, 0.0433,
        0.0527, 0.0980, 0.1220, 0.1414, 0.1601, 0.1778, 0.1947,
        0.2122, 0.3415, 0.4789,
        0.614, 0.750, 0.884, 1.023, 1.160,
        1.300, 1.437, 1.576, 1.716, 1.853,
        1.996, 2.139, 2.282, 2.423, 2.564]
    
    P_unc_mW = [0.00001] * 7 + [0.0001] * 15 + [0.001] * 15
    
    peaks_nominal=(1167*10**6, 9192*10**6)

    fr=(16, 37)

    fit, cov = curve_fit(lin_model, I_mA[fr[0]:fr[1]], P_mW[fr[0]:fr[1]], sigma=P_unc_mW[fr[0]:fr[1]], absolute_sigma=True, p0=[0.0, 0.0])
    
    m = fit[0]
    
    print("m: ", m)
    
    sigma_m = np.sqrt(cov[0, 0])
    
    q = fit[1]
    
    sigma_q = np.sqrt(cov[1, 1])
    
    th = -q / m
    
    sigma_th = th * np.sqrt((sigma_q/q)**2 + (sigma_m/m)**2)
    
    print(f"fitted threshol current: {th} mA, variance: {sigma_th} mA")

    if PLOT_OPTIONS.get("laser_power_current", True):
        plt.figure("laser power-current")
        plt.errorbar(I_mA, P_mW, yerr=P_unc_mW, fmt='o-', label="power measurements")
        # plt.plot(I_mA[fr[0]:fr[1]], P_mW[fr[0]:fr[1]], 's')
        plt.plot(range(49, 205), np.polyval(fit, range(49, 205)), label="linear fit")
        plt.xlabel('Current (mA)')
        plt.ylabel('Power (mW)')
        plt.tight_layout()
        plt.grid(True)
        plt.legend()
        
        
#---------------------
# interferometer data loading
#---------------------
    
    
    
    # 0.001208 V effective resolution
    
    resolution = 0.001208  # V
    
    standard_deviation = resolution / np.sqrt(12)
    
    unc = standard_deviation / np.sqrt(INTERFEROMETER_BLOCK_SIZE)
    
    print("Loading interferometer data...")
    
    int_x, covlon_ch1 = load(covlon, "CH1", 2*INTERFEROMETER_BLOCK_SIZE)
    
    
    _, covshort_ch1 = load(covshort, "CH1", 2*INTERFEROMETER_BLOCK_SIZE)
        
    
    covsum = covlon_ch1 + covshort_ch1
    
    min_time, max_time = detect_working_window(_, covsum)
    
    if PRINT_OPTIONS.get("working window", True):
        print(f"Detected working window: {min_time:.6f} s to {max_time:.6f} s")
    
    
    _, covfinal_ch1 = load(csv_final, "CH1", INTERFEROMETER_BLOCK_SIZE)
    
    if PLOT_OPTIONS.get("iterferometer_raw", True):
        plt.figure("interferometer raw signal")
        plt.plot(int_x, covfinal_ch1, label="interferometer raw signal")
        plt.grid(True)
        plt.xlabel("Time (s)")
        plt.ylabel("Voltage (V)")
    
    fringes = covfinal_ch1 - covsum
    
    unc *= np.sqrt(3)
    
    int_x, fringes = apply_window(int_x, fringes, min_time=min_time, max_time=max_time)
    
    peaks, mins, spacings, midpoints = detect_peaks_and_spacings(int_x, savgol_filter(fringes, window_length=30, polyorder=3))

    peak_pos_err = 0.00008 # s assumption from data
    
    unc_peak_pos = peak_pos_err / np.sqrt(12)
    
    unc_peak_distance = unc_peak_pos * np.sqrt(2)
    
    if PLOT_OPTIONS.get("interferometer_fringes", True):
        plt.figure("interferometer fringes")
        plt.plot(int_x, fringes, label="interferometer fringes")
        plt.plot(peaks, fringes[np.searchsorted(int_x, peaks)], "ro", label="detected peaks")
        plt.plot(mins, fringes[np.searchsorted(int_x, mins)], "go", label="detected mins")
        plt.grid(True)
        plt.xlabel("Time (s)")
        plt.ylabel("Voltage (V)")
        plt.axis([-0.015, 0.04, -0.05, 0.05])
    
#---------------------
# fringe fitting and linearization
#---------------------
    
    print("Fitting exponential to fringe spacings...")
    
    unc_vector = np.full(spacings.shape, unc_peak_distance)
    
    fit, unc_fit = fit_exponential(midpoints, spacings, unc=unc_vector, unc_calc=True)
    rect_int_x = rectify_exp(int_x.copy(), *fit)    
    
    res_peaks, res_mins, res_spacings, res_midpoints = detect_peaks_and_spacings(rect_int_x, fringes)    
    
    if PLOT_OPTIONS.get("rescaled_interferometer_fringes", True):
        plt.figure("rescaled interferometer fringes")
        plt.plot(rect_int_x, fringes)
        plt.grid(True)
        # plt.axis([-0.015, 0.04, -0.05, 0.05])
        plt.xlabel("Time (s)")
        plt.ylabel("Voltage (V)")
    
    original_int_x = int_x.copy()
    
    if PEAKS_DISTANCE_CORRECTION:
        int_x=rect_int_x
        peaks=res_peaks
        mins=res_mins
        spacings=res_spacings
        midpoints=res_midpoints    
        
        
#---------------------
# spettroscopy data loading
#---------------------

    print("Loading spectroscopy data...")
        
    absorption_raw = load(csv_final, "CH2", SPECT_BLOCK_SIZE)[1]
    
    resolution = 0.001544 # V effective resolution
    
    uncertainty_abs = resolution / np.sqrt(12)
    
    spet_x, sub_dop_absorption = load(final_doppler, "CH2", SPECT_BLOCK_SIZE)
        
    resolution = 0.00116 # V effective resolution
    
    uncertainty_sub_dop = resolution / np.sqrt(12)    
    
    
    absorption_raw = apply_window(spet_x, absorption_raw, min_time=min_time, max_time=max_time)[1]
    spet_x, sub_doppler_absorption = apply_window(spet_x, sub_dop_absorption, min_time=min_time, max_time=max_time)
    
    if PLOT_OPTIONS.get("sub_doppler_absorption", True):
        plt.figure("sub doppler absorption")
        plt.plot(spet_x, sub_doppler_absorption, label="sub doppler absorption", color="red")
        plt.grid(True)
        plt.xlabel("Time (s)")
        plt.ylabel("Voltage (V)")
    
    fine_absorption = sub_doppler_absorption - absorption_raw 
        
    unc_fine_abs = np.sqrt(uncertainty_abs**2 + uncertainty_sub_dop**2)
        
        
    if PLOT_OPTIONS.get("fine_absorption_peaks", True):
        plt.figure("fine absorption peaks")
        plt.plot(spet_x, fine_absorption, label="fine absorption peaks", color="brown")
        plt.grid(True)
        plt.xlabel("Time (s)")
        plt.ylabel("Voltage (V)")
#---------------------
# baseline and absorption calculation
#---------------------
 
    print("Fitting baseline to absorption spectrum...")

    absorption_clean = absorption_raw.copy()
    baseline_clean = spet_x.copy()
        
    keep = [True]*len(baseline_clean)
    
    
    fine_absorption_peaks = detect_peaks_and_spacings(spet_x, fine_absorption, relative_amplitude_threshold=0.75, max_relative_distance=0.02)[0]
    
    for peak in fine_absorption_peaks:
        left_index, right_index = find_peak_window(spet_x, fine_absorption, peak, th=SUB_DOPPLER_PEAK_THRESHOLD, consecutive=30, bonus=2.0)
        
        keep[left_index:right_index] = [False]*(right_index - left_index)
        
    baseline_clean = baseline_clean[keep]
    absorption_clean = absorption_clean[keep]
    
    unc_vector = np.full(absorption_clean.shape, uncertainty_abs)
    
    baseline, unc_baseline = curve_fit(lin_model, baseline_clean, absorption_clean, maxfev=10000, sigma=unc_vector, absolute_sigma=True)
    
    sigma_m = np.sqrt(unc_baseline[0,0])
    sigma_c = np.sqrt(unc_baseline[1,1])
    cov_mc  = unc_baseline[0,1]

    sigma_baseline = np.sqrt((spet_x * sigma_m)**2 + sigma_c**2 + 2 * spet_x * cov_mc)
    
    baseline = np.polyval(baseline, spet_x)
    
    
    if PEAKS_DISTANCE_CORRECTION:
        unrect_spet_x = spet_x.copy()
        spet_x=rectify_exp(spet_x, *fit)   
    
    
    
    if PLOT_OPTIONS.get("absorption_baseline_fit", True):
        plt.figure("absorption baseline fit")
        plt.plot(spet_x, absorption_raw, label="raw absorption", color="blue")
        plt.plot(baseline_clean, absorption_clean, label="baseline fit points", color="green")
        plt.plot(spet_x, baseline, label="baseline fit", color="orange")
        plt.grid(True)
        plt.xlabel("Time (s)")
        plt.ylabel("Voltage (V)")

    print("Calculating transmittance and absorption...")

    T = absorption_raw / baseline
    
    # T * np.sqrt( (uncertainty_abs / absorption_raw)**2 + (sigma_baseline / baseline)**2 )
    
    absorption = - np.log(T)
    
    # unc -> 1 / T
    
    abs_unc = np.sqrt( (uncertainty_abs / absorption_raw)**2 + (sigma_baseline / baseline)**2 )
    
    
#---------------------
# fine peaks detection and x axis calibration
#---------------------
    
    print("Calibrating frequency axis...")
    
    calibration_peaks = (0, 2)
    
    reference = peaks_nominal[1]
    
    
    # fine_absorption_peaks_0 = detect_peaks_and_spacings(spet_x, fine_absorption, relative_amplitude_threshold=0.75, max_relative_distance=0.02)[0]
    
    
    fine_absorption_peaks_0 = spet_x[find_peaks(savgol_filter(fine_absorption, window_length=30, polyorder=3), prominence=max(fine_absorption)*0.2)[0]]
    
    
    
    
    spet_x = conversion(spet_x, fine_absorption_peaks_0[calibration_peaks[0]], fine_absorption_peaks_0[calibration_peaks[1]],reference)
    
    int_x = conversion(int_x, fine_absorption_peaks_0[calibration_peaks[0]], fine_absorption_peaks_0[calibration_peaks[1]],reference)
    
    fine_absorption_peaks_1 = spet_x[find_peaks(savgol_filter(fine_absorption, window_length=30, polyorder=3), prominence=max(fine_absorption)*0.2)[0]]
    
    spet_x = centering_zero(spet_x, fine_absorption_peaks_1[0], fine_absorption_peaks_1[3])
    
    int_x = centering_zero(int_x, fine_absorption_peaks_1[0], fine_absorption_peaks_1[3])
    
    fine_absorption_peaks = spet_x[find_peaks(savgol_filter(fine_absorption, window_length=30, polyorder=3), prominence=max(fine_absorption)*0.2)[0]]
   
    
    unc_peak_pos=[None]*len(fine_absorption_peaks)
    
    fits = [None]*len(fine_absorption_peaks)
    
    fit_sum = np.full(len(spet_x), 0.0)

 
    for i in range(len(fine_absorption_peaks)):
        fits[i], unc_peak_pos[i] = fit_peak(spet_x, fine_absorption, fine_absorption_peaks[i], y_sigma=unc_fine_abs) 
        fine_absorption_peaks[i] = fits[i][1]
        fit_sum += gaussian(spet_x, *fits[i])
        
        
    fit_gradient = np.gradient(fit_sum, spet_x)

    peak_dist_unc = np.sqrt(unc_peak_pos[calibration_peaks[0]][1][1] + unc_peak_pos[calibration_peaks[1]][1][1])
    
    abs_unc = np.sqrt(abs_unc**2 + (fit_gradient * peak_dist_unc)**2)
    
    if PLOT_OPTIONS.get("transmittance", True):
        plt.figure("transmittance")
        plt.plot(spet_x / 10**9, T, label="transmittance", color="purple")
        plt.xlabel("Frequency (GHz)")
        plt.ylabel("Transmittance")
        plt.grid(True)

    # fine_absorption_peaks = detect_peaks_and_spacings(spet_x, fine_absorption, relative_amplitude_threshold=0.75, max_relative_distance=0.02)[0]
    
    fine_absorption_peaks.sort() 
    
    
    if PLOT_OPTIONS.get("peaks_spacings", True):
        
        # unrect_absorption_peaks = detect_peaks_and_spacings(unrect_spet_x, fine_absorption, relative_amplitude_threshold=0.75, max_relative_distance=0.02)[0]
        
        unrect_absorption_peaks = unrect_spet_x[find_peaks(savgol_filter(fine_absorption, window_length=30, polyorder=3), prominence=max(fine_absorption)*0.2)[0]]
   
        
        raw_x = conversion(original_int_x, unrect_absorption_peaks[calibration_peaks[0]], unrect_absorption_peaks[calibration_peaks[1]],reference)
        
        unrect_spet_x = conversion(unrect_spet_x, unrect_absorption_peaks[calibration_peaks[0]], unrect_absorption_peaks[calibration_peaks[1]],reference)
        
        unrect_absorption_peaks = detect_peaks_and_spacings(unrect_spet_x, fine_absorption, relative_amplitude_threshold=0.75, max_relative_distance=0.02)[0]
        
        unrect_absorption_peaks = unrect_spet_x[find_peaks(savgol_filter(fine_absorption, window_length=30, polyorder=3), prominence=max(fine_absorption)*0.2)[0]]
   
        
        # raw_x = raw_x - raw_x[0]
        
        raw_x = centering_zero(raw_x, unrect_absorption_peaks[0], unrect_absorption_peaks[3])
        
        peaks, mins, spacings, midpoints = detect_peaks_and_spacings(raw_x, fringes)
        fit = fit_exponential(midpoints, spacings)
        # rescaled_x = rectify_exp(raw_x, *fit)          
        res_peaks, res_mins, res_spacings, res_midpoints = detect_peaks_and_spacings(int_x, fringes)
        lin_fit = np.polyfit(res_midpoints, res_spacings,0)
        plt.figure("peak spacings")
        plt.plot(midpoints / 10**9, spacings / 10**6, label="fringe spacings", marker="o")
        plt.plot(midpoints / 10**9, exp_model(midpoints, *fit) / 10**6, label="exponential fit", linestyle="--")
        plt.xlabel("Frequency (GHz)")
        plt.ylabel("Spacing (MHz)")
        plt.grid(True)
        plt.axis([-10.5, 9, 300, 800])
        plt.figure("peak spacings after rescaling")
        plt.plot(res_midpoints / 10**9, res_spacings / 10**6, label="rescaled fringe spacings",marker="o")
        plt.plot(res_midpoints / 10**9, np.polyval(lin_fit, res_midpoints) / 10**6, label="linear fit", linestyle="--")
        plt.xlabel("Frequency (GHz)")
        plt.ylabel("Spacing (MHz)")
        plt.grid(True)
        plt.axis([-9, 9, 300, 800])
    
    
    if PLOT_OPTIONS.get("absorption+interferometer", True):
        plt.figure("absorption peaks + interferometer fringes")
        plt.plot(spet_x / 10**9, absorption, label="absorption peaks", color="orange")
        plt.plot(int_x / 10**9, fringes, label="interferometer fringes", color="green")
        plt.xlabel("Frequency (GHz)")
        plt.legend()
        plt.grid(True)
    
    absorption_peaks = detect_peaks_and_spacings(spet_x, absorption, relative_amplitude_threshold=0.75, max_relative_distance=0.02)[0]
    
    absorption_peaks = spet_x[find_peaks(savgol_filter(absorption, window_length=30, polyorder=3), prominence=max(absorption)*0.2)[0]]
    
    absorption_peaks.sort()
    
    
    print(f"1 -> 3: {(abs(absorption_peaks[0] - absorption_peaks[2])/10**6):.3f} MHz")
    print(f"1 -> 2: {(abs(absorption_peaks[0] - absorption_peaks[1])/10**6):.3f} MHz") 
    print(f"3 -> 4: {(abs(absorption_peaks[2] - absorption_peaks[3])/10**6):.3f} MHz")
    
    print(f"1 -> 3: {(abs(fine_absorption_peaks[0] - fine_absorption_peaks[2])/10**6):.3f} MHz")
    print(f"1 -> 2: {(abs(fine_absorption_peaks[0] - fine_absorption_peaks[1])/10**6):.3f} MHz") 
    print(f"3 -> 4: {(abs(fine_absorption_peaks[2] - fine_absorption_peaks[3])/10**6):.3f} MHz")
    
    print("Fitting absorption peaks...")
    
    fits = {}
    
    for i in range(len(absorption_peaks)):
        peak = absorption_peaks[i]
        params, cov = fit_peak(spet_x, absorption, peak, y_sigma=abs_unc)
        fits[i] = { "params": params, "cov": cov, "window": find_peak_window(spet_x, absorption, peak, th=PEAK_THRESHOLD, consecutive=30, bonus=1.5) }
        fits[i]["window"] = [spet_x[fits[i]["window"][0]], spet_x[fits[i]["window"][1]]] 
        if PRINT_OPTIONS.get("peak_fitting",True):
            print(f"Fitted peak {i}: {peak:.0f} Hz")
            print(f"fit parameters {params}")
    
    sigma_values = np.asarray([fits[i]["params"][2] for i in fits.keys()])
    
    sigma_uncs = np.asarray([np.sqrt(fits[i]["cov"][2,2]) for i in fits.keys()])
    
    amp_values = np.asarray([fits[i]["params"][0] for i in fits.keys()])
    
    amp_uncs = np.asarray([np.sqrt(fits[i]["cov"][0,0]) for i in fits.keys()])
    
    amp_int = np.sqrt(2*np.pi) * sigma_values * amp_values
    
    amp_int_unc = np.sqrt(2*np.pi) * np.abs(amp_int) * np.sqrt( (sigma_uncs / sigma_values)**2 + (amp_uncs / amp_values)**2 )
    
    print( f"amp values: {amp_int}")
    
    amp_int = np.sum(amp_int)
    
    print(f"amp sum: {amp_int}")
    
    amp_int_unc = np.sqrt( np.sum(amp_int_unc**2) )
    
    
    n = (2 * ( 2 * np.pi * CENTRAL_FREQ)**2 * amp_int) / (np.pi * SPEED_OF_LIGHT**2 * CELL_LENGTH * DECAY_RATE)
    
    n_unc = 2 * n * np.sqrt(((( 2 * np.pi * CENTRAL_FREQ)**2 * amp_int_unc) / (( 2 * np.pi * CENTRAL_FREQ)**2 * amp_int))**2 + (((np.pi * SPEED_OF_LIGHT**2 * CELL_LENGTH * DECAY_RATE)*np.sqrt((CELL_LENGTH_UNC/CELL_LENGTH)**2+(DECAY_RATE_UNC/DECAY_RATE)**2))/((np.pi * SPEED_OF_LIGHT**2 * CELL_LENGTH * DECAY_RATE)))**2)
    
    print(f"calculated density: {n:.6e} m^-3")
    
    print(f"calculated density uncertainty: {n_unc:.6e} m^-3")
    
    if PLOT_OPTIONS.get("gaussian_fits", True):
        plt.figure("absorption peaks with fits")
        plt.plot(spet_x / 10**9, absorption, label="absorption peaks", color="orange")
        for fit in fits.values():
            params = fit["params"]
            left_index = np.searchsorted(spet_x, fit["window"][0])
            right_index = np.searchsorted(spet_x, fit["window"][1])
            plt.plot(spet_x[left_index:right_index] / 10**9, gaussian(spet_x[left_index:right_index],*params))
            plt.plot([fit["window"][0] / 10**9, fit["window"][1] / 10**9], gaussian(np.array(fit["window"]),*params), marker="x", color="red")
        plt.xlabel("Frequency (GHz)")
        plt.grid(True)
    
    plt.show()

if __name__ == "__main__":
    main()