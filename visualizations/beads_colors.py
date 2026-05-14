import numpy as np
from typing import Tuple, Optional, Dict
from enum import Enum


class ColorScheme(Enum):
    STANDARD = "standard"
    DISCONTINUOUS = "discontinuous"
    CONTINUOUS = "continuous"
    HIGH_CONTRAST = "high_contrast"
    COLORBLIND = "colorblind"
    GRAYSCALE = "grayscale"


def standard_red_green(value: float) -> Tuple[float, float, float, float]:
    value = np.clip(value, -1.0, 1.0)
    band = np.round(value * 10) / 10
    if band >= 0.5:
        t = (band - 0.5) * 2
        r, g, b = 1.0, t, 0.0
    elif band > -0.5:
        if band >= 0:
            t = band * 2
            r, g, b = 1.0 - t, 1.0 - t, 0.0
        else:
            t = abs(band) * 2
            r, g, b = 0.0, t, t
    else:
        t = (abs(band) - 0.5) * 2
        r, g, b = 0.0, 1.0, 1.0 - t
    return (r, g, b, 1.0)


def standard_yellow_blue(value: float) -> Tuple[float, float, float, float]:
    value = np.clip(value, -1.0, 1.0)
    band = np.round(value * 10) / 10
    if band >= 0:
        t = band
        r, g, b = 1.0 - t, 1.0 - t, 0.0
    else:
        t = abs(band)
        r, g, b = 0.0, 0.0, t
    return (r, g, b, 1.0)


def continuous_red_green(value: float) -> Tuple[float, float, float, float]:
    value = np.clip(value, -1.0, 1.0)
    if value >= 0:
        r, g, b = value, 0.0, 0.0
    else:
        r, g, b = 0.0, abs(value), 0.0
    return (r, g, b, 1.0)


def continuous_yellow_blue(value: float) -> Tuple[float, float, float, float]:
    value = np.clip(value, -1.0, 1.0)
    if value >= 0:
        r, g, b = value, value, 0.0
    else:
        r, g, b = 0.0, 0.0, abs(value)
    return (r, g, b, 1.0)


def high_contrast_red_green(value: float) -> Tuple[float, float, float, float]:
    value = np.clip(value, -1.0, 1.0)
    if value >= 0:
        r, g, b = 1.0, 1.0 - value, 1.0 - value
    else:
        r, g, b = 1.0 - abs(value), 1.0, 1.0 - abs(value)
    return (r, g, b, 1.0)


def high_contrast_yellow_blue(value: float) -> Tuple[float, float, float, float]:
    value = np.clip(value, -1.0, 1.0)
    if value >= 0:
        r, g, b = 1.0, 1.0, 1.0 - value
    else:
        r, g, b = 1.0 - abs(value), 1.0 - abs(value), 1.0
    return (r, g, b, 1.0)


def colorblind_red_blue(value: float) -> Tuple[float, float, float, float]:
    value = np.clip(value, -1.0, 1.0)
    if value >= 0:
        r, g, b = 0.8 + 0.2 * (1 - value), 1.0 - value, 1.0 - value
    else:
        r, g, b = 1.0 - abs(value), 1.0 - abs(value), 0.8 + 0.2 * (1 - abs(value))
    return (r, g, b, 1.0)


def colorblind_yellow_green(value: float) -> Tuple[float, float, float, float]:
    value = np.clip(value, -1.0, 1.0)
    if value >= 0:
        r, g, b = 1.0, 1.0, 1.0 - value
    else:
        r, g, b = 1.0 - abs(value), 1.0 + value, 1.0 - abs(value) * 0.4
    return (r, g, b, 1.0)


def grayscale(value: float) -> Tuple[float, float, float, float]:
    value = np.clip(value, -1.0, 1.0)
    gray = (value + 1.0) / 2.0
    return (gray, gray, gray, 1.0)


def prob_0_to_color(prob_0: float) -> Tuple[float, float, float, float]:
    prob_0 = np.clip(prob_0, 0.0, 1.0)
    if prob_0 >= 0.5:
        t = 2 * (prob_0 - 0.5)
        r, g, b = 1.0, 1.0 - t, 0.0
    else:
        t = 2 * prob_0
        r, g, b = t, 1.0, 0.0
    return (r, g, b, 1.0)


def prob_0_to_color_continuous(prob_0: float) -> Tuple[float, float, float, float]:
    prob_0 = np.clip(prob_0, 0.0, 1.0)
    if prob_0 >= 0.5:
        t = 2 * (prob_0 - 0.5)
        r, g, b = 1.0, t, 0.0
    else:
        t = 2 * prob_0
        r, g, b = t, 1.0, 0.0
    return (r, g, b, 1.0)


def value_to_color(value: float, scheme: str = "standard", bead_type: str = "Q") -> Tuple[float, float, float, float]:
    if bead_type in ('Q', 'C'):
        scale_type = 'red_green'
    elif bead_type == 'E':
        scale_type = 'yellow_blue'
    else:
        scale_type = 'red_green'
    if scheme == "standard":
        if scale_type == 'red_green':
            return standard_red_green(value)
        else:
            return standard_yellow_blue(value)
    elif scheme == "continuous":
        if scale_type == 'red_green':
            return continuous_red_green(value)
        else:
            return continuous_yellow_blue(value)
    elif scheme == "high_contrast":
        if scale_type == 'red_green':
            return high_contrast_red_green(value)
        else:
            return high_contrast_yellow_blue(value)
    elif scheme == "colorblind":
        if scale_type == 'red_green':
            return colorblind_red_blue(value)
        else:
            return colorblind_yellow_green(value)
    elif scheme == "grayscale":
        return grayscale(value)
    else:
        if scale_type == 'red_green':
            return standard_red_green(value)
        else:
            return standard_yellow_blue(value)


def value_to_color_array(values: np.ndarray, scheme: str = "standard", bead_type: str = "Q") -> np.ndarray:
    shape = values.shape
    colors = np.zeros(shape + (4,), dtype=np.float32)
    if bead_type in ('Q', 'C'):
        scale_type = 'red_green'
    elif bead_type == 'E':
        scale_type = 'yellow_blue'
    else:
        scale_type = 'red_green'
    clipped = np.clip(values, -1.0, 1.0)
    if scheme == "standard":
        if scale_type == 'red_green':
            bands = np.round(clipped * 10) / 10
            mask_pos = bands >= 0.5
            mask_mid_pos = (bands >= 0) & (bands < 0.5)
            mask_neg = bands < 0
            t = (bands - 0.5) * 2
            colors[mask_pos, 0] = 1.0
            colors[mask_pos, 1] = np.clip(t[mask_pos], 0, 1)
            colors[mask_pos, 2] = 0.0
            colors[mask_pos, 3] = 1.0
            t_mid = bands[mask_mid_pos] * 2
            colors[mask_mid_pos, 0] = np.clip(1.0 - t_mid, 0, 1)
            colors[mask_mid_pos, 1] = np.clip(1.0 - t_mid, 0, 1)
            colors[mask_mid_pos, 2] = 0.0
            colors[mask_mid_pos, 3] = 1.0
            t_neg = (np.abs(bands) - 0.5) * 2
            colors[mask_neg, 0] = 0.0
            colors[mask_neg, 1] = 1.0
            colors[mask_neg, 2] = np.clip(1.0 - t_neg[mask_neg], 0, 1)
            colors[mask_neg, 3] = 1.0
        else:
            bands = np.round(clipped * 10) / 10
            mask_pos = bands >= 0
            mask_neg = bands < 0
            t_pos = np.clip(bands[mask_pos], 0, 1)
            colors[mask_pos, 0] = 1.0 - t_pos
            colors[mask_pos, 1] = 1.0 - t_pos
            colors[mask_pos, 2] = 0.0
            colors[mask_pos, 3] = 1.0
            t_neg = np.clip(np.abs(bands[mask_neg]), 0, 1)
            colors[mask_neg, 0] = 0.0
            colors[mask_neg, 1] = 0.0
            colors[mask_neg, 2] = t_neg
            colors[mask_neg, 3] = 1.0
    else:
        for i in range(values.shape[0]):
            for j in range(values.shape[1]):
                colors[i, j] = value_to_color(values[i, j], scheme, bead_type)
    return colors


def total_correlation_color(T: float, E: float, C: float, scheme: str = "standard") -> Tuple[float, float, float, float]:
    T = np.clip(T, -1.0, 1.0)
    E = np.clip(E, -1.0, 1.0)
    C = np.clip(C, -1.0, 1.0)
    if scheme == "standard":
        Gamma_C = np.array(standard_red_green(C)[:3])
        Gamma_E = np.array(standard_yellow_blue(E)[:3])
    else:
        Gamma_C = np.array(value_to_color(C, scheme, 'C')[:3])
        Gamma_E = np.array(value_to_color(E, scheme, 'E')[:3])
    if abs(C) < 1e-10:
        theta_blend = np.pi / 2
    else:
        theta_blend = np.arctan(abs(E) / abs(C))
    blend_factor = (2 * theta_blend / np.pi)
    blended = Gamma_C + blend_factor * (Gamma_E - Gamma_C)
    return (float(blended[0]), float(blended[1]), float(blended[2]), 1.0)


def generate_color_wheel(n_samples: int = 200, scheme: str = "standard") -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    r = np.linspace(0, 1, n_samples)
    phi = np.linspace(0, 2 * np.pi, n_samples)
    R, PHI = np.meshgrid(r, phi)
    colors = np.zeros((n_samples, n_samples, 4))
    for i in range(n_samples):
        for j in range(n_samples):
            r_val = R[i, j]
            phi_val = PHI[i, j]
            if phi_val <= np.pi / 2:
                E = r_val * np.sin(phi_val)
                C = r_val * np.cos(phi_val)
            elif phi_val <= np.pi:
                E = r_val * np.sin(phi_val)
                C = r_val * np.cos(phi_val)
            elif phi_val <= 3 * np.pi / 2:
                E = r_val * np.sin(phi_val)
                C = r_val * np.cos(phi_val)
            else:
                E = r_val * np.sin(phi_val)
                C = r_val * np.cos(phi_val)
            T = E + C
            colors[i, j] = total_correlation_color(T, E, C, scheme)
    return colors


def create_colorbar(ax, scheme: str = "standard", bead_type: str = "Q", label: str = ""):
    import matplotlib.pyplot as plt
    import matplotlib.colors as mcolors
    gradient = np.linspace(-1, 1, 256).reshape(1, -1)
    gradient = np.repeat(gradient, 10, axis=0)
    if bead_type in ('Q', 'C'):
        if scheme == "standard":
            cmap = mcolors.LinearSegmentedColormap.from_list('red_green', [(0, 1, 0), (0.5, 0.5, 0), (1, 0, 0)])
        elif scheme == "continuous":
            cmap = mcolors.LinearSegmentedColormap.from_list('red_green_cont', [(0, 1, 0), (0, 0, 0), (1, 0, 0)])
        else:
            cmap = plt.cm.RdYlGn
    else:
        if scheme == "standard":
            cmap = mcolors.LinearSegmentedColormap.from_list('yellow_blue', [(1, 1, 0), (0, 0, 0), (0, 0, 1)])
        else:
            cmap = plt.cm.RdYlBu
    ax.imshow(gradient, aspect='auto', cmap=cmap, extent=[-1, 1, 0, 1])
    ax.set_xlim(-1, 1)
    ax.set_ylim(0, 1)
    ax.set_xlabel(label)
    ax.set_yticks([])
    if bead_type in ('Q', 'C'):
        ax.text(-1, -0.2, "|1⟩ (Green)", fontsize=8, ha='left')
        ax.text(0, -0.2, "0.5", fontsize=8, ha='center')
        ax.text(1, -0.2, "|0⟩ (Red)", fontsize=8, ha='right')
    else:
        ax.text(-1, -0.2, "Anti-correlated (Blue)", fontsize=8, ha='left')
        ax.text(0, -0.2, "0", fontsize=8, ha='center')
        ax.text(1, -0.2, "Correlated (Yellow)", fontsize=8, ha='right')
