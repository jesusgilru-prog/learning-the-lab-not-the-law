"""Dimensionless group calculator for rotating machinery windage.

Computes standard Pi groups for centrifuge/rotor windage analysis:
- Re_Omega: rotational Reynolds number
- Cp: power coefficient
- M_tip: tip Mach number
- Pi_geom: geometric aspect ratio
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Optional

import numpy as np


# Physical constants
R_AIR = 287.05  # J/(kg·K), specific gas constant for air
GAMMA_AIR = 1.4  # ratio of specific heats for air


@dataclass
class PiGroups:
    """Dimensionless groups for a single operating point.

    Attributes
    ----------
    Re_Omega : float
        Rotational Reynolds number: rho * omega * R^2 / mu.
    Cp : float
        Power coefficient: P_w / (0.5 * rho * omega^3 * R^5).
    M_tip : float or None
        Tip Mach number: omega * R / c_sound.
    Pi_geom : float or None
        Geometric ratio (gap/radius or length/radius).
    Ro : float or None
        Rossby-like number: omega * L^2 / nu.
    """
    Re_Omega: float
    Cp: float
    M_tip: Optional[float] = None
    Pi_geom: Optional[float] = None
    Ro: Optional[float] = None


def sound_speed(T_K: float) -> float:
    """Speed of sound in air at temperature T.

    Parameters
    ----------
    T_K : float
        Temperature in Kelvin.

    Returns
    -------
    float
        Speed of sound in m/s.
    """
    return np.sqrt(GAMMA_AIR * R_AIR * T_K)


def dynamic_viscosity_air(T_K: float) -> float:
    """Sutherland's law for dynamic viscosity of air.

    Parameters
    ----------
    T_K : float
        Temperature in Kelvin.

    Returns
    -------
    float
        Dynamic viscosity in Pa·s.
    """
    T_ref = 291.15  # K
    mu_ref = 1.827e-5  # Pa·s
    S = 120.0  # K, Sutherland's constant
    return mu_ref * (T_K / T_ref) ** 1.5 * (T_ref + S) / (T_K + S)


def air_density_ideal(p_Pa: float, T_K: float) -> float:
    """Air density from ideal gas law.

    Parameters
    ----------
    p_Pa : float
        Pressure in Pascals.
    T_K : float
        Temperature in Kelvin.

    Returns
    -------
    float
        Density in kg/m^3.
    """
    return p_Pa / (R_AIR * T_K)


def pi_groups(
    rho: float,
    mu: float,
    omega: float,
    R: float,
    P_w: float,
    T_K: Optional[float] = None,
    gap: Optional[float] = None,
    L_char: Optional[float] = None,
) -> PiGroups:
    """Compute dimensionless groups for a single operating point.

    Parameters
    ----------
    rho : float
        Fluid density in kg/m^3.
    mu : float
        Dynamic viscosity in Pa·s.
    omega : float
        Angular velocity in rad/s.
    R : float
        Rotor radius in m.
    P_w : float
        Windage power in W.
    T_K : float, optional
        Temperature in K (for Mach number).
    gap : float, optional
        Air gap width in m (for geometric ratio).
    L_char : float, optional
        Characteristic length in m (for Rossby number).

    Returns
    -------
    PiGroups
        Computed dimensionless groups.
    """
    nu = mu / rho  # kinematic viscosity

    # Rotational Reynolds number
    Re_Omega = rho * omega * R**2 / mu

    # Power coefficient
    denom = 0.5 * rho * omega**3 * R**5
    Cp = P_w / denom if abs(denom) > 1e-30 else np.nan

    # Tip Mach number
    M_tip = None
    if T_K is not None and T_K > 0:
        c = sound_speed(T_K)
        M_tip = omega * R / c

    # Geometric ratio (gap/R)
    Pi_geom = None
    if gap is not None and R > 0:
        Pi_geom = gap / R

    # Rossby-like number
    Ro = None
    if L_char is not None and nu > 0:
        Ro = omega * L_char**2 / nu

    return PiGroups(
        Re_Omega=Re_Omega,
        Cp=Cp,
        M_tip=M_tip,
        Pi_geom=Pi_geom,
        Ro=Ro,
    )


def pi_groups_batch(
    rho: np.ndarray,
    mu: np.ndarray,
    omega: np.ndarray,
    R: np.ndarray,
    P_w: np.ndarray,
    T_K: Optional[np.ndarray] = None,
    gap: Optional[np.ndarray] = None,
) -> dict[str, np.ndarray]:
    """Vectorized Pi group computation for arrays.

    Parameters
    ----------
    rho, mu, omega, R, P_w : np.ndarray
        Arrays of physical quantities, all same length.
    T_K : np.ndarray, optional
        Temperature array.
    gap : np.ndarray, optional
        Air gap array.

    Returns
    -------
    dict of np.ndarray
        Keys: Re_Omega, Cp, M_tip, Pi_geom.
    """
    Re_Omega = rho * omega * R**2 / mu
    denom = 0.5 * rho * omega**3 * R**5
    Cp = np.where(np.abs(denom) > 1e-30, P_w / denom, np.nan)

    result = {
        'Re_Omega': Re_Omega,
        'Cp': Cp,
    }

    if T_K is not None:
        c = np.sqrt(GAMMA_AIR * R_AIR * T_K)
        result['M_tip'] = omega * R / c

    if gap is not None:
        result['Pi_geom'] = gap / R

    return result


def compute_geometric_pi_groups(
    R: np.ndarray,
    R_chamber: np.ndarray,
    h_rotor: np.ndarray,
    gap_radial: np.ndarray | None = None,
) -> dict[str, np.ndarray]:
    """Compute candidate geometric Pi groups for confinement analysis.

    Parameters
    ----------
    R : np.ndarray
        Rotor tip radius in m.
    R_chamber : np.ndarray
        Chamber radius in m.
    h_rotor : np.ndarray
        Rotor axial height in m.
    gap_radial : np.ndarray, optional
        Radial gap in m. Derived from R_chamber - R if not provided.

    Returns
    -------
    dict of np.ndarray
        Keys: Pi_confinement, Pi_gap, Pi_aspect_axial, Pi_blockage.
        All dimensionless by construction: [m]/[m] or [m²]/[m²].
    """
    R = np.asarray(R, dtype=float)
    R_chamber = np.asarray(R_chamber, dtype=float)
    h_rotor = np.asarray(h_rotor, dtype=float)

    if gap_radial is None:
        gap_radial = R_chamber - R
    else:
        gap_radial = np.asarray(gap_radial, dtype=float)

    result = {}

    # Pi_confinement = R / R_chamber  (0 → tiny rotor, 1 → fills chamber)
    result['Pi_confinement'] = np.where(R_chamber > 0, R / R_chamber, np.nan)

    # Pi_gap = gap_radial / R  (normalized clearance)
    result['Pi_gap'] = np.where(R > 0, gap_radial / R, np.nan)

    # Pi_aspect_axial = h_rotor / R  (<1 flat, >1 elongated)
    result['Pi_aspect_axial'] = np.where(R > 0, h_rotor / R, np.nan)

    # Pi_blockage = A_frontal / A_chamber_cross
    # A_frontal ≈ 2*R * h_rotor (rectangular cross-section of arm/rotor)
    A_frontal = 2 * R * h_rotor
    A_chamber = np.pi * R_chamber**2
    result['Pi_blockage'] = np.where(A_chamber > 0, A_frontal / A_chamber, np.nan)

    return result
