"""
Trajectory Analysis Module

Implementation of trajectory analysis methods from:
Sbalzarini & Koumoutsakos (2005) "Feature point tracking and trajectory 
analysis for video imaging in cell biology" J. Struct. Biol. 151(2):182-195

Includes:
- Mean Square Displacement (MSD)
- Moment Scaling Spectrum (MSS)
- Diffusion coefficient estimation
- Motion classification
"""

import numpy as np
from dataclasses import dataclass
from typing import Tuple, Optional, List
from .linking import Trajectory


@dataclass
class MSDResult:
    """Result of MSD analysis."""
    time_lags: np.ndarray  # Time lags (dt)
    msd: np.ndarray  # MSD values
    slope: float  # Slope in log-log plot (alpha)
    intercept: float  # Intercept in log-log plot
    diffusion_coeff: float  # Diffusion coefficient D
    r_squared: float  # R-squared of fit
    
    @property
    def motion_type(self) -> str:
        """Classify motion type based on alpha."""
        if self.slope < 0.3:
            return "confined"
        elif self.slope < 0.7:
            return "subdiffusive"
        elif self.slope < 1.3:
            return "normal"
        elif self.slope < 1.7:
            return "superdiffusive"
        else:
            return "ballistic"


@dataclass
class MSSResult:
    """Result of Moment Scaling Spectrum analysis."""
    orders: np.ndarray  # Moment orders (m = 0, 1, 2, ...)
    scaling_exponents: np.ndarray  # Scaling exponents (gamma_m)
    slope: float  # MSS slope
    intercept: float  # MSS intercept
    r_squared: float  # R-squared of fit
    diffusion_coeffs: np.ndarray  # Generalized diffusion coefficients D_m
    
    @property
    def motion_type(self) -> str:
        """Classify motion type based on MSS slope."""
        if self.slope < 0.25:
            return "stationary"
        elif self.slope < 0.4:
            return "subdiffusive/confined"
        elif self.slope < 0.6:
            return "normal_diffusion"
        elif self.slope < 0.85:
            return "superdiffusive"
        else:
            return "ballistic/directed"
    
    @property
    def is_self_similar(self) -> bool:
        """Check if motion is strongly self-similar (linear MSS)."""
        return self.r_squared > 0.95


def compute_displacements(trajectory: Trajectory, 
                         pixel_size: float = 1.0,
                         frame_interval: float = 1.0) -> Tuple[np.ndarray, np.ndarray]:
    """
    Extract positions and time points from trajectory.
    
    Parameters
    ----------
    trajectory : Trajectory
        Input trajectory
    pixel_size : float
        Physical size per pixel (e.g., µm/pixel)
    frame_interval : float
        Time between frames (e.g., seconds)
        
    Returns
    -------
    positions : ndarray
        Array of (x, y) positions in physical units, shape (n, 2)
    times : ndarray
        Array of time points
    """
    positions = trajectory.positions * pixel_size
    times = trajectory.frames * frame_interval
    return positions, times


def compute_msd(trajectory: Trajectory,
               pixel_size: float = 1.0,
               frame_interval: float = 1.0,
               max_lag_fraction: float = 0.33) -> MSDResult:
    """
    Compute Mean Square Displacement for a trajectory.
    
    MSD(dt) = <|r(t + dt) - r(t)|^2>
    
    For normal diffusion in 2D: MSD = 4D*t where D is diffusion coefficient.
    
    Parameters
    ----------
    trajectory : Trajectory
        Input trajectory
    pixel_size : float
        Physical size per pixel
    frame_interval : float
        Time between frames
    max_lag_fraction : float
        Maximum lag as fraction of trajectory length
        
    Returns
    -------
    result : MSDResult
        MSD analysis result
    """
    positions, times = compute_displacements(trajectory, pixel_size, frame_interval)
    n_points = len(positions)
    
    if n_points < 3:
        return MSDResult(
            time_lags=np.array([frame_interval]),
            msd=np.array([0.0]),
            slope=0.0,
            intercept=0.0,
            diffusion_coeff=0.0,
            r_squared=0.0
        )
    
    max_lag = max(2, int(n_points * max_lag_fraction))
    
    time_lags = []
    msd_values = []
    
    # Get unique frame differences
    frames = trajectory.frames
    frame_diffs = np.unique(frames[1:] - frames[:-1])
    
    for dn in range(1, max_lag):
        # Find all pairs with this frame difference
        sq_displacements = []
        
        for i in range(n_points - 1):
            for j in range(i + 1, min(i + max_lag + 1, n_points)):
                actual_dn = frames[j] - frames[i]
                if actual_dn == dn:
                    dr = positions[j] - positions[i]
                    sq_displacements.append(np.sum(dr**2))
        
        if sq_displacements:
            time_lags.append(dn * frame_interval)
            msd_values.append(np.mean(sq_displacements))
    
    if len(time_lags) < 2:
        return MSDResult(
            time_lags=np.array([frame_interval]),
            msd=np.array([0.0]),
            slope=0.0,
            intercept=0.0,
            diffusion_coeff=0.0,
            r_squared=0.0
        )
    
    time_lags = np.array(time_lags)
    msd_values = np.array(msd_values)
    
    # Fit in log-log space: log(MSD) = alpha * log(t) + log(4D)
    valid = (time_lags > 0) & (msd_values > 0)
    if np.sum(valid) < 2:
        return MSDResult(
            time_lags=time_lags,
            msd=msd_values,
            slope=0.0,
            intercept=0.0,
            diffusion_coeff=0.0,
            r_squared=0.0
        )
    
    log_t = np.log(time_lags[valid])
    log_msd = np.log(msd_values[valid])
    
    # Linear regression
    n = len(log_t)
    sum_x = np.sum(log_t)
    sum_y = np.sum(log_msd)
    sum_xy = np.sum(log_t * log_msd)
    sum_x2 = np.sum(log_t**2)
    
    denom = n * sum_x2 - sum_x**2
    if abs(denom) < 1e-10:
        slope = 0.0
        intercept = np.mean(log_msd)
    else:
        slope = (n * sum_xy - sum_x * sum_y) / denom
        intercept = (sum_y - slope * sum_x) / n
    
    # R-squared
    y_pred = slope * log_t + intercept
    ss_res = np.sum((log_msd - y_pred)**2)
    ss_tot = np.sum((log_msd - np.mean(log_msd))**2)
    r_squared = 1 - ss_res / ss_tot if ss_tot > 0 else 0.0
    
    # Diffusion coefficient: MSD = 4D*t^alpha
    # At t=1: MSD = 4D, so D = exp(intercept) / 4
    diffusion_coeff = np.exp(intercept) / 4.0
    
    return MSDResult(
        time_lags=time_lags,
        msd=msd_values,
        slope=slope,
        intercept=intercept,
        diffusion_coeff=diffusion_coeff,
        r_squared=r_squared
    )


def compute_moment(trajectory: Trajectory,
                  order: int,
                  lag: int,
                  pixel_size: float = 1.0) -> float:
    """
    Compute displacement moment of given order.
    
    mu_m(dn) = (1/N) * sum(|r(n + dn) - r(n)|^m)
    
    Parameters
    ----------
    trajectory : Trajectory
        Input trajectory
    order : int
        Moment order (m)
    lag : int
        Frame lag (dn)
    pixel_size : float
        Physical size per pixel
        
    Returns
    -------
    moment : float
        m-th order moment at given lag
    """
    positions = trajectory.positions * pixel_size
    frames = trajectory.frames
    n_points = len(positions)
    
    displacements = []
    for i in range(n_points):
        for j in range(i + 1, n_points):
            if frames[j] - frames[i] == lag:
                dr = np.linalg.norm(positions[j] - positions[i])
                displacements.append(dr ** order)
    
    if displacements:
        return np.mean(displacements)
    return 0.0


def compute_mss(trajectory: Trajectory,
               pixel_size: float = 1.0,
               frame_interval: float = 1.0,
               max_order: int = 6,
               max_lag_fraction: float = 0.33) -> MSSResult:
    """
    Compute Moment Scaling Spectrum for a trajectory.
    
    For each moment order m, fit mu_m(dt) ~ dt^gamma_m.
    MSS is the plot of gamma_m vs m.
    
    For strongly self-similar processes, MSS is linear through origin
    with slope indicating motion type:
    - slope = 0: stationary
    - slope = 0.5: normal diffusion
    - slope = 1: ballistic motion
    
    Parameters
    ----------
    trajectory : Trajectory
        Input trajectory
    pixel_size : float
        Physical size per pixel
    frame_interval : float
        Time between frames
    max_order : int
        Maximum moment order to compute
    max_lag_fraction : float
        Maximum lag as fraction of trajectory length
        
    Returns
    -------
    result : MSSResult
        MSS analysis result
    """
    n_points = trajectory.length
    max_lag = max(2, int(n_points * max_lag_fraction))
    
    orders = np.arange(0, max_order + 1)
    scaling_exponents = np.zeros(len(orders))
    diffusion_coeffs = np.zeros(len(orders))
    
    scaling_exponents[0] = 0.0  # gamma_0 is always 0
    diffusion_coeffs[0] = 0.0
    
    for m_idx, m in enumerate(orders[1:], 1):
        # Compute moments for all lags
        lags = []
        moments = []
        
        for dn in range(1, max_lag + 1):
            moment = compute_moment(trajectory, m, dn, pixel_size)
            if moment > 0:
                lags.append(dn * frame_interval)
                moments.append(moment)
        
        if len(lags) < 2:
            scaling_exponents[m_idx] = 0.0
            diffusion_coeffs[m_idx] = 0.0
            continue
        
        # Fit in log-log space
        log_lags = np.log(lags)
        log_moments = np.log(moments)
        
        # Linear regression
        n = len(log_lags)
        sum_x = np.sum(log_lags)
        sum_y = np.sum(log_moments)
        sum_xy = np.sum(log_lags * log_moments)
        sum_x2 = np.sum(log_lags**2)
        
        denom = n * sum_x2 - sum_x**2
        if abs(denom) < 1e-10:
            gamma = 0.0
            y0 = np.mean(log_moments)
        else:
            gamma = (n * sum_xy - sum_x * sum_y) / denom
            y0 = (sum_y - gamma * sum_x) / n
        
        scaling_exponents[m_idx] = gamma
        
        # Generalized diffusion coefficient: D_m = (1/2m) * exp(y0)
        diffusion_coeffs[m_idx] = np.exp(y0) / (2.0 * m)
    
    # Fit MSS (gamma_m vs m)
    # Force through origin: gamma_m = slope * m
    if len(orders) > 1:
        # Weighted least squares (exclude m=0)
        m_vals = orders[1:]
        gamma_vals = scaling_exponents[1:]
        
        # slope = sum(m * gamma) / sum(m^2)
        mss_slope = np.sum(m_vals * gamma_vals) / np.sum(m_vals**2)
        
        # R-squared
        gamma_pred = mss_slope * m_vals
        ss_res = np.sum((gamma_vals - gamma_pred)**2)
        ss_tot = np.sum((gamma_vals - np.mean(gamma_vals))**2)
        r_squared = 1 - ss_res / ss_tot if ss_tot > 0 else 0.0
    else:
        mss_slope = 0.0
        r_squared = 0.0
    
    return MSSResult(
        orders=orders,
        scaling_exponents=scaling_exponents,
        slope=mss_slope,
        intercept=0.0,  # Forced through origin
        r_squared=r_squared,
        diffusion_coeffs=diffusion_coeffs
    )


def compute_velocity(trajectory: Trajectory,
                    pixel_size: float = 1.0,
                    frame_interval: float = 1.0) -> Tuple[np.ndarray, float, float]:
    """
    Compute instantaneous velocities and mean velocity.
    
    Parameters
    ----------
    trajectory : Trajectory
        Input trajectory
    pixel_size : float
        Physical size per pixel
    frame_interval : float
        Time between frames
        
    Returns
    -------
    velocities : ndarray
        Instantaneous velocities, shape (n-1, 2)
    mean_speed : float
        Mean speed
    mean_direction : float
        Mean direction in radians
    """
    positions = trajectory.positions * pixel_size
    frames = trajectory.frames
    
    velocities = []
    for i in range(len(positions) - 1):
        dt = (frames[i + 1] - frames[i]) * frame_interval
        if dt > 0:
            v = (positions[i + 1] - positions[i]) / dt
            velocities.append(v)
    
    velocities = np.array(velocities) if velocities else np.zeros((0, 2))
    
    if len(velocities) > 0:
        speeds = np.linalg.norm(velocities, axis=1)
        mean_speed = np.mean(speeds)
        
        # Mean direction (circular mean)
        angles = np.arctan2(velocities[:, 1], velocities[:, 0])
        mean_direction = np.arctan2(np.mean(np.sin(angles)), np.mean(np.cos(angles)))
    else:
        mean_speed = 0.0
        mean_direction = 0.0
    
    return velocities, mean_speed, mean_direction


def compute_confinement_ratio(trajectory: Trajectory,
                             pixel_size: float = 1.0) -> float:
    """
    Compute confinement ratio (range / path length).
    
    Ratio of 1 indicates straight-line motion, 
    ratio approaching 0 indicates confined motion.
    
    Parameters
    ----------
    trajectory : Trajectory
        Input trajectory
    pixel_size : float
        Physical size per pixel
        
    Returns
    -------
    ratio : float
        Confinement ratio
    """
    positions = trajectory.positions * pixel_size
    
    if len(positions) < 2:
        return 1.0
    
    # Total path length
    displacements = np.diff(positions, axis=0)
    path_length = np.sum(np.linalg.norm(displacements, axis=1))
    
    # End-to-end distance
    end_to_end = np.linalg.norm(positions[-1] - positions[0])
    
    if path_length > 0:
        return end_to_end / path_length
    return 1.0


@dataclass
class VACResult:
    """Result of Velocity Autocorrelation analysis."""
    time_lags: np.ndarray  # Time lags (tau)
    vac: np.ndarray  # VAC values (normalized)
    vac_unnorm: np.ndarray  # VAC values (unnormalized)
    decay_time: float  # Characteristic decay time (where VAC drops to 1/e)
    persistence_time: float  # Time where VAC first crosses zero
    integral: float  # Integral of VAC (related to diffusion via Green-Kubo)
    diffusion_coeff_gk: float  # Diffusion coefficient from Green-Kubo relation
    
    @property
    def is_persistent(self) -> bool:
        """Check if motion shows persistent correlations."""
        # Persistent if VAC > 0.1 at tau = 2
        if len(self.vac) > 2:
            return self.vac[2] > 0.1
        return False
    
    @property
    def is_antipersistent(self) -> bool:
        """Check if motion shows anti-persistent (confined) correlations."""
        # Anti-persistent if VAC becomes significantly negative
        return np.any(self.vac < -0.1)
    
    @property
    def motion_character(self) -> str:
        """Characterize motion based on VAC shape."""
        if self.persistence_time < 0:
            # VAC never crosses zero - persistent/ballistic
            if np.mean(self.vac) > 0.5:
                return "ballistic"
            else:
                return "persistent"
        elif self.is_antipersistent:
            return "confined/antipersistent"
        elif self.decay_time < 1.5:
            return "diffusive"
        else:
            return "weakly_persistent"


def compute_velocity_autocorrelation(trajectory: Trajectory,
                                     pixel_size: float = 1.0,
                                     frame_interval: float = 1.0,
                                     max_lag_fraction: float = 0.25,
                                     normalize: bool = True) -> VACResult:
    """
    Compute Velocity Autocorrelation Function.
    
    The VAC measures how velocity at time t correlates with velocity at time t+τ:
    
        C_v(τ) = <v(t) · v(t+τ)> / <|v|²>  (normalized)
        C_v(τ) = <v(t) · v(t+τ)>           (unnormalized)
    
    Physical interpretation:
    - VAC > 0: Velocity persists in same direction (persistent/ballistic motion)
    - VAC ≈ 0: Random direction changes (diffusive motion)
    - VAC < 0: Velocity tends to reverse (confined/antipersistent motion)
    
    The integral of VAC relates to diffusion coefficient via Green-Kubo relation:
        D = (1/d) * ∫₀^∞ C_v(τ) dτ  (d = dimensionality)
    
    Parameters
    ----------
    trajectory : Trajectory
        Input trajectory
    pixel_size : float
        Physical size per pixel (µm)
    frame_interval : float
        Time between frames (seconds)
    max_lag_fraction : float
        Maximum lag as fraction of trajectory length
    normalize : bool
        Whether to normalize by <|v|²>
        
    Returns
    -------
    result : VACResult
        VAC analysis result including decay time, persistence, and diffusion estimate
    """
    # Compute velocities
    positions = trajectory.positions * pixel_size
    frames = trajectory.frames
    
    # Handle gaps in trajectory
    velocities = []
    time_points = []
    
    for i in range(len(positions) - 1):
        dt = (frames[i + 1] - frames[i]) * frame_interval
        if dt > 0 and dt <= 2 * frame_interval:  # Only consecutive or nearly consecutive
            v = (positions[i + 1] - positions[i]) / dt
            velocities.append(v)
            time_points.append(i)
    
    velocities = np.array(velocities)
    n_vel = len(velocities)
    
    if n_vel < 3:
        # Not enough data
        return VACResult(
            time_lags=np.array([0.0]),
            vac=np.array([1.0]),
            vac_unnorm=np.array([0.0]),
            decay_time=0.0,
            persistence_time=0.0,
            integral=0.0,
            diffusion_coeff_gk=0.0
        )
    
    # Compute mean squared velocity for normalization
    v_dot_v = np.sum(velocities * velocities, axis=1)  # |v|² for each time
    mean_v_sq = np.mean(v_dot_v)
    
    if mean_v_sq < 1e-10:
        # Stationary trajectory
        return VACResult(
            time_lags=np.array([0.0]),
            vac=np.array([1.0]),
            vac_unnorm=np.array([0.0]),
            decay_time=0.0,
            persistence_time=0.0,
            integral=0.0,
            diffusion_coeff_gk=0.0
        )
    
    # Compute VAC for different lags
    max_lag = max(2, int(n_vel * max_lag_fraction))
    
    time_lags = []
    vac_values = []
    vac_unnorm_values = []
    
    for lag in range(max_lag):
        # Compute <v(t) · v(t+lag)>
        n_pairs = n_vel - lag
        if n_pairs < 1:
            break
        
        # Dot product of v(t) and v(t+lag)
        dot_products = np.sum(velocities[:n_pairs] * velocities[lag:n_pairs+lag], axis=1)
        mean_dot = np.mean(dot_products)
        
        time_lags.append(lag * frame_interval)
        vac_unnorm_values.append(mean_dot)
        vac_values.append(mean_dot / mean_v_sq if normalize else mean_dot)
    
    time_lags = np.array(time_lags)
    vac = np.array(vac_values)
    vac_unnorm = np.array(vac_unnorm_values)
    
    # Compute decay time (where VAC drops to 1/e ≈ 0.368)
    decay_time = -1.0  # Default: never decays
    threshold = 1.0 / np.e
    for i, v in enumerate(vac):
        if v < threshold:
            # Linear interpolation
            if i > 0:
                t0, t1 = time_lags[i-1], time_lags[i]
                v0, v1 = vac[i-1], vac[i]
                if v0 != v1:
                    decay_time = t0 + (threshold - v0) * (t1 - t0) / (v1 - v0)
                else:
                    decay_time = t0
            else:
                decay_time = 0.0
            break
    
    # Compute persistence time (where VAC first crosses zero)
    persistence_time = -1.0  # Default: never crosses zero
    for i, v in enumerate(vac):
        if v <= 0:
            if i > 0:
                t0, t1 = time_lags[i-1], time_lags[i]
                v0, v1 = vac[i-1], vac[i]
                if v0 != v1:
                    persistence_time = t0 + (0 - v0) * (t1 - t0) / (v1 - v0)
                else:
                    persistence_time = t0
            else:
                persistence_time = 0.0
            break
    
    # Compute integral using trapezoidal rule (for Green-Kubo relation)
    # D = (1/d) * ∫ C_v(τ) dτ, where d = 2 for 2D
    # We integrate the unnormalized VAC
    if len(time_lags) > 1:
        # Use numpy.trapezoid if available (numpy >= 2.0), otherwise trapz
        try:
            integral = np.trapezoid(vac_unnorm, time_lags)
        except AttributeError:
            integral = np.trapz(vac_unnorm, time_lags)
    else:
        integral = 0.0
    
    # Green-Kubo diffusion coefficient (2D)
    diffusion_coeff_gk = integral / 2.0
    
    return VACResult(
        time_lags=time_lags,
        vac=vac,
        vac_unnorm=vac_unnorm,
        decay_time=decay_time if decay_time > 0 else time_lags[-1],
        persistence_time=persistence_time,
        integral=integral,
        diffusion_coeff_gk=max(0, diffusion_coeff_gk)  # D should be non-negative
    )


def compute_directional_persistence(trajectory: Trajectory,
                                   pixel_size: float = 1.0,
                                   frame_interval: float = 1.0) -> Tuple[np.ndarray, np.ndarray, float]:
    """
    Compute directional persistence (cosine of angle between successive velocities).
    
    This is related to VAC but focuses on direction rather than magnitude.
    
    Parameters
    ----------
    trajectory : Trajectory
        Input trajectory
    pixel_size : float
        Physical size per pixel
    frame_interval : float
        Time between frames
        
    Returns
    -------
    angles : ndarray
        Angles between successive velocity vectors (radians)
    cos_angles : ndarray
        Cosines of angles (persistence values, 1 = same direction, -1 = reversal)
    mean_persistence : float
        Mean directional persistence
    """
    positions = trajectory.positions * pixel_size
    
    if len(positions) < 3:
        return np.array([]), np.array([]), 0.0
    
    # Compute velocity vectors
    velocities = np.diff(positions, axis=0)
    
    # Compute angles between successive velocities
    angles = []
    cos_angles = []
    
    for i in range(len(velocities) - 1):
        v1 = velocities[i]
        v2 = velocities[i + 1]
        
        norm1 = np.linalg.norm(v1)
        norm2 = np.linalg.norm(v2)
        
        if norm1 > 1e-10 and norm2 > 1e-10:
            cos_theta = np.dot(v1, v2) / (norm1 * norm2)
            cos_theta = np.clip(cos_theta, -1, 1)  # Numerical stability
            theta = np.arccos(cos_theta)
            
            angles.append(theta)
            cos_angles.append(cos_theta)
    
    angles = np.array(angles)
    cos_angles = np.array(cos_angles)
    
    mean_persistence = np.mean(cos_angles) if len(cos_angles) > 0 else 0.0
    
    return angles, cos_angles, mean_persistence


def analyze_trajectory(trajectory: Trajectory,
                      pixel_size: float = 1.0,
                      frame_interval: float = 1.0,
                      include_vac: bool = True) -> dict:
    """
    Comprehensive trajectory analysis.
    
    Parameters
    ----------
    trajectory : Trajectory
        Input trajectory
    pixel_size : float
        Physical size per pixel
    frame_interval : float
        Time between frames
    include_vac : bool
        Whether to include Velocity Autocorrelation analysis
        
    Returns
    -------
    results : dict
        Dictionary containing all analysis results
    """
    msd_result = compute_msd(trajectory, pixel_size, frame_interval)
    mss_result = compute_mss(trajectory, pixel_size, frame_interval)
    velocities, mean_speed, mean_direction = compute_velocity(
        trajectory, pixel_size, frame_interval
    )
    confinement = compute_confinement_ratio(trajectory, pixel_size)
    
    result = {
        'trajectory_id': trajectory.id,
        'length': trajectory.length,
        'duration': (trajectory.end_frame - trajectory.start_frame) * frame_interval,
        'start_frame': trajectory.start_frame,
        'end_frame': trajectory.end_frame,
        'msd': msd_result,
        'mss': mss_result,
        'diffusion_coefficient': msd_result.diffusion_coeff,
        'alpha': msd_result.slope,
        'mss_slope': mss_result.slope,
        'motion_type': mss_result.motion_type,
        'mean_speed': mean_speed,
        'mean_direction': mean_direction,
        'confinement_ratio': confinement,
        'is_self_similar': mss_result.is_self_similar
    }
    
    if include_vac:
        vac_result = compute_velocity_autocorrelation(
            trajectory, pixel_size, frame_interval
        )
        _, _, mean_persistence = compute_directional_persistence(
            trajectory, pixel_size, frame_interval
        )
        
        result['vac'] = vac_result
        result['vac_decay_time'] = vac_result.decay_time
        result['vac_persistence_time'] = vac_result.persistence_time
        result['diffusion_coeff_gk'] = vac_result.diffusion_coeff_gk
        result['vac_motion_character'] = vac_result.motion_character
        result['directional_persistence'] = mean_persistence
    
    return result


def analyze_all_trajectories(trajectories: List[Trajectory],
                            pixel_size: float = 1.0,
                            frame_interval: float = 1.0,
                            min_length: int = 10,
                            include_vac: bool = True) -> List[dict]:
    """
    Analyze all trajectories.
    
    Parameters
    ----------
    trajectories : list of Trajectory
        Input trajectories
    pixel_size : float
        Physical size per pixel
    frame_interval : float
        Time between frames
    min_length : int
        Minimum trajectory length for analysis
    include_vac : bool
        Whether to include Velocity Autocorrelation analysis
        
    Returns
    -------
    results : list of dict
        Analysis results for each trajectory
    """
    results = []
    for traj in trajectories:
        if traj.length >= min_length:
            results.append(analyze_trajectory(
                traj, pixel_size, frame_interval, include_vac
            ))
    return results


@dataclass
class EnsembleMSDResult:
    """Result of ensemble-averaged MSD analysis."""
    time_lags: np.ndarray  # Time lags (dt)
    msd_mean: np.ndarray  # Mean MSD at each time lag
    msd_std: np.ndarray  # Standard deviation of MSD
    msd_sem: np.ndarray  # Standard error of the mean
    n_trajectories: np.ndarray  # Number of trajectories contributing at each lag
    slope: float  # Slope in log-log plot (alpha)
    intercept: float  # Intercept in log-log plot
    diffusion_coeff: float  # Diffusion coefficient D
    r_squared: float  # R-squared of fit
    
    @property
    def motion_type(self) -> str:
        """Classify motion type based on alpha."""
        if self.slope < 0.3:
            return "confined"
        elif self.slope < 0.7:
            return "subdiffusive"
        elif self.slope < 1.3:
            return "normal"
        elif self.slope < 1.7:
            return "superdiffusive"
        else:
            return "ballistic"


def compute_ensemble_msd(trajectories: List[Trajectory],
                         pixel_size: float = 1.0,
                         frame_interval: float = 1.0,
                         max_lag_fraction: float = 0.25,
                         min_trajectories: int = 3) -> EnsembleMSDResult:
    """
    Compute ensemble-averaged Mean Square Displacement.
    
    This averages the MSD across multiple trajectories at each time lag,
    providing a population-level view of diffusion behavior.
    
    MSD(dt) = <|r(t + dt) - r(t)|^2>_ensemble
    
    Parameters
    ----------
    trajectories : list of Trajectory
        Input trajectories
    pixel_size : float
        Physical size per pixel (µm)
    frame_interval : float
        Time between frames (seconds)
    max_lag_fraction : float
        Maximum lag as fraction of median trajectory length
    min_trajectories : int
        Minimum number of trajectories required at each lag
        
    Returns
    -------
    result : EnsembleMSDResult
        Ensemble MSD analysis result with mean, std, and sem
    """
    if not trajectories:
        return EnsembleMSDResult(
            time_lags=np.array([0.0]),
            msd_mean=np.array([0.0]),
            msd_std=np.array([0.0]),
            msd_sem=np.array([0.0]),
            n_trajectories=np.array([0]),
            slope=0.0,
            intercept=0.0,
            diffusion_coeff=0.0,
            r_squared=0.0
        )
    
    # Determine max lag from median trajectory length
    lengths = [traj.length for traj in trajectories]
    median_length = np.median(lengths)
    max_lag = max(3, int(median_length * max_lag_fraction))
    
    # Collect MSD values for each lag
    msd_by_lag = {dn: [] for dn in range(1, max_lag + 1)}
    
    for traj in trajectories:
        positions = traj.positions * pixel_size
        frames = traj.frames
        n_points = len(positions)
        
        if n_points < 3:
            continue
        
        # Compute squared displacements for each lag
        for dn in range(1, min(max_lag + 1, n_points)):
            sq_displacements = []
            
            for i in range(n_points - 1):
                for j in range(i + 1, n_points):
                    actual_dn = frames[j] - frames[i]
                    if actual_dn == dn:
                        dr = positions[j] - positions[i]
                        sq_displacements.append(np.sum(dr**2))
            
            if sq_displacements:
                # Use mean of this trajectory's displacements at this lag
                msd_by_lag[dn].append(np.mean(sq_displacements))
    
    # Compute ensemble statistics
    time_lags = []
    msd_mean = []
    msd_std = []
    msd_sem = []
    n_trajs = []
    
    for dn in sorted(msd_by_lag.keys()):
        values = msd_by_lag[dn]
        if len(values) >= min_trajectories:
            time_lags.append(dn * frame_interval)
            msd_mean.append(np.mean(values))
            msd_std.append(np.std(values))
            msd_sem.append(np.std(values) / np.sqrt(len(values)))
            n_trajs.append(len(values))
    
    if len(time_lags) < 2:
        return EnsembleMSDResult(
            time_lags=np.array([0.0]),
            msd_mean=np.array([0.0]),
            msd_std=np.array([0.0]),
            msd_sem=np.array([0.0]),
            n_trajectories=np.array([0]),
            slope=0.0,
            intercept=0.0,
            diffusion_coeff=0.0,
            r_squared=0.0
        )
    
    time_lags = np.array(time_lags)
    msd_mean = np.array(msd_mean)
    msd_std = np.array(msd_std)
    msd_sem = np.array(msd_sem)
    n_trajs = np.array(n_trajs)
    
    # Fit in log-log space
    valid = (time_lags > 0) & (msd_mean > 0)
    if np.sum(valid) >= 2:
        log_t = np.log(time_lags[valid])
        log_msd = np.log(msd_mean[valid])
        
        # Linear regression
        A = np.vstack([log_t, np.ones(len(log_t))]).T
        result = np.linalg.lstsq(A, log_msd, rcond=None)
        slope, intercept = result[0]
        
        # R-squared
        y_pred = slope * log_t + intercept
        ss_res = np.sum((log_msd - y_pred)**2)
        ss_tot = np.sum((log_msd - np.mean(log_msd))**2)
        r_squared = 1 - (ss_res / ss_tot) if ss_tot > 0 else 0.0
        
        # Diffusion coefficient: MSD = 4D*t^alpha, so log(MSD) = log(4D) + alpha*log(t)
        diffusion_coeff = np.exp(intercept) / 4.0
    else:
        slope = 0.0
        intercept = 0.0
        r_squared = 0.0
        diffusion_coeff = 0.0
    
    return EnsembleMSDResult(
        time_lags=time_lags,
        msd_mean=msd_mean,
        msd_std=msd_std,
        msd_sem=msd_sem,
        n_trajectories=n_trajs,
        slope=slope,
        intercept=intercept,
        diffusion_coeff=diffusion_coeff,
        r_squared=r_squared
    )
