"""
Visualization Module

Plotting functions for particle tracking results.
"""

import numpy as np
from typing import List, Optional, Tuple, Union
import warnings


def plot_frame_with_particles(movie: np.ndarray,
                             particles: List,
                             frame_idx: int = 0,
                             ax=None,
                             marker_size: float = 100,
                             marker_color: str = 'lime',
                             linewidth: float = 1.5,
                             show_ids: bool = False,
                             **imshow_kwargs):
    """
    Plot a frame with detected particles overlaid as hollow circles.
    
    Parameters
    ----------
    movie : ndarray
        Movie array (frames, height, width)
    particles : list of list of Particle
        Detected particles per frame
    frame_idx : int
        Frame index to plot
    ax : matplotlib axes, optional
        Axes to plot on
    marker_size : float
        Marker size (area)
    marker_color : str
        Marker color (default: lime green for visibility)
    linewidth : float
        Width of circle outline
    show_ids : bool
        Whether to show particle indices
        
    Returns
    -------
    ax : matplotlib axes
    """
    import matplotlib.pyplot as plt
    
    if ax is None:
        fig, ax = plt.subplots(figsize=(8, 8))
    
    # Plot frame
    defaults = {'cmap': 'gray', 'origin': 'upper'}
    defaults.update(imshow_kwargs)
    ax.imshow(movie[frame_idx], **defaults)
    
    # Plot particles as hollow circles
    frame_particles = particles[frame_idx]
    if frame_particles:
        xs = [p.x for p in frame_particles]
        ys = [p.y for p in frame_particles]
        ax.scatter(xs, ys, s=marker_size, 
                  facecolors='none',  # Hollow (中抜き)
                  edgecolors=marker_color, 
                  linewidths=linewidth)
        
        if show_ids:
            for i, p in enumerate(frame_particles):
                ax.annotate(str(i), (p.x + 3, p.y), fontsize=8, color=marker_color)
    
    ax.set_title(f'Frame {frame_idx} ({len(frame_particles)} particles)')
    ax.set_xlabel('x (pixels)')
    ax.set_ylabel('y (pixels)')
    
    return ax


def plot_trajectories(trajectories: List,
                     ax=None,
                     color_by: str = 'id',
                     cmap: str = 'viridis',
                     alpha: float = 0.7,
                     linewidth: float = 1.0,
                     show_start: bool = True,
                     show_end: bool = True,
                     background: Optional[np.ndarray] = None,
                     **kwargs):
    """
    Plot all trajectories.
    
    Parameters
    ----------
    trajectories : list of Trajectory
        Trajectories to plot
    ax : matplotlib axes, optional
        Axes to plot on
    color_by : str
        'id', 'length', 'frame', or 'velocity'
    cmap : str
        Colormap name
    alpha : float
        Line transparency
    linewidth : float
        Line width
    show_start : bool
        Mark trajectory start
    show_end : bool
        Mark trajectory end
    background : ndarray, optional
        Background image to show
        
    Returns
    -------
    ax : matplotlib axes
    """
    import matplotlib.pyplot as plt
    from matplotlib import cm
    
    if ax is None:
        fig, ax = plt.subplots(figsize=(10, 10))
    
    if background is not None:
        ax.imshow(background, cmap='gray', alpha=0.5)
    
    # Get colormap
    cmap_func = cm.get_cmap(cmap)
    
    # Determine colors
    if color_by == 'id':
        colors = [cmap_func(t.id / max(1, len(trajectories))) for t in trajectories]
    elif color_by == 'length':
        lengths = [t.length for t in trajectories]
        max_len = max(lengths) if lengths else 1
        colors = [cmap_func(l / max_len) for l in lengths]
    elif color_by == 'frame':
        starts = [t.start_frame for t in trajectories]
        max_frame = max(starts) if starts else 1
        colors = [cmap_func(s / max_frame) for s in starts]
    else:
        colors = [cmap_func(0.5)] * len(trajectories)
    
    for traj, color in zip(trajectories, colors):
        positions = traj.positions
        ax.plot(positions[:, 0], positions[:, 1], 
               color=color, alpha=alpha, linewidth=linewidth, **kwargs)
        
        if show_start:
            ax.plot(positions[0, 0], positions[0, 1], 
                   'o', color=color, markersize=4)
        if show_end:
            ax.plot(positions[-1, 0], positions[-1, 1], 
                   's', color=color, markersize=4)
    
    ax.set_xlabel('x (pixels)')
    ax.set_ylabel('y (pixels)')
    ax.set_title(f'{len(trajectories)} trajectories')
    ax.set_aspect('equal')
    
    return ax


def plot_trajectory(trajectory,
                   ax=None,
                   color: str = 'blue',
                   linewidth: float = 2.0,
                   show_points: bool = True,
                   color_by_time: bool = False,
                   cmap: str = 'viridis',
                   **kwargs):
    """
    Plot a single trajectory.
    
    Parameters
    ----------
    trajectory : Trajectory
        Trajectory to plot
    ax : matplotlib axes, optional
        Axes to plot on
    color : str
        Line color
    linewidth : float
        Line width
    show_points : bool
        Show individual points
    color_by_time : bool
        Color by time instead of solid color
    cmap : str
        Colormap for time coloring
        
    Returns
    -------
    ax : matplotlib axes
    """
    import matplotlib.pyplot as plt
    from matplotlib import cm
    from matplotlib.collections import LineCollection
    
    if ax is None:
        fig, ax = plt.subplots(figsize=(8, 8))
    
    positions = trajectory.positions
    
    if color_by_time:
        # Create colored line segments
        points = positions.reshape(-1, 1, 2)
        segments = np.concatenate([points[:-1], points[1:]], axis=1)
        
        norm = plt.Normalize(0, len(positions))
        lc = LineCollection(segments, cmap=cmap, norm=norm)
        lc.set_array(np.arange(len(positions)))
        lc.set_linewidth(linewidth)
        ax.add_collection(lc)
        plt.colorbar(lc, ax=ax, label='Frame')
    else:
        ax.plot(positions[:, 0], positions[:, 1], 
               color=color, linewidth=linewidth, **kwargs)
    
    if show_points:
        ax.scatter(positions[:, 0], positions[:, 1], 
                  c=np.arange(len(positions)) if color_by_time else color,
                  cmap=cmap if color_by_time else None,
                  s=20, zorder=5)
    
    # Mark start and end
    ax.plot(positions[0, 0], positions[0, 1], 'go', markersize=10, label='Start')
    ax.plot(positions[-1, 0], positions[-1, 1], 'rs', markersize=10, label='End')
    
    ax.set_xlabel('x')
    ax.set_ylabel('y')
    ax.set_title(f'Trajectory {trajectory.id} ({trajectory.length} points)')
    ax.legend()
    ax.set_aspect('equal')
    
    return ax


def plot_msd(msd_result,
            ax=None,
            show_fit: bool = True,
            color: str = 'blue',
            fit_color: str = 'red'):
    """
    Plot MSD vs time lag.
    
    Parameters
    ----------
    msd_result : MSDResult
        MSD analysis result
    ax : matplotlib axes, optional
        Axes to plot on
    show_fit : bool
        Show linear fit
    color : str
        Data point color
    fit_color : str
        Fit line color
        
    Returns
    -------
    ax : matplotlib axes
    """
    import matplotlib.pyplot as plt
    
    if ax is None:
        fig, ax = plt.subplots(figsize=(8, 6))
    
    # Plot data
    ax.loglog(msd_result.time_lags, msd_result.msd, 
             'o', color=color, label='Data')
    
    if show_fit and len(msd_result.time_lags) > 1:
        # Plot fit
        t_fit = np.logspace(
            np.log10(msd_result.time_lags.min()),
            np.log10(msd_result.time_lags.max()),
            100
        )
        msd_fit = np.exp(msd_result.intercept) * t_fit ** msd_result.slope
        ax.loglog(t_fit, msd_fit, '--', color=fit_color, 
                 label=f'Fit: α={msd_result.slope:.3f}')
    
    ax.set_xlabel('Time lag')
    ax.set_ylabel('MSD')
    ax.set_title(f'Mean Square Displacement (D={msd_result.diffusion_coeff:.4f})')
    ax.legend()
    ax.grid(True, alpha=0.3)
    
    return ax


def plot_mss(mss_result,
            ax=None,
            show_fit: bool = True,
            color: str = 'blue',
            fit_color: str = 'red'):
    """
    Plot Moment Scaling Spectrum.
    
    Parameters
    ----------
    mss_result : MSSResult
        MSS analysis result
    ax : matplotlib axes, optional
        Axes to plot on
    show_fit : bool
        Show linear fit through origin
    color : str
        Data point color
    fit_color : str
        Fit line color
        
    Returns
    -------
    ax : matplotlib axes
    """
    import matplotlib.pyplot as plt
    
    if ax is None:
        fig, ax = plt.subplots(figsize=(8, 6))
    
    # Plot data
    ax.plot(mss_result.orders, mss_result.scaling_exponents, 
           'o-', color=color, label='Data')
    
    if show_fit:
        # Plot fit through origin
        m_fit = np.linspace(0, mss_result.orders.max(), 100)
        gamma_fit = mss_result.slope * m_fit
        ax.plot(m_fit, gamma_fit, '--', color=fit_color, 
               label=f'Fit: slope={mss_result.slope:.3f}')
    
    # Reference lines
    ax.plot([0, mss_result.orders.max()], [0, 0.5 * mss_result.orders.max()], 
           ':', color='gray', alpha=0.5, label='Normal diffusion (0.5)')
    ax.plot([0, mss_result.orders.max()], [0, mss_result.orders.max()], 
           ':', color='gray', alpha=0.5, label='Ballistic (1.0)')
    
    ax.set_xlabel('Moment order (m)')
    ax.set_ylabel('Scaling exponent (γ)')
    ax.set_title(f'Moment Scaling Spectrum ({mss_result.motion_type})')
    ax.legend()
    ax.grid(True, alpha=0.3)
    ax.set_xlim(0, None)
    ax.set_ylim(0, None)
    
    return ax


def plot_ensemble_msd(ensemble_result,
                      ax=None,
                      show_fit: bool = True,
                      show_error: str = 'sem',
                      color: str = 'blue',
                      fit_color: str = 'red',
                      log_scale: bool = True,
                      show_reference: bool = True):
    """
    Plot ensemble-averaged MSD with error bars/bands.
    
    Parameters
    ----------
    ensemble_result : EnsembleMSDResult
        Ensemble MSD analysis result
    ax : matplotlib axes, optional
        Axes to plot on
    show_fit : bool
        Show power-law fit
    show_error : str
        Error display: 'sem' (standard error), 'std' (standard deviation), 
        'band' (shaded band), or None
    color : str
        Data color
    fit_color : str
        Fit line color
    log_scale : bool
        Use log-log scale
    show_reference : bool
        Show reference lines for normal diffusion
        
    Returns
    -------
    ax : matplotlib axes
    """
    import matplotlib.pyplot as plt
    
    if ax is None:
        fig, ax = plt.subplots(figsize=(8, 6))
    
    t = ensemble_result.time_lags
    msd = ensemble_result.msd_mean
    
    if len(t) < 2:
        ax.text(0.5, 0.5, 'Not enough data', ha='center', va='center', transform=ax.transAxes)
        return ax
    
    # Choose error values
    if show_error == 'sem':
        error = ensemble_result.msd_sem
        error_label = 'SEM'
    elif show_error == 'std':
        error = ensemble_result.msd_std
        error_label = 'STD'
    else:
        error = None
        error_label = None
    
    if log_scale:
        # Log-log plot
        ax.set_xscale('log')
        ax.set_yscale('log')
        
        # Plot data
        ax.plot(t, msd, 'o-', color=color, label='Ensemble MSD', markersize=6)
        
        # Error bands
        if error is not None and show_error == 'band':
            ax.fill_between(t, msd - error, msd + error, 
                           alpha=0.3, color=color, label=f'±{error_label}')
        elif error is not None:
            # Error bars (need to handle log scale carefully)
            ax.errorbar(t, msd, yerr=error, fmt='none', color=color, 
                       capsize=3, alpha=0.7)
        
        # Fit line
        if show_fit and len(t) > 1:
            t_fit = np.logspace(np.log10(t.min()), np.log10(t.max()), 100)
            msd_fit = np.exp(ensemble_result.intercept) * t_fit ** ensemble_result.slope
            ax.plot(t_fit, msd_fit, '--', color=fit_color, lw=2,
                   label=f'Fit: α={ensemble_result.slope:.3f}, D={ensemble_result.diffusion_coeff:.4f}')
        
        # Reference line for normal diffusion
        if show_reference and len(t) > 1:
            # Normal diffusion: MSD = 4Dt, slope = 1
            t_ref = np.logspace(np.log10(t.min()), np.log10(t.max()), 50)
            msd_ref = 4 * ensemble_result.diffusion_coeff * t_ref
            ax.plot(t_ref, msd_ref, ':', color='gray', alpha=0.5, 
                   label='Normal diffusion (α=1)')
    else:
        # Linear plot
        ax.plot(t, msd, 'o-', color=color, label='Ensemble MSD', markersize=6)
        
        if error is not None and show_error == 'band':
            ax.fill_between(t, msd - error, msd + error, 
                           alpha=0.3, color=color, label=f'±{error_label}')
        elif error is not None:
            ax.errorbar(t, msd, yerr=error, fmt='none', color=color, 
                       capsize=3, alpha=0.7)
        
        if show_fit:
            t_fit = np.linspace(t.min(), t.max(), 100)
            msd_fit = np.exp(ensemble_result.intercept) * t_fit ** ensemble_result.slope
            ax.plot(t_fit, msd_fit, '--', color=fit_color, lw=2,
                   label=f'Fit: α={ensemble_result.slope:.3f}')
    
    ax.set_xlabel('Time lag (τ)')
    ax.set_ylabel('MSD [µm²]')
    ax.set_title(f'Ensemble MSD (n={ensemble_result.n_trajectories[0]} trajectories)\n'
                 f'α={ensemble_result.slope:.3f}, D={ensemble_result.diffusion_coeff:.4f} µm²/s')
    ax.legend(loc='upper left')
    ax.grid(True, alpha=0.3, which='both')
    
    return ax


def plot_msd_population(trajectories,
                        pixel_size: float = 1.0,
                        frame_interval: float = 1.0,
                        ax=None,
                        alpha: float = 0.3,
                        color: str = 'blue',
                        show_ensemble: bool = True,
                        ensemble_color: str = 'red',
                        log_scale: bool = True,
                        max_lag_fraction: float = 0.25):
    """
    Plot MSD curves for all trajectories with optional ensemble average overlay.
    
    Parameters
    ----------
    trajectories : list of Trajectory
        Input trajectories
    pixel_size : float
        Physical size per pixel
    frame_interval : float
        Time between frames
    ax : matplotlib axes, optional
        Axes to plot on
    alpha : float
        Alpha for individual curves
    color : str
        Color for individual curves
    show_ensemble : bool
        Overlay ensemble average
    ensemble_color : str
        Color for ensemble average
    log_scale : bool
        Use log-log scale
    max_lag_fraction : float
        Maximum lag fraction
        
    Returns
    -------
    ax : matplotlib axes
    """
    import matplotlib.pyplot as plt
    from .analysis import compute_msd, compute_ensemble_msd
    
    if ax is None:
        fig, ax = plt.subplots(figsize=(8, 6))
    
    if log_scale:
        ax.set_xscale('log')
        ax.set_yscale('log')
    
    # Plot individual MSDs
    for traj in trajectories:
        if traj.length < 5:
            continue
        
        msd_result = compute_msd(traj, pixel_size, frame_interval, max_lag_fraction)
        if len(msd_result.time_lags) > 1 and np.any(msd_result.msd > 0):
            ax.plot(msd_result.time_lags, msd_result.msd, '-', 
                   color=color, alpha=alpha, linewidth=0.8)
    
    # Ensemble average
    if show_ensemble:
        ensemble = compute_ensemble_msd(trajectories, pixel_size, frame_interval, 
                                        max_lag_fraction)
        if len(ensemble.time_lags) > 1:
            ax.plot(ensemble.time_lags, ensemble.msd_mean, 'o-', 
                   color=ensemble_color, linewidth=2, markersize=5,
                   label=f'Ensemble (α={ensemble.slope:.2f}, D={ensemble.diffusion_coeff:.4f})')
            ax.fill_between(ensemble.time_lags, 
                           ensemble.msd_mean - ensemble.msd_sem,
                           ensemble.msd_mean + ensemble.msd_sem,
                           color=ensemble_color, alpha=0.3)
    
    ax.set_xlabel('Time lag (τ)')
    ax.set_ylabel('MSD [µm²]')
    ax.set_title(f'MSD Population (n={len(trajectories)} trajectories)')
    ax.legend()
    ax.grid(True, alpha=0.3, which='both')
    
    return ax


def plot_diffusion_histogram(results: List[dict],
                            ax=None,
                            bins: int = 30,
                            log_scale: bool = True,
                            color: str = 'steelblue',
                            show_kde: bool = True,
                            kde_color: str = 'red',
                            kde_bw: str = 'scott'):
    """
    Plot histogram of diffusion coefficients with optional KDE overlay.
    
    Parameters
    ----------
    results : list of dict
        Analysis results
    ax : matplotlib axes, optional
        Axes to plot on
    bins : int
        Number of histogram bins
    log_scale : bool
        Use log scale for x-axis
    color : str
        Histogram color
    show_kde : bool
        Show Kernel Density Estimation curve
    kde_color : str
        KDE curve color
    kde_bw : str
        KDE bandwidth method ('scott', 'silverman', or float)
        
    Returns
    -------
    ax : matplotlib axes
    """
    import matplotlib.pyplot as plt
    
    if ax is None:
        fig, ax = plt.subplots(figsize=(8, 6))
    
    d_coeffs = [r['diffusion_coefficient'] for r in results]
    d_coeffs = np.array([d for d in d_coeffs if d > 0])  # Filter zeros
    
    if len(d_coeffs) == 0:
        ax.text(0.5, 0.5, 'No valid data', ha='center', va='center', transform=ax.transAxes)
        return ax
    
    # Use log10 values if log_scale
    if log_scale:
        data = np.log10(d_coeffs)
        xlabel = 'log₁₀(D) [µm²/s]'
    else:
        data = d_coeffs
        xlabel = 'D [µm²/s]'
    
    # Plot histogram (normalized for KDE comparison)
    if show_kde:
        ax.hist(data, bins=bins, color=color, edgecolor='black', 
                alpha=0.7, density=True, label='Histogram')
    else:
        ax.hist(data, bins=bins, color=color, edgecolor='black', alpha=0.7)
    
    # KDE overlay
    if show_kde and len(d_coeffs) > 2:
        from scipy import stats
        
        # Compute KDE
        kde = stats.gaussian_kde(data, bw_method=kde_bw)
        x_range = np.linspace(data.min() - 0.5, data.max() + 0.5, 200)
        kde_values = kde(x_range)
        
        ax.plot(x_range, kde_values, color=kde_color, lw=2, label='KDE')
    
    ax.set_xlabel(xlabel)
    ax.set_ylabel('Density' if show_kde else 'Count')
    ax.set_title(f'Diffusion Coefficient Distribution (n={len(d_coeffs)})')
    
    # Add statistics
    mean_d = np.mean(d_coeffs)
    median_d = np.median(d_coeffs)
    std_d = np.std(d_coeffs)
    
    if log_scale:
        mean_pos = np.log10(mean_d)
        median_pos = np.log10(median_d)
    else:
        mean_pos = mean_d
        median_pos = median_d
    
    ax.axvline(mean_pos, color='darkgreen', linestyle='--', lw=1.5,
               label=f'Mean: {mean_d:.4f}')
    ax.axvline(median_pos, color='orange', linestyle='--', lw=1.5,
               label=f'Median: {median_d:.4f}')
    
    ax.legend(loc='upper right')
    
    # Add text box with statistics
    stats_text = f'n = {len(d_coeffs)}\nMean = {mean_d:.4f}\nMedian = {median_d:.4f}\nStd = {std_d:.4f}'
    ax.text(0.02, 0.98, stats_text, transform=ax.transAxes, fontsize=9,
            verticalalignment='top', bbox=dict(boxstyle='round', facecolor='white', alpha=0.8))
    
    return ax


def plot_diffusion_kde(results: List[dict],
                       ax=None,
                       log_scale: bool = True,
                       color: str = 'steelblue',
                       fill: bool = True,
                       fill_alpha: float = 0.3,
                       bw_method: str = 'scott',
                       show_rug: bool = True):
    """
    Plot Kernel Density Estimation of diffusion coefficients.
    
    Parameters
    ----------
    results : list of dict
        Analysis results
    ax : matplotlib axes, optional
        Axes to plot on
    log_scale : bool
        Use log scale for x-axis
    color : str
        Line and fill color
    fill : bool
        Fill under KDE curve
    fill_alpha : float
        Alpha for fill
    bw_method : str
        Bandwidth method ('scott', 'silverman', or float)
    show_rug : bool
        Show rug plot (individual data points)
        
    Returns
    -------
    ax : matplotlib axes
    """
    import matplotlib.pyplot as plt
    from scipy import stats
    
    if ax is None:
        fig, ax = plt.subplots(figsize=(8, 6))
    
    d_coeffs = [r['diffusion_coefficient'] for r in results]
    d_coeffs = np.array([d for d in d_coeffs if d > 0])
    
    if len(d_coeffs) < 3:
        ax.text(0.5, 0.5, 'Not enough data for KDE', ha='center', va='center', transform=ax.transAxes)
        return ax
    
    # Use log10 values if log_scale
    if log_scale:
        data = np.log10(d_coeffs)
        xlabel = 'log₁₀(D) [µm²/s]'
    else:
        data = d_coeffs
        xlabel = 'D [µm²/s]'
    
    # Compute KDE
    kde = stats.gaussian_kde(data, bw_method=bw_method)
    x_range = np.linspace(data.min() - 0.5, data.max() + 0.5, 200)
    kde_values = kde(x_range)
    
    # Plot KDE
    ax.plot(x_range, kde_values, color=color, lw=2)
    
    if fill:
        ax.fill_between(x_range, kde_values, alpha=fill_alpha, color=color)
    
    # Rug plot
    if show_rug:
        ax.plot(data, np.zeros_like(data) - 0.02 * kde_values.max(), '|', 
                color=color, alpha=0.5, markersize=10)
    
    ax.set_xlabel(xlabel)
    ax.set_ylabel('Density')
    ax.set_title(f'Diffusion Coefficient KDE (n={len(d_coeffs)})')
    
    # Statistics
    mean_d = np.mean(d_coeffs)
    median_d = np.median(d_coeffs)
    
    if log_scale:
        ax.axvline(np.log10(mean_d), color='darkgreen', linestyle='--', 
                   label=f'Mean: {mean_d:.4f}')
        ax.axvline(np.log10(median_d), color='orange', linestyle='--',
                   label=f'Median: {median_d:.4f}')
    else:
        ax.axvline(mean_d, color='darkgreen', linestyle='--', label=f'Mean: {mean_d:.4f}')
        ax.axvline(median_d, color='orange', linestyle='--', label=f'Median: {median_d:.4f}')
    
    ax.legend()
    ax.set_ylim(bottom=0)
    
    return ax


def plot_motion_type_pie(results: List[dict], ax=None):
    """
    Plot pie chart of motion types.
    
    Parameters
    ----------
    results : list of dict
        Analysis results
    ax : matplotlib axes, optional
        Axes to plot on
        
    Returns
    -------
    ax : matplotlib axes
    """
    import matplotlib.pyplot as plt
    
    if ax is None:
        fig, ax = plt.subplots(figsize=(8, 8))
    
    # Count motion types
    motion_types = {}
    for r in results:
        mt = r['motion_type']
        motion_types[mt] = motion_types.get(mt, 0) + 1
    
    labels = list(motion_types.keys())
    sizes = list(motion_types.values())
    
    ax.pie(sizes, labels=labels, autopct='%1.1f%%', startangle=90)
    ax.set_title('Motion Type Distribution')
    
    return ax


def create_tracking_movie(movie: np.ndarray,
                         trajectories: List,
                         output_path: str,
                         fps: int = 10,
                         trail_length: int = 10,
                         cmap: str = 'viridis'):
    """
    Create movie with trajectory overlay.
    
    Parameters
    ----------
    movie : ndarray
        Original movie
    trajectories : list of Trajectory
        Trajectories to overlay
    output_path : str
        Output file path
    fps : int
        Frames per second
    trail_length : int
        Number of frames to show trajectory trail
    cmap : str
        Colormap for trajectories
    """
    import matplotlib.pyplot as plt
    from matplotlib import animation, cm
    
    fig, ax = plt.subplots(figsize=(8, 8))
    
    # Precompute colors
    cmap_func = cm.get_cmap(cmap)
    colors = {t.id: cmap_func(t.id / len(trajectories)) for t in trajectories}
    
    # Create position lookup
    positions_by_frame = {}
    for traj in trajectories:
        for i, p in enumerate(traj.particles):
            if p.frame not in positions_by_frame:
                positions_by_frame[p.frame] = []
            positions_by_frame[p.frame].append((traj.id, p.x, p.y, i))
    
    def update(frame):
        ax.clear()
        ax.imshow(movie[frame], cmap='gray')
        
        # Draw trails
        for traj in trajectories:
            frames = traj.frames
            mask = (frames <= frame) & (frames > frame - trail_length)
            if np.any(mask):
                pos = traj.positions[mask]
                ax.plot(pos[:, 0], pos[:, 1], '-', 
                       color=colors[traj.id], alpha=0.5, linewidth=1)
        
        # Draw current positions
        if frame in positions_by_frame:
            for traj_id, x, y, _ in positions_by_frame[frame]:
                ax.plot(x, y, 'o', color=colors[traj_id], markersize=5)
        
        ax.set_title(f'Frame {frame}')
        ax.axis('off')
        return []
    
    anim = animation.FuncAnimation(fig, update, frames=len(movie), blit=True)
    
    # Save
    if output_path.endswith('.gif'):
        anim.save(output_path, writer='pillow', fps=fps)
    else:
        anim.save(output_path, writer='ffmpeg', fps=fps)
    
    plt.close(fig)


def plot_detection_comparison(image: np.ndarray,
                             radius: int = 3,
                             percentiles: List[float] = [0.1, 0.5, 1.0, 2.0, 5.0],
                             cutoff_score: float = 0.0,
                             figsize: Tuple[int, int] = (16, 6)):
    """
    Compare detection with different percentile thresholds.
    Useful for choosing the best parameters for your data.
    
    Parameters
    ----------
    image : ndarray
        Input image
    radius : int
        Particle radius
    percentiles : list of float
        Percentile values to compare
    cutoff_score : float
        Non-particle discrimination cutoff
    figsize : tuple
        Figure size
        
    Returns
    -------
    fig : matplotlib Figure
    """
    import matplotlib.pyplot as plt
    from .detection import detect_particles, restore_image
    
    n_cols = len(percentiles) + 1
    fig, axes = plt.subplots(2, n_cols, figsize=figsize)
    
    # Normalize and filter image
    img_min, img_max = image.min(), image.max()
    if img_max > img_min:
        normalized = (image.astype(float) - img_min) / (img_max - img_min)
    else:
        normalized = image.astype(float)
    filtered = restore_image(normalized, radius, 1.0)
    
    # Show original and filtered
    axes[0, 0].imshow(image, cmap='gray')
    axes[0, 0].set_title('Original')
    axes[0, 0].axis('off')
    
    axes[1, 0].imshow(filtered, cmap='gray')
    axes[1, 0].set_title('Filtered')
    axes[1, 0].axis('off')
    
    # Compare different percentiles
    for i, pct in enumerate(percentiles):
        particles = detect_particles(
            image, frame_idx=0, radius=radius,
            percentile=pct, cutoff_score=cutoff_score
        )
        
        col = i + 1
        
        # Top row: filtered image
        axes[0, col].imshow(filtered, cmap='gray')
        axes[0, col].set_title(f'percentile={pct}')
        axes[0, col].axis('off')
        
        # Bottom row: original with detections
        axes[1, col].imshow(image, cmap='gray')
        if particles:
            xs = [p.x for p in particles]
            ys = [p.y for p in particles]
            axes[1, col].scatter(xs, ys, s=80, 
                               facecolors='none', edgecolors='lime', linewidths=1.5)
        axes[1, col].set_title(f'{len(particles)} particles')
        axes[1, col].axis('off')
    
    plt.tight_layout()
    return fig


def plot_vac(vac_result, ax=None, show_zero_line: bool = True,
             show_decay_time: bool = True, **kwargs):
    """
    Plot Velocity Autocorrelation Function.
    
    Parameters
    ----------
    vac_result : VACResult
        Result from compute_velocity_autocorrelation
    ax : matplotlib axes, optional
        Axes to plot on
    show_zero_line : bool
        Show horizontal line at VAC=0
    show_decay_time : bool
        Show vertical line at decay time
        
    Returns
    -------
    ax : matplotlib axes
    """
    import matplotlib.pyplot as plt
    
    if ax is None:
        fig, ax = plt.subplots(figsize=(8, 5))
    
    # Plot VAC
    defaults = {'linewidth': 2, 'color': 'C0'}
    defaults.update(kwargs)
    ax.plot(vac_result.time_lags, vac_result.vac, '-', **defaults)
    
    # Zero line
    if show_zero_line:
        ax.axhline(y=0, color='gray', linestyle='--', alpha=0.5)
    
    # Decay time marker
    if show_decay_time and vac_result.decay_time > 0:
        ax.axvline(x=vac_result.decay_time, color='r', linestyle=':', 
                   alpha=0.7, label=f'τ_decay = {vac_result.decay_time:.3f}')
    
    # 1/e line
    ax.axhline(y=1/np.e, color='gray', linestyle=':', alpha=0.3)
    
    ax.set_xlabel('Time lag τ')
    ax.set_ylabel('VAC(τ)')
    ax.set_title('Velocity Autocorrelation')
    
    if show_decay_time and vac_result.decay_time > 0:
        ax.legend()
    
    return ax


def plot_vac_population(results: List[dict], ax=None, 
                        alpha: float = 0.3, show_mean: bool = True):
    """
    Plot VAC for multiple trajectories.
    
    Parameters
    ----------
    results : list of dict
        Results from analyze_all_trajectories (must include 'vac')
    ax : matplotlib axes, optional
        Axes to plot on
    alpha : float
        Transparency for individual curves
    show_mean : bool
        Whether to plot the mean VAC
        
    Returns
    -------
    ax : matplotlib axes
    """
    import matplotlib.pyplot as plt
    
    if ax is None:
        fig, ax = plt.subplots(figsize=(8, 5))
    
    # Collect all VACs
    all_vacs = []
    max_len = 0
    
    for r in results:
        if 'vac' in r:
            vac = r['vac']
            ax.plot(vac.time_lags, vac.vac, '-', color='C0', alpha=alpha)
            all_vacs.append((vac.time_lags, vac.vac))
            max_len = max(max_len, len(vac.time_lags))
    
    # Compute and plot mean
    if show_mean and all_vacs:
        # Interpolate to common time base
        common_times = all_vacs[0][0]  # Use first trajectory's times
        mean_vac = np.zeros(len(common_times))
        counts = np.zeros(len(common_times))
        
        for times, vac in all_vacs:
            for i, t in enumerate(common_times):
                if i < len(vac):
                    mean_vac[i] += vac[i]
                    counts[i] += 1
        
        mean_vac = mean_vac / np.maximum(counts, 1)
        ax.plot(common_times, mean_vac, '-', color='red', linewidth=2, 
                label='Mean VAC')
        ax.legend()
    
    ax.axhline(y=0, color='gray', linestyle='--', alpha=0.5)
    ax.set_xlabel('Time lag τ')
    ax.set_ylabel('VAC(τ)')
    ax.set_title(f'Velocity Autocorrelation ({len(all_vacs)} trajectories)')
    
    return ax


def plot_persistence_histogram(results: List[dict], ax=None, bins: int = 30):
    """
    Plot histogram of directional persistence values.
    
    Parameters
    ----------
    results : list of dict
        Results from analyze_all_trajectories
    ax : matplotlib axes, optional
        Axes to plot on
    bins : int
        Number of histogram bins
        
    Returns
    -------
    ax : matplotlib axes
    """
    import matplotlib.pyplot as plt
    
    if ax is None:
        fig, ax = plt.subplots(figsize=(8, 5))
    
    persistence_values = [r['directional_persistence'] for r in results 
                         if 'directional_persistence' in r]
    
    if persistence_values:
        ax.hist(persistence_values, bins=bins, edgecolor='black', alpha=0.7)
        
        mean_p = np.mean(persistence_values)
        ax.axvline(x=mean_p, color='r', linestyle='--', 
                   label=f'Mean = {mean_p:.3f}')
        ax.axvline(x=0, color='gray', linestyle=':', alpha=0.5)
        
        ax.set_xlabel('Directional Persistence (cos θ)')
        ax.set_ylabel('Count')
        ax.set_title('Directional Persistence Distribution')
        ax.set_xlim(-1, 1)
        ax.legend()
    
    return ax


def plot_vac_decay_times(results: List[dict], ax=None, bins: int = 20):
    """
    Plot histogram of VAC decay times.
    
    Parameters
    ----------
    results : list of dict
        Results from analyze_all_trajectories
    ax : matplotlib axes, optional
        Axes to plot on
    bins : int
        Number of histogram bins
        
    Returns
    -------
    ax : matplotlib axes
    """
    import matplotlib.pyplot as plt
    
    if ax is None:
        fig, ax = plt.subplots(figsize=(8, 5))
    
    decay_times = [r['vac_decay_time'] for r in results 
                  if 'vac_decay_time' in r and r['vac_decay_time'] > 0]
    
    if decay_times:
        ax.hist(decay_times, bins=bins, edgecolor='black', alpha=0.7)
        
        mean_t = np.mean(decay_times)
        ax.axvline(x=mean_t, color='r', linestyle='--', 
                   label=f'Mean = {mean_t:.3f}')
        
        ax.set_xlabel('VAC Decay Time (τ)')
        ax.set_ylabel('Count')
        ax.set_title('VAC Decay Time Distribution')
        ax.legend()
    
    return ax
