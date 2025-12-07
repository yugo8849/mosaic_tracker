"""
Mosaic Tracker - Python implementation of MosaicSuite Particle Tracker

This package provides single particle tracking and trajectory analysis
based on the algorithm described in:

Sbalzarini & Koumoutsakos (2005) "Feature point tracking and trajectory 
analysis for video imaging in cell biology" J. Struct. Biol. 151(2):182-195

Main Features:
- Particle detection using local maxima and intensity thresholding
- Non-particle discrimination using intensity moment clustering  
- Trajectory linking using graph-theoretic optimization
- Mean Square Displacement (MSD) analysis
- Moment Scaling Spectrum (MSS) analysis
- Diffusion coefficient estimation
- Motion type classification
- GPU acceleration (optional, requires CuPy)

Example Usage:
-------------
>>> from mosaic_tracker import ParticleTracker, TrackerParameters
>>> 
>>> # Configure tracking parameters (following MosaicSuite conventions)
>>> params = TrackerParameters(
...     radius=3,            # Particle radius in pixels
...     percentile=0.5,      # Top 0.5% brightest pixels as candidates
...     cutoff_score=0,      # Non-particle discrimination (0=disabled)
...     max_displacement=5,  # Max displacement per frame
...     link_range=2,        # Frames to look ahead for linking
...     pixel_size=0.08,     # µm per pixel
...     frame_interval=0.05  # seconds between frames
... )
>>> 
>>> # Create tracker and run
>>> tracker = ParticleTracker(params)
>>> tracker.load_movie("movie.tif")
>>> tracker.detect_particles(verbose=True)
>>> tracker.link_trajectories()
>>> results = tracker.analyze()
>>> 
>>> # Access results
>>> print(tracker.summary())
>>> df = tracker.get_summary_dataframe()
"""

__version__ = "0.2.0"
__author__ = "Cell Cycle Laboratory"

# Main classes
from .tracker import ParticleTracker, TrackerParameters

# Detection
from .detection import (
    Particle,
    detect_particles,
    detect_all_frames,
    restore_image,
    find_local_maxima,
    refine_position,
    discriminate_particles,
    preview_detection,
    HAS_CUPY
)

# JAX support (for Mac Apple Silicon)
try:
    from .detection_jax import (
        HAS_JAX,
        JAX_BACKEND,
        check_jax,
        detect_particles_jax,
        detect_all_frames_jax,
        restore_image_jax
    )
except ImportError:
    HAS_JAX = False
    JAX_BACKEND = "unavailable"
    def check_jax():
        print("JAX module not available")
        return False

def check_gpu():
    """Check GPU availability and print status."""
    print("=== GPU Status ===")
    
    # Check CuPy (NVIDIA)
    if HAS_CUPY:
        import cupy as cp
        device = cp.cuda.Device()
        print(f"CuPy (NVIDIA): Available")
        print(f"  Device: {cp.cuda.runtime.getDeviceProperties(0)['name'].decode()}")
        mem = device.mem_info
        print(f"  Memory: {mem[1]/1e9:.1f} GB total, {mem[0]/1e9:.1f} GB free")
    else:
        print("CuPy (NVIDIA): Not available")
    
    # Check JAX
    if HAS_JAX:
        import jax
        print(f"JAX: Available (backend: {JAX_BACKEND})")
        for d in jax.devices():
            print(f"  Device: {d}")
    else:
        print("JAX: Not available")
    
    print()
    print("Installation hints:")
    print("  NVIDIA GPU: pip install cupy-cuda12x")
    print("  Mac (Apple Silicon): pip install jax jax-metal")
    print("  CPU only: pip install jax")
    
    return HAS_CUPY or HAS_JAX

# Linking (optimized with scipy.optimize.linear_sum_assignment + KD-tree)
from .linking_fast import (
    Trajectory,
    link_trajectories_fast as link_trajectories,
    link_frame_pair_fast as link_particles_pair,
    compute_cost
)

# Analysis
from .analysis import (
    MSDResult,
    MSSResult,
    VACResult,
    compute_msd,
    compute_mss,
    compute_velocity,
    compute_confinement_ratio,
    compute_velocity_autocorrelation,
    compute_directional_persistence,
    analyze_trajectory,
    analyze_all_trajectories
)

# I/O
from .io import (
    load_tiff,
    save_tiff,
    load_csv_trajectories,
    save_csv_trajectories,
    save_analysis_results,
    create_synthetic_movie
)

# Visualization
from .visualization import (
    plot_frame_with_particles,
    plot_trajectories,
    plot_trajectory,
    plot_msd,
    plot_mss,
    plot_diffusion_histogram,
    plot_motion_type_pie,
    create_tracking_movie,
    plot_detection_comparison,
    plot_vac,
    plot_vac_population,
    plot_persistence_histogram,
    plot_vac_decay_times
)

__all__ = [
    # Main
    'ParticleTracker',
    'TrackerParameters',
    
    # Data structures
    'Particle',
    'Trajectory',
    'MSDResult', 
    'MSSResult',
    
    # Detection
    'detect_particles',
    'detect_all_frames',
    'restore_image',
    'find_local_maxima',
    'refine_position',
    'discriminate_particles',
    'preview_detection',
    'HAS_CUPY',
    'check_gpu',
    
    # Linking
    'link_trajectories',
    'link_particles_pair',
    'compute_cost',
    
    # Analysis
    'compute_msd',
    'compute_mss',
    'compute_velocity',
    'compute_confinement_ratio',
    'analyze_trajectory',
    'analyze_all_trajectories',
    
    # I/O
    'load_tiff',
    'save_tiff',
    'load_csv_trajectories',
    'save_csv_trajectories',
    'save_analysis_results',
    'create_synthetic_movie',
    
    # Visualization
    'plot_frame_with_particles',
    'plot_trajectories',
    'plot_trajectory',
    'plot_msd',
    'plot_mss',
    'plot_diffusion_histogram',
    'plot_motion_type_pie',
    'create_tracking_movie',
    'plot_detection_comparison',
]

# Print GPU status on import (suppress if running non-interactively)
import sys
if sys.flags.interactive or hasattr(sys, 'ps1'):
    if HAS_CUPY:
        print(f"mosaic_tracker v{__version__}: GPU acceleration available (CuPy)")
    else:
        print(f"mosaic_tracker v{__version__}: CPU mode")
