# Mosaic Tracker

Python implementation of the MosaicSuite Particle Tracker for single particle tracking (SPT) and trajectory analysis.

## Overview

This package implements the particle tracking algorithm described in:

> Sbalzarini & Koumoutsakos (2005) "Feature point tracking and trajectory analysis for video imaging in cell biology" *Journal of Structural Biology* 151(2):182-195

The algorithm was ranked as one of the top choices in the objective comparison study:

> Chenouard et al. (2014) "Objective comparison of particle tracking methods" *Nature Methods* 11(3):281-289

## Features

- **Particle Detection**
  - Background subtraction and noise filtering
  - Local maxima detection with sub-pixel refinement
  - Non-particle discrimination using intensity moment clustering

- **Trajectory Linking**
  - Graph-theoretic optimization for optimal particle linking
  - Gap closing for temporary occlusion
  - Handles particle appearance/disappearance

- **Trajectory Analysis**
  - Mean Square Displacement (MSD) calculation
  - Moment Scaling Spectrum (MSS) analysis
  - Diffusion coefficient estimation
  - Motion type classification (confined, diffusive, directed)

## Installation

```bash
# Basic installation
pip install mosaic-tracker

# With visualization support
pip install mosaic-tracker[viz]

# With all dependencies
pip install mosaic-tracker[all]

# From source
git clone https://github.com/your-lab/mosaic-tracker.git
cd mosaic-tracker
pip install -e .[all]
```

## Quick Start

```python
from mosaic_tracker import ParticleTracker, TrackerParameters

# Configure parameters
params = TrackerParameters(
    radius=3,              # Particle radius (pixels)
    percentile=0.5,        # Intensity percentile threshold
    max_displacement=5.0,  # Max displacement per frame (pixels)
    link_range=2,          # Frames to look ahead for linking
    min_length=10,         # Minimum trajectory length
    pixel_size=0.080,      # Physical pixel size (µm)
    frame_interval=0.050   # Frame interval (seconds)
)

# Create tracker
tracker = ParticleTracker(params)

# Load movie and run tracking
tracker.load_movie("your_movie.tif")
tracker.detect_particles()
tracker.link_trajectories()
results = tracker.analyze()

# Print summary
print(tracker.summary())

# Export results
df = tracker.get_summary_dataframe()
df.to_csv("results.csv")
```

## Detailed Usage

### Parameter Selection

| Parameter | Description | Typical Range |
|-----------|-------------|---------------|
| `radius` | Particle radius in pixels | 2-5 for GEMs |
| `percentile` | Intensity threshold (0-100) | 0.1-2.0 |
| `cutoff_score` | Non-particle discrimination | 0-5 |
| `max_displacement` | Max movement per frame | 5-20 pixels |
| `link_range` | Gap closing range | 1-5 frames |

### Analysis Output

The analysis returns a dictionary for each trajectory containing:

- `diffusion_coefficient`: Estimated D (µm²/s)
- `alpha`: MSD slope (anomalous diffusion exponent)
- `mss_slope`: MSS slope (motion type indicator)
- `motion_type`: Classification (confined, subdiffusive, normal, superdiffusive, ballistic)
- `mean_speed`: Average velocity
- `confinement_ratio`: End-to-end distance / path length

### Visualization

```python
from mosaic_tracker import (
    plot_trajectories,
    plot_msd,
    plot_mss,
    plot_diffusion_histogram
)

# Plot all trajectories
fig, ax = plt.subplots()
plot_trajectories(tracker.trajectories, ax=ax)

# Plot MSD for specific trajectory
msd = tracker.get_msd(trajectory_id=0)
plot_msd(msd)

# Plot diffusion coefficient distribution
plot_diffusion_histogram(results)
```

## For 40nm GEM Tracking

Recommended parameters for tracking 40nm GEM particles:

```python
params = TrackerParameters(
    radius=2,              # GEMs appear as ~2-3 pixel spots
    percentile=0.1,        # Lower = more stringent detection
    cutoff_score=0.0,      # Disable discrimination for homogeneous particles
    max_displacement=10,   # Adjust based on expected diffusion
    link_range=2,          # Allow 1-frame gaps
    min_length=20,         # Require longer tracks for reliable MSD
    pixel_size=0.080,      # Typical for 100x objective
    frame_interval=0.033   # 30 fps
)
```

## Citation

If you use this software, please cite:

```bibtex
@article{sbalzarini2005feature,
  title={Feature point tracking and trajectory analysis for video imaging in cell biology},
  author={Sbalzarini, Ivo F and Koumoutsakos, Petros},
  journal={Journal of structural biology},
  volume={151},
  number={2},
  pages={182--195},
  year={2005},
  publisher={Elsevier}
}
```

## License

MIT License
