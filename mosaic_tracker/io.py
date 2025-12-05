"""
I/O Utilities Module

File loading and saving functions for particle tracking.
"""

import numpy as np
from pathlib import Path
from typing import Union, List, Optional, Tuple
import warnings


def load_tiff(path: Union[str, Path]) -> np.ndarray:
    """
    Load TIFF file(s) as numpy array.
    
    Parameters
    ----------
    path : str or Path
        Path to TIFF file or directory containing TIFF files
        
    Returns
    -------
    movie : ndarray
        3D array (frames, height, width)
    """
    path = Path(path)
    
    try:
        import tifffile
        
        if path.is_file():
            return tifffile.imread(str(path))
        elif path.is_dir():
            files = sorted(path.glob("*.tif")) + sorted(path.glob("*.tiff"))
            if not files:
                raise ValueError(f"No TIFF files found in {path}")
            return np.stack([tifffile.imread(str(f)) for f in files])
        else:
            raise ValueError(f"Path does not exist: {path}")
            
    except ImportError:
        # Fallback to PIL/Pillow
        from PIL import Image
        
        if path.is_file():
            img = Image.open(str(path))
            frames = []
            try:
                while True:
                    frames.append(np.array(img))
                    img.seek(img.tell() + 1)
            except EOFError:
                pass
            return np.stack(frames)
        elif path.is_dir():
            files = sorted(path.glob("*.tif")) + sorted(path.glob("*.tiff"))
            if not files:
                raise ValueError(f"No TIFF files found in {path}")
            return np.stack([np.array(Image.open(str(f))) for f in files])
        else:
            raise ValueError(f"Path does not exist: {path}")


def save_tiff(movie: np.ndarray, path: Union[str, Path]) -> None:
    """
    Save movie as TIFF file.
    
    Parameters
    ----------
    movie : ndarray
        3D array (frames, height, width)
    path : str or Path
        Output path
    """
    try:
        import tifffile
        tifffile.imwrite(str(path), movie)
    except ImportError:
        from PIL import Image
        frames = [Image.fromarray(movie[i]) for i in range(len(movie))]
        frames[0].save(
            str(path), 
            save_all=True, 
            append_images=frames[1:]
        )


def load_csv_trajectories(path: Union[str, Path], 
                         has_header: bool = True) -> 'List[Trajectory]':
    """
    Load trajectories from CSV file.
    
    Expected columns: trajectory_id, frame, x, y [, m0, m2]
    
    Parameters
    ----------
    path : str or Path
        Path to CSV file
    has_header : bool
        Whether file has header row
        
    Returns
    -------
    trajectories : list of Trajectory
    """
    import pandas as pd
    from .linking import Trajectory
    from .detection import Particle
    
    df = pd.read_csv(path, header=0 if has_header else None)
    
    if not has_header:
        df.columns = ['trajectory_id', 'frame', 'x', 'y'][:len(df.columns)]
    
    trajectories = []
    for traj_id in df['trajectory_id'].unique():
        traj_df = df[df['trajectory_id'] == traj_id].sort_values('frame')
        
        particles = []
        for _, row in traj_df.iterrows():
            p = Particle(
                x=row['x'],
                y=row['y'],
                m0=row.get('m0', 1.0),
                m2=row.get('m2', 1.0),
                frame=int(row['frame'])
            )
            particles.append(p)
        
        trajectories.append(Trajectory(id=int(traj_id), particles=particles))
    
    return trajectories


def save_csv_trajectories(trajectories: 'List[Trajectory]',
                         path: Union[str, Path],
                         pixel_size: float = 1.0) -> None:
    """
    Save trajectories to CSV file.
    
    Parameters
    ----------
    trajectories : list of Trajectory
        Trajectories to save
    path : str or Path
        Output path
    pixel_size : float
        Physical pixel size for coordinate scaling
    """
    import pandas as pd
    
    data = []
    for traj in trajectories:
        for p in traj.particles:
            data.append({
                'trajectory_id': traj.id,
                'frame': p.frame,
                'x': p.x * pixel_size,
                'y': p.y * pixel_size,
                'x_pixel': p.x,
                'y_pixel': p.y,
                'm0': p.m0,
                'm2': p.m2
            })
    
    df = pd.DataFrame(data)
    df.to_csv(path, index=False)


def save_analysis_results(results: List[dict],
                         path: Union[str, Path]) -> None:
    """
    Save analysis results to CSV file.
    
    Parameters
    ----------
    results : list of dict
        Analysis results from analyze_all_trajectories
    path : str or Path
        Output path
    """
    import pandas as pd
    
    data = []
    for r in results:
        data.append({
            'trajectory_id': r['trajectory_id'],
            'length': r['length'],
            'duration': r['duration'],
            'start_frame': r['start_frame'],
            'end_frame': r['end_frame'],
            'diffusion_coefficient': r['diffusion_coefficient'],
            'alpha': r['alpha'],
            'mss_slope': r['mss_slope'],
            'motion_type': r['motion_type'],
            'mean_speed': r['mean_speed'],
            'confinement_ratio': r['confinement_ratio']
        })
    
    pd.DataFrame(data).to_csv(path, index=False)


def create_synthetic_movie(n_frames: int = 100,
                          size: Tuple[int, int] = (256, 256),
                          n_particles: int = 10,
                          diffusion_coeff: float = 0.5,
                          snr: float = 5.0,
                          particle_sigma: float = 2.0,
                          background: float = 100.0,
                          seed: Optional[int] = None) -> np.ndarray:
    """
    Create synthetic movie with diffusing particles.
    
    Parameters
    ----------
    n_frames : int
        Number of frames
    size : tuple
        (height, width) in pixels
    n_particles : int
        Number of particles
    diffusion_coeff : float
        Diffusion coefficient in pixels^2/frame
    snr : float
        Signal-to-noise ratio
    particle_sigma : float
        Particle Gaussian width
    background : float
        Background intensity
    seed : int, optional
        Random seed
        
    Returns
    -------
    movie : ndarray
        Synthetic movie
    """
    from typing import Tuple
    
    if seed is not None:
        np.random.seed(seed)
    
    height, width = size
    movie = np.full((n_frames, height, width), background, dtype=float)
    
    # Initialize particle positions
    positions = np.random.rand(n_particles, 2) * [width - 20, height - 20] + 10
    
    # Compute peak intensity from SNR
    # SNR = (peak - background) / sqrt(peak)
    peak = (snr + np.sqrt(snr**2 + 4 * background)) / 2 * snr / 2 + background
    
    # Generate particle trajectories
    for t in range(n_frames):
        for i in range(n_particles):
            # Add particle to frame
            x, y = positions[i]
            
            # Create Gaussian blob
            xx, yy = np.meshgrid(
                np.arange(max(0, int(x) - 5), min(width, int(x) + 6)),
                np.arange(max(0, int(y) - 5), min(height, int(y) + 6))
            )
            
            blob = (peak - background) * np.exp(
                -((xx - x)**2 + (yy - y)**2) / (2 * particle_sigma**2)
            )
            
            movie[t, 
                  max(0, int(y) - 5):min(height, int(y) + 6),
                  max(0, int(x) - 5):min(width, int(x) + 6)] += blob
            
            # Random walk for next frame
            if t < n_frames - 1:
                step = np.random.randn(2) * np.sqrt(2 * diffusion_coeff)
                positions[i] += step
                
                # Reflective boundaries
                positions[i] = np.clip(positions[i], 2, [width - 2, height - 2])
    
    # Add Poisson noise
    movie = np.random.poisson(movie.astype(int)).astype(float)
    
    return movie
