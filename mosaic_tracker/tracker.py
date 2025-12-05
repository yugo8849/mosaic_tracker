"""
Main Particle Tracker Class

High-level interface for particle tracking and analysis.
"""

import numpy as np
from typing import List, Optional, Union, Tuple
from dataclasses import dataclass, field
from pathlib import Path

from .detection import detect_all_frames, Particle
from .linking import link_trajectories, Trajectory
from .analysis import (
    analyze_trajectory, 
    analyze_all_trajectories,
    compute_msd,
    compute_mss,
    MSDResult,
    MSSResult
)


@dataclass
class TrackerParameters:
    """
    Parameters for particle tracking.
    
    Detection Parameters (following MosaicSuite):
    - radius: Approximate radius of particles in pixels. Should be slightly
              larger than visible particle radius but smaller than inter-particle separation.
    - percentile: Percentile threshold (0-100). All local maxima in the upper r-th
                  percentile are candidates. Higher = more particles detected.
                  Default 0.1 = top 0.1% (very stringent).
    - absolute_threshold: If set, use absolute intensity instead of percentile.
    - cutoff_score: Score cutoff for non-particle discrimination. 
                    Higher = more strict. Set to 0 to disable.
    
    Linking Parameters:
    - max_displacement: Maximum displacement per frame (pixels)
    - link_range: Number of future frames to consider for gap closing
    - min_length: Minimum trajectory length to keep
    
    Physical Parameters:
    - pixel_size: Physical size per pixel (e.g., µm)
    - frame_interval: Time between frames (e.g., seconds)
    """
    # Detection parameters
    radius: int = 3
    percentile: float = 0.1  # Default: top 0.1% (stringent)
    absolute_threshold: Optional[float] = None  # Override percentile with absolute value
    cutoff_score: float = 0.0  # Non-particle discrimination (0 = disabled)
    lambda_n: float = 1.0  # Noise correlation length
    
    # Linking parameters
    max_displacement: float = 5.0
    link_range: int = 2
    min_length: int = 5
    
    # Physical parameters
    pixel_size: float = 1.0
    frame_interval: float = 1.0
    
    # GPU acceleration
    use_gpu: bool = False


class ParticleTracker:
    """
    Main particle tracking class.
    
    Implements the algorithm from Sbalzarini & Koumoutsakos (2005).
    
    Parameters
    ----------
    params : TrackerParameters, optional
        Tracking parameters. If None, defaults are used.
        
    Examples
    --------
    >>> tracker = ParticleTracker()
    >>> tracker.load_movie("movie.tif")  # or pass numpy array
    >>> tracker.detect_particles()
    >>> tracker.link_trajectories()
    >>> results = tracker.analyze()
    """
    
    def __init__(self, params: Optional[TrackerParameters] = None):
        self.params = params or TrackerParameters()
        self._movie: Optional[np.ndarray] = None
        self._particles: Optional[List[List[Particle]]] = None
        self._trajectories: Optional[List[Trajectory]] = None
        self._analysis_results: Optional[List[dict]] = None
    
    @property
    def movie(self) -> Optional[np.ndarray]:
        """The loaded movie."""
        return self._movie
    
    @property
    def particles(self) -> Optional[List[List[Particle]]]:
        """Detected particles per frame."""
        return self._particles
    
    @property
    def trajectories(self) -> Optional[List[Trajectory]]:
        """Linked trajectories."""
        return self._trajectories
    
    @property
    def n_frames(self) -> int:
        """Number of frames in movie."""
        return self._movie.shape[0] if self._movie is not None else 0
    
    @property
    def n_particles(self) -> int:
        """Total number of detected particles."""
        if self._particles is None:
            return 0
        return sum(len(p) for p in self._particles)
    
    @property
    def n_trajectories(self) -> int:
        """Number of trajectories."""
        return len(self._trajectories) if self._trajectories else 0
    
    def load_movie(self, movie: Union[np.ndarray, str, Path]) -> 'ParticleTracker':
        """
        Load movie data.
        
        Parameters
        ----------
        movie : ndarray or str or Path
            3D array (frames, height, width) or path to TIFF file
            
        Returns
        -------
        self : ParticleTracker
            For method chaining
        """
        if isinstance(movie, (str, Path)):
            from .io import load_tiff
            self._movie = load_tiff(movie)
        else:
            self._movie = np.asarray(movie)
        
        # Reset derived data
        self._particles = None
        self._trajectories = None
        self._analysis_results = None
        
        return self
    
    def detect_particles(self, 
                        radius: Optional[int] = None,
                        percentile: Optional[float] = None,
                        absolute_threshold: Optional[float] = None,
                        cutoff_score: Optional[float] = None,
                        use_gpu: Optional[bool] = None,
                        verbose: bool = False) -> 'ParticleTracker':
        """
        Detect particles in all frames.
        
        Parameters
        ----------
        radius : int, optional
            Particle radius (override)
        percentile : float, optional
            Intensity percentile threshold (override)
        absolute_threshold : float, optional
            Absolute intensity threshold (override)
        cutoff_score : float, optional
            Non-particle discrimination cutoff (override)
        use_gpu : bool, optional
            Use GPU acceleration
        verbose : bool
            Print progress
            
        Returns
        -------
        self : ParticleTracker
            For method chaining
        """
        if self._movie is None:
            raise ValueError("No movie loaded. Call load_movie() first.")
        
        r = radius if radius is not None else self.params.radius
        p = percentile if percentile is not None else self.params.percentile
        at = absolute_threshold if absolute_threshold is not None else self.params.absolute_threshold
        c = cutoff_score if cutoff_score is not None else self.params.cutoff_score
        gpu = use_gpu if use_gpu is not None else self.params.use_gpu
        
        self._particles = detect_all_frames(
            self._movie,
            radius=r,
            percentile=p,
            absolute_threshold=at,
            cutoff_score=c,
            lambda_n=self.params.lambda_n,
            use_gpu=gpu,
            verbose=verbose
        )
        
        # Reset downstream data
        self._trajectories = None
        self._analysis_results = None
        
        return self
    
    def link_trajectories(self,
                         max_displacement: Optional[float] = None,
                         link_range: Optional[int] = None,
                         min_length: Optional[int] = None) -> 'ParticleTracker':
        """
        Link detected particles into trajectories.
        
        Parameters
        ----------
        max_displacement : float, optional
            Override parameter
        link_range : int, optional
            Override parameter
        min_length : int, optional
            Override parameter
            
        Returns
        -------
        self : ParticleTracker
            For method chaining
        """
        if self._particles is None:
            raise ValueError("No particles detected. Call detect_particles() first.")
        
        d = max_displacement if max_displacement is not None else self.params.max_displacement
        r = link_range if link_range is not None else self.params.link_range
        m = min_length if min_length is not None else self.params.min_length
        
        self._trajectories = link_trajectories(
            self._particles,
            max_displacement=d,
            link_range=r,
            min_length=m
        )
        
        # Reset analysis
        self._analysis_results = None
        
        return self
    
    def analyze(self, min_length: Optional[int] = None) -> List[dict]:
        """
        Analyze all trajectories.
        
        Parameters
        ----------
        min_length : int, optional
            Minimum trajectory length for analysis
            
        Returns
        -------
        results : list of dict
            Analysis results for each trajectory
        """
        if self._trajectories is None:
            raise ValueError("No trajectories. Call link_trajectories() first.")
        
        m = min_length if min_length is not None else self.params.min_length
        
        self._analysis_results = analyze_all_trajectories(
            self._trajectories,
            pixel_size=self.params.pixel_size,
            frame_interval=self.params.frame_interval,
            min_length=m
        )
        
        return self._analysis_results
    
    def analyze_trajectory(self, trajectory_id: int) -> dict:
        """
        Analyze a single trajectory.
        
        Parameters
        ----------
        trajectory_id : int
            Trajectory ID
            
        Returns
        -------
        result : dict
            Analysis result
        """
        if self._trajectories is None:
            raise ValueError("No trajectories. Call link_trajectories() first.")
        
        for traj in self._trajectories:
            if traj.id == trajectory_id:
                return analyze_trajectory(
                    traj,
                    pixel_size=self.params.pixel_size,
                    frame_interval=self.params.frame_interval
                )
        
        raise ValueError(f"Trajectory {trajectory_id} not found")
    
    def get_msd(self, trajectory_id: int) -> MSDResult:
        """
        Get MSD for a trajectory.
        
        Parameters
        ----------
        trajectory_id : int
            Trajectory ID
            
        Returns
        -------
        msd : MSDResult
            MSD analysis result
        """
        for traj in self._trajectories:
            if traj.id == trajectory_id:
                return compute_msd(
                    traj,
                    pixel_size=self.params.pixel_size,
                    frame_interval=self.params.frame_interval
                )
        raise ValueError(f"Trajectory {trajectory_id} not found")
    
    def get_mss(self, trajectory_id: int) -> MSSResult:
        """
        Get MSS for a trajectory.
        
        Parameters
        ----------
        trajectory_id : int
            Trajectory ID
            
        Returns
        -------
        mss : MSSResult
            MSS analysis result
        """
        for traj in self._trajectories:
            if traj.id == trajectory_id:
                return compute_mss(
                    traj,
                    pixel_size=self.params.pixel_size,
                    frame_interval=self.params.frame_interval
                )
        raise ValueError(f"Trajectory {trajectory_id} not found")
    
    def get_trajectory(self, trajectory_id: int) -> Trajectory:
        """
        Get a trajectory by ID.
        
        Parameters
        ----------
        trajectory_id : int
            Trajectory ID
            
        Returns
        -------
        trajectory : Trajectory
            The trajectory
        """
        if self._trajectories is None:
            raise ValueError("No trajectories")
        
        for traj in self._trajectories:
            if traj.id == trajectory_id:
                return traj
        raise ValueError(f"Trajectory {trajectory_id} not found")
    
    def get_positions_dataframe(self):
        """
        Export all particle positions as pandas DataFrame.
        
        Returns
        -------
        df : pandas.DataFrame
            DataFrame with columns: trajectory_id, frame, x, y, m0, m2
        """
        import pandas as pd
        
        if self._trajectories is None:
            raise ValueError("No trajectories")
        
        data = []
        for traj in self._trajectories:
            for p in traj.particles:
                data.append({
                    'trajectory_id': traj.id,
                    'frame': p.frame,
                    'x': p.x * self.params.pixel_size,
                    'y': p.y * self.params.pixel_size,
                    'x_pixel': p.x,
                    'y_pixel': p.y,
                    'm0': p.m0,
                    'm2': p.m2
                })
        
        return pd.DataFrame(data)
    
    def get_summary_dataframe(self):
        """
        Export trajectory summary as pandas DataFrame.
        
        Returns
        -------
        df : pandas.DataFrame
            Summary DataFrame
        """
        import pandas as pd
        
        if self._analysis_results is None:
            self.analyze()
        
        data = []
        for result in self._analysis_results:
            data.append({
                'trajectory_id': result['trajectory_id'],
                'length': result['length'],
                'duration': result['duration'],
                'start_frame': result['start_frame'],
                'end_frame': result['end_frame'],
                'diffusion_coefficient': result['diffusion_coefficient'],
                'alpha': result['alpha'],
                'mss_slope': result['mss_slope'],
                'motion_type': result['motion_type'],
                'mean_speed': result['mean_speed'],
                'confinement_ratio': result['confinement_ratio']
            })
        
        return pd.DataFrame(data)
    
    def run(self) -> 'ParticleTracker':
        """
        Run complete tracking pipeline.
        
        Returns
        -------
        self : ParticleTracker
            For method chaining
        """
        if self._movie is None:
            raise ValueError("No movie loaded")
        
        self.detect_particles()
        self.link_trajectories()
        self.analyze()
        
        return self
    
    def summary(self) -> str:
        """
        Get summary string.
        
        Returns
        -------
        summary : str
            Summary of tracking results
        """
        lines = ["Particle Tracking Summary", "=" * 40]
        
        if self._movie is not None:
            lines.append(f"Movie: {self._movie.shape[0]} frames, "
                        f"{self._movie.shape[1]}x{self._movie.shape[2]} pixels")
        
        if self._particles is not None:
            lines.append(f"Detected particles: {self.n_particles}")
            particles_per_frame = [len(p) for p in self._particles]
            lines.append(f"  Mean per frame: {np.mean(particles_per_frame):.1f}")
        
        if self._trajectories is not None:
            lines.append(f"Trajectories: {self.n_trajectories}")
            lengths = [t.length for t in self._trajectories]
            lines.append(f"  Mean length: {np.mean(lengths):.1f} frames")
        
        if self._analysis_results:
            d_coeffs = [r['diffusion_coefficient'] for r in self._analysis_results]
            lines.append(f"Mean diffusion coefficient: {np.mean(d_coeffs):.4f}")
            
            motion_types = {}
            for r in self._analysis_results:
                mt = r['motion_type']
                motion_types[mt] = motion_types.get(mt, 0) + 1
            lines.append("Motion types:")
            for mt, count in sorted(motion_types.items()):
                lines.append(f"  {mt}: {count}")
        
        return "\n".join(lines)
    
    def __repr__(self):
        return (f"ParticleTracker(n_frames={self.n_frames}, "
                f"n_particles={self.n_particles}, "
                f"n_trajectories={self.n_trajectories})")
    
    def preview_detection(self, 
                         frame_idx: int = 0,
                         radius: Optional[int] = None,
                         percentile: Optional[float] = None,
                         absolute_threshold: Optional[float] = None,
                         cutoff_score: Optional[float] = None) -> Tuple[np.ndarray, List]:
        """
        Preview detection on a single frame for parameter tuning.
        
        Parameters
        ----------
        frame_idx : int
            Frame to preview
        radius, percentile, absolute_threshold, cutoff_score : optional
            Override parameters
            
        Returns
        -------
        filtered : ndarray
            Preprocessed image
        particles : list of Particle
            Detected particles
        """
        from .detection import preview_detection
        
        if self._movie is None:
            raise ValueError("No movie loaded")
        
        r = radius if radius is not None else self.params.radius
        p = percentile if percentile is not None else self.params.percentile
        at = absolute_threshold if absolute_threshold is not None else self.params.absolute_threshold
        c = cutoff_score if cutoff_score is not None else self.params.cutoff_score
        
        return preview_detection(
            self._movie[frame_idx],
            radius=r,
            percentile=p,
            absolute_threshold=at,
            cutoff_score=c
        )
