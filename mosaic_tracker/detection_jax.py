"""
JAX-accelerated Detection Module

Supports Apple Silicon GPU via jax-metal.
Falls back to CPU if JAX is not available.

Installation:
  # Mac (Apple Silicon)
  pip install jax jax-metal
  
  # Linux/Windows (NVIDIA)
  pip install jax[cuda12]
  
  # CPU only
  pip install jax
"""

import numpy as np
from typing import List, Tuple, Optional
from dataclasses import dataclass

# Try to import JAX
HAS_JAX = False
JAX_BACKEND = "cpu"

try:
    import jax
    import jax.numpy as jnp
    from jax import jit, vmap
    from jax.scipy.signal import convolve2d
    HAS_JAX = True
    
    # Check backend
    devices = jax.devices()
    if devices:
        JAX_BACKEND = devices[0].platform  # 'cpu', 'gpu', 'metal', etc.
except ImportError:
    pass


@dataclass
class Particle:
    """Detected particle with position and intensity moments."""
    x: float
    y: float
    m0: float
    m2: float
    frame: int
    
    def __repr__(self):
        return f"Particle(x={self.x:.2f}, y={self.y:.2f}, m0={self.m0:.1f}, frame={self.frame})"


def check_jax():
    """Check JAX availability and backend."""
    if HAS_JAX:
        import jax
        devices = jax.devices()
        print(f"JAX available: {JAX_BACKEND}")
        for d in devices:
            print(f"  Device: {d}")
        return True
    else:
        print("JAX not available. Install with:")
        print("  Mac: pip install jax jax-metal")
        print("  NVIDIA: pip install jax[cuda12]")
        print("  CPU: pip install jax")
        return False


if HAS_JAX:
    from functools import partial
    
    def _create_restoration_kernel_jax(w: int, lambda_n: float = 1.0) -> jnp.ndarray:
        """Create restoration kernel using JAX (no JIT due to dynamic size)."""
        size = 2 * w + 1
        i_range = jnp.arange(-w, w + 1)
        
        # Compute normalization B
        B = jnp.sum(jnp.exp(-i_range**2 / (4 * lambda_n**2)))**2
        
        # Create kernel
        ii, jj = jnp.meshgrid(i_range, i_range, indexing='ij')
        gaussian_term = (1.0 / B) * jnp.exp(-(ii**2 + jj**2) / (4 * lambda_n**2))
        boxcar_term = 1.0 / (size**2)
        kernel = gaussian_term - boxcar_term
        
        # Normalize
        K0 = (1.0 / B) * jnp.sum(jnp.exp(-i_range**2 / (2 * lambda_n**2)))**2 - B / (size**2)
        kernel = jnp.where(jnp.abs(K0) > 1e-10, kernel / K0, kernel)
        
        return kernel
    
    @jit
    def _convolve2d_jax(image: jnp.ndarray, kernel: jnp.ndarray) -> jnp.ndarray:
        """Convolve image with kernel (JIT-compiled)."""
        return convolve2d(image, kernel, mode='same')
    
    def _restore_image_jax(image: jnp.ndarray, kernel: jnp.ndarray) -> jnp.ndarray:
        """Apply restoration filter using JAX convolution."""
        filtered = _convolve2d_jax(image, kernel)
        filtered = jnp.maximum(filtered, 0)
        return filtered
    
    def _find_local_maxima_jax(image: jnp.ndarray, w: int) -> np.ndarray:
        """
        Find local maxima using scipy.ndimage (faster for this operation).
        Returns a boolean mask of local maxima positions.
        """
        from scipy import ndimage as ndi
        
        # Convert to numpy for scipy
        image_np = np.asarray(image)
        
        # Max filter using scipy (this is actually very fast)
        size = 2 * w + 1
        footprint = np.ones((size, size), dtype=bool)
        dilated = ndi.grey_dilation(image_np, footprint=footprint)
        
        is_maximum = (image_np == dilated) & (image_np > 0)
        return is_maximum


def restore_image_jax(image: np.ndarray, w: int, lambda_n: float = 1.0) -> np.ndarray:
    """
    Restore image using JAX acceleration.
    
    Parameters
    ----------
    image : ndarray
        Input image (normalized to [0, 1])
    w : int
        Particle radius
    lambda_n : float
        Noise correlation length
        
    Returns
    -------
    filtered : ndarray
        Restored image
    """
    if not HAS_JAX:
        raise RuntimeError("JAX not available")
    
    kernel = _create_restoration_kernel_jax(w, lambda_n)
    image_jax = jnp.asarray(image.astype(np.float32))
    filtered = _restore_image_jax(image_jax, kernel)
    
    return np.asarray(filtered)


def detect_particles_jax(image: np.ndarray,
                         frame_idx: int,
                         radius: int = 3,
                         percentile: float = 0.1,
                         absolute_threshold: Optional[float] = None,
                         cutoff_score: float = 0.0,
                         lambda_n: float = 1.0,
                         global_min: Optional[float] = None,
                         global_max: Optional[float] = None) -> List[Particle]:
    """
    Detect particles using JAX acceleration.
    
    Falls back to NumPy if JAX is not available.
    """
    if not HAS_JAX:
        # Fallback to numpy version
        from .detection import detect_particles
        return detect_particles(
            image, frame_idx, radius, percentile, 
            absolute_threshold, cutoff_score, lambda_n,
            global_min, global_max, use_gpu=False
        )
    
    # Normalize
    if global_min is not None and global_max is not None:
        denom = global_max - global_min
        if denom > 0:
            normalized = (image.astype(np.float32) - global_min) / denom
        else:
            normalized = image.astype(np.float32)
    else:
        img_min, img_max = float(image.min()), float(image.max())
        denom = img_max - img_min
        if denom > 0:
            normalized = (image.astype(np.float32) - img_min) / denom
        else:
            normalized = image.astype(np.float32)
    
    # Create kernel and filter
    kernel = _create_restoration_kernel_jax(radius, lambda_n)
    image_jax = jnp.asarray(normalized)
    filtered_jax = _restore_image_jax(image_jax, kernel)
    filtered = np.asarray(filtered_jax)
    
    # Compute threshold
    if absolute_threshold is not None:
        threshold = absolute_threshold
    else:
        positive_vals = filtered[filtered > 0]
        if len(positive_vals) > 0:
            threshold = np.percentile(positive_vals, 100 - percentile)
        else:
            threshold = 0
    
    # Find local maxima (use scipy - faster than JAX for this)
    is_maximum = _find_local_maxima_jax(filtered_jax, radius)
    is_maximum = is_maximum & (filtered >= threshold)
    
    # Extract positions
    maxima = list(zip(*np.where(is_maximum)))
    
    # Refine positions (CPU, since this is sequential)
    particles = []
    for y, x in maxima:
        y_ref, x_ref, m0, m2 = _refine_position_numpy(filtered, y, x, radius)
        if m0 > 0:
            particles.append(Particle(
                x=x_ref,
                y=y_ref,
                m0=m0,
                m2=m2,
                frame=frame_idx
            ))
    
    # Non-particle discrimination
    if cutoff_score > 0 and len(particles) > 1:
        particles = _discriminate_numpy(particles, cutoff_score)
    
    return particles


def _refine_position_numpy(image: np.ndarray, y: int, x: int, w: int) -> Tuple[float, float, float, float]:
    """Refine position using brightness-weighted centroid (NumPy version)."""
    height, width = image.shape
    
    y_min = max(0, y - w)
    y_max = min(height, y + w + 1)
    x_min = max(0, x - w)
    x_max = min(width, x + w + 1)
    
    region = image[y_min:y_max, x_min:x_max]
    
    yy, xx = np.meshgrid(
        np.arange(y_min - y, y_max - y),
        np.arange(x_min - x, x_max - x),
        indexing='ij'
    )
    
    r2 = xx**2 + yy**2
    mask = r2 <= w**2
    
    if not np.any(mask):
        return float(y), float(x), 0.0, 0.0
    
    m0 = np.sum(region[mask])
    if m0 <= 0:
        return float(y), float(x), 0.0, 0.0
    
    epsilon_y = np.sum(yy[mask] * region[mask]) / m0
    epsilon_x = np.sum(xx[mask] * region[mask]) / m0
    m2 = np.sum(r2[mask] * region[mask]) / m0
    
    return y + epsilon_y, x + epsilon_x, m0, m2


def _discriminate_numpy(particles: List[Particle], cutoff_score: float, 
                        sigma_scale: float = 0.1) -> List[Particle]:
    """Non-particle discrimination (NumPy version)."""
    m0_values = np.array([p.m0 for p in particles])
    m2_values = np.array([p.m2 for p in particles])
    
    m0_range = np.max(m0_values) - np.min(m0_values)
    m2_range = np.max(m2_values) - np.min(m2_values)
    
    sigma0 = sigma_scale * m0_range if m0_range > 0 else 1.0
    sigma2 = sigma_scale * m2_range if m2_range > 0 else 1.0
    
    n = len(particles)
    scores = np.zeros(n)
    
    for i in range(n):
        for j in range(n):
            if i != j:
                d0 = (m0_values[i] - m0_values[j])**2 / (2 * sigma0**2)
                d2 = (m2_values[i] - m2_values[j])**2 / (2 * sigma2**2)
                scores[i] += np.exp(-d0 - d2) / (2 * np.pi * sigma0 * sigma2 * n)
    
    return [p for p, s in zip(particles, scores) if s >= cutoff_score]


def detect_all_frames_jax(movie: np.ndarray,
                          radius: int = 3,
                          percentile: float = 0.1,
                          absolute_threshold: Optional[float] = None,
                          cutoff_score: float = 0.0,
                          lambda_n: float = 1.0,
                          verbose: bool = False) -> List[List[Particle]]:
    """
    Detect particles in all frames using JAX acceleration.
    
    Parameters
    ----------
    movie : ndarray
        3D array (frames, height, width)
    radius : int
        Particle radius
    percentile : float
        Intensity percentile threshold
    absolute_threshold : float, optional
        Absolute intensity threshold
    cutoff_score : float
        Non-particle discrimination cutoff
    lambda_n : float
        Noise correlation length
    verbose : bool
        Print progress
        
    Returns
    -------
    all_particles : list of list of Particle
    """
    if not HAS_JAX:
        from .detection import detect_all_frames
        return detect_all_frames(
            movie, radius, percentile, absolute_threshold,
            cutoff_score, lambda_n, use_gpu=False, verbose=verbose
        )
    
    n_frames = movie.shape[0]
    global_min = float(movie.min())
    global_max = float(movie.max())
    
    if verbose:
        print(f"Using JAX backend: {JAX_BACKEND}")
    
    # Pre-compile kernel
    kernel = _create_restoration_kernel_jax(radius, lambda_n)
    
    all_particles = []
    for t in range(n_frames):
        if verbose and (t % 100 == 0 or t == n_frames - 1):
            print(f"Processing frame {t+1}/{n_frames}")
        
        particles = detect_particles_jax(
            movie[t],
            frame_idx=t,
            radius=radius,
            percentile=percentile,
            absolute_threshold=absolute_threshold,
            cutoff_score=cutoff_score,
            lambda_n=lambda_n,
            global_min=global_min,
            global_max=global_max
        )
        all_particles.append(particles)
    
    if verbose:
        total = sum(len(p) for p in all_particles)
        print(f"Total particles detected: {total} ({total/n_frames:.1f}/frame)")
    
    return all_particles


# Batch processing for better GPU utilization
if HAS_JAX:
    @jit
    def _batch_restore_images(images: jnp.ndarray, kernel: jnp.ndarray) -> jnp.ndarray:
        """Restore multiple images in parallel."""
        # vmap over the batch dimension
        restore_single = lambda img: _restore_image_jax(img, kernel)
        return vmap(restore_single)(images)
    
    def detect_batch_jax(movie: np.ndarray,
                         radius: int = 3,
                         lambda_n: float = 1.0,
                         batch_size: int = 32) -> np.ndarray:
        """
        Batch process frames for filtering (returns filtered movie).
        
        This is more efficient than frame-by-frame processing on GPU.
        """
        n_frames = movie.shape[0]
        
        # Normalize
        global_min = float(movie.min())
        global_max = float(movie.max())
        denom = global_max - global_min
        if denom > 0:
            normalized = (movie.astype(np.float32) - global_min) / denom
        else:
            normalized = movie.astype(np.float32)
        
        kernel = _create_restoration_kernel_jax(radius, lambda_n)
        
        # Process in batches
        filtered_frames = []
        for i in range(0, n_frames, batch_size):
            batch = jnp.asarray(normalized[i:i+batch_size])
            filtered_batch = _batch_restore_images(batch, kernel)
            filtered_frames.append(np.asarray(filtered_batch))
        
        return np.concatenate(filtered_frames, axis=0)
