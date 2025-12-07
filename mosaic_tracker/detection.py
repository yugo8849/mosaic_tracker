"""
Feature Point Detection Module

Implementation of the particle detection algorithm from:
Sbalzarini & Koumoutsakos (2005) "Feature point tracking and trajectory 
analysis for video imaging in cell biology" J. Struct. Biol. 151(2):182-195

The detection consists of four steps:
1. Image restoration (background subtraction + noise filtering)
2. Estimation of point locations (local maxima)
3. Refinement of point locations (centroid-based)
4. Non-particle discrimination (intensity moment clustering)
"""

import numpy as np
from scipy import ndimage
from dataclasses import dataclass
from typing import List, Tuple, Optional

# Try to import CuPy for GPU acceleration
HAS_CUPY = False
try:
    import cupy as cp
    from cupyx.scipy import ndimage as cp_ndimage
    # Test if CUDA is actually usable
    cp.cuda.runtime.getDeviceCount()
    HAS_CUPY = True
except ImportError:
    pass
except Exception as e:
    # CUDA driver issues (version mismatch, no GPU, etc.)
    print(f"Warning: CuPy available but CUDA not usable: {e}")
    pass


@dataclass
class Particle:
    """Detected particle with position and intensity moments."""
    x: float  # refined x position (sub-pixel)
    y: float  # refined y position (sub-pixel)
    m0: float  # 0th order intensity moment (total intensity)
    m2: float  # 2nd order intensity moment (spatial extent)
    frame: int  # frame index
    
    def __repr__(self):
        return f"Particle(x={self.x:.2f}, y={self.y:.2f}, m0={self.m0:.1f}, frame={self.frame})"


def create_restoration_kernel(w: int, lambda_n: float = 1.0) -> np.ndarray:
    """
    Create the image restoration kernel combining background subtraction
    and Gaussian noise filtering (Eq. 4 in Sbalzarini 2005).
    
    Parameters
    ----------
    w : int
        Particle radius (pixels). Should be larger than particle apparent radius
        but smaller than inter-particle distance.
    lambda_n : float
        Noise correlation length (typically 1 pixel for CCD cameras)
        
    Returns
    -------
    kernel : ndarray
        2D convolution kernel of size (2w+1, 2w+1)
    """
    size = 2 * w + 1
    i_range = np.arange(-w, w + 1)
    
    # Compute normalization B for Gaussian (Eq. 3)
    B = np.sum(np.exp(-i_range**2 / (4 * lambda_n**2)))**2
    
    # Create kernel (Eq. 4)
    ii, jj = np.meshgrid(i_range, i_range, indexing='ij')
    gaussian_term = (1.0 / B) * np.exp(-(ii**2 + jj**2) / (4 * lambda_n**2))
    boxcar_term = 1.0 / (size**2)
    kernel = gaussian_term - boxcar_term
    
    # Normalize (Eq. 5)
    K0 = (1.0 / B) * np.sum(np.exp(-i_range**2 / (2 * lambda_n**2)))**2 - B / (size**2)
    if abs(K0) > 1e-10:
        kernel = kernel / K0
    
    return kernel


def restore_image(image: np.ndarray, w: int, lambda_n: float = 1.0, 
                  use_gpu: bool = False) -> np.ndarray:
    """
    Restore image by removing background variations and filtering noise (Eq. 6).
    
    Parameters
    ----------
    image : ndarray
        Input image (2D array), should be normalized to [0, 1]
    w : int
        Particle radius
    lambda_n : float
        Noise correlation length
    use_gpu : bool
        Use GPU acceleration if available
        
    Returns
    -------
    filtered : ndarray
        Restored image
    """
    kernel = create_restoration_kernel(w, lambda_n)
    
    if use_gpu and HAS_CUPY:
        # GPU version
        image_gpu = cp.asarray(image.astype(np.float32))
        kernel_gpu = cp.asarray(kernel.astype(np.float32))
        padded = cp.pad(image_gpu, w, mode='reflect')
        filtered = cp_ndimage.convolve(padded, kernel_gpu, mode='constant')
        filtered = filtered[w:-w, w:-w]
        filtered = cp.maximum(filtered, 0)
        return cp.asnumpy(filtered)
    else:
        # CPU version
        padded = np.pad(image.astype(float), w, mode='reflect')
        filtered = ndimage.convolve(padded, kernel, mode='constant')
        filtered = filtered[w:-w, w:-w]
        filtered = np.maximum(filtered, 0)
        return filtered


def find_local_maxima(image: np.ndarray, 
                      w: int, 
                      percentile: float = 0.1,
                      absolute_threshold: Optional[float] = None,
                      use_gpu: bool = False) -> List[Tuple[int, int]]:
    """
    Find local maxima in the image that are above intensity threshold.
    
    Parameters
    ----------
    image : ndarray
        Filtered image
    w : int
        Particle radius (search radius for local maxima)
    percentile : float
        Percentile threshold (0-100). 
        All local maxima in the upper r-th percentile are candidates.
        Higher values = more particles detected.
        Default 0.1 means top 0.1% brightest.
    absolute_threshold : float, optional
        If provided, use absolute intensity threshold instead of percentile.
    use_gpu : bool
        Use GPU acceleration if available
        
    Returns
    -------
    maxima : list of (y, x) tuples
        Positions of local maxima
    """
    # Create structuring element for dilation
    size = 2 * w + 1
    footprint = np.ones((size, size), dtype=bool)
    
    if use_gpu and HAS_CUPY:
        # GPU version
        image_gpu = cp.asarray(image.astype(np.float32))
        footprint_gpu = cp.asarray(footprint)
        dilated = cp_ndimage.grey_dilation(image_gpu, footprint=footprint_gpu)
        is_maximum = (image_gpu == dilated) & (image_gpu > 0)
        
        # Apply threshold
        if absolute_threshold is not None:
            threshold = absolute_threshold
        else:
            # Percentile threshold: top r% of intensity values
            positive_vals = image_gpu[image_gpu > 0]
            if positive_vals.size > 0:
                threshold = cp.percentile(positive_vals, 100 - percentile)
            else:
                threshold = 0
        
        is_maximum = is_maximum & (image_gpu >= threshold)
        maxima_indices = cp.where(is_maximum)
        maxima = list(zip(cp.asnumpy(maxima_indices[0]), cp.asnumpy(maxima_indices[1])))
    else:
        # CPU version
        dilated = ndimage.grey_dilation(image, footprint=footprint)
        is_maximum = (image == dilated) & (image > 0)
        
        # Apply threshold
        if absolute_threshold is not None:
            threshold = absolute_threshold
        else:
            positive_vals = image[image > 0]
            if positive_vals.size > 0:
                threshold = np.percentile(positive_vals, 100 - percentile)
            else:
                threshold = 0
        
        is_maximum = is_maximum & (image >= threshold)
        maxima = list(zip(*np.where(is_maximum)))
    
    return maxima


def refine_position(image: np.ndarray, y: int, x: int, w: int) -> Tuple[float, float, float, float]:
    """
    Refine particle position using brightness-weighted centroid (Eq. 7).
    Also compute intensity moments m0 (Eq. 8) and m2 (Eq. 9).
    
    Parameters
    ----------
    image : ndarray
        Filtered image
    y, x : int
        Initial position estimate
    w : int
        Particle radius
        
    Returns
    -------
    y_refined, x_refined : float
        Refined position (sub-pixel)
    m0 : float
        0th order intensity moment (total intensity)
    m2 : float
        2nd order intensity moment
    """
    height, width = image.shape
    
    # Extract region around particle
    y_min = max(0, y - w)
    y_max = min(height, y + w + 1)
    x_min = max(0, x - w)
    x_max = min(width, x + w + 1)
    
    region = image[y_min:y_max, x_min:x_max]
    
    # Create coordinate grids relative to particle center
    yy, xx = np.meshgrid(
        np.arange(y_min - y, y_max - y),
        np.arange(x_min - x, x_max - x),
        indexing='ij'
    )
    
    # Compute mask for circular region (i^2 + j^2 <= w^2)
    r2 = xx**2 + yy**2
    mask = r2 <= w**2
    
    if not np.any(mask):
        return float(y), float(x), 0.0, 0.0
    
    # Compute m0 (Eq. 8)
    m0 = np.sum(region[mask])
    
    if m0 <= 0:
        return float(y), float(x), 0.0, 0.0
    
    # Compute centroid offset (Eq. 7)
    epsilon_y = np.sum(yy[mask] * region[mask]) / m0
    epsilon_x = np.sum(xx[mask] * region[mask]) / m0
    
    # Compute m2 (Eq. 9)
    m2 = np.sum(r2[mask] * region[mask]) / m0
    
    # Refine position
    y_refined = y + epsilon_y
    x_refined = x + epsilon_x
    
    return y_refined, x_refined, m0, m2


def discriminate_particles(particles: List[Particle], 
                          cutoff_score: float,
                          sigma_scale: float = 0.1) -> List[Particle]:
    """
    Discriminate true particles from spurious detections using
    intensity moment clustering in (m0, m2) plane (Eq. 10-11).
    
    Each particle carries a 2D Gaussian in (m0, m2) space. The score
    of each particle is the sum of contributions from all other particles.
    Particles with scores below cutoff are rejected.
    
    Parameters
    ----------
    particles : list of Particle
        Detected particles
    cutoff_score : float
        Minimum score for particle to be considered real.
        Higher values = more strict filtering.
        Set to 0 to skip discrimination.
    sigma_scale : float
        Scale factor for Gaussian standard deviations (relative to value range)
        
    Returns
    -------
    filtered : list of Particle
        Particles passing discrimination
    """
    if cutoff_score <= 0 or len(particles) < 2:
        return particles
    
    # Extract m0 and m2 values
    m0_values = np.array([p.m0 for p in particles])
    m2_values = np.array([p.m2 for p in particles])
    
    # Compute standard deviations based on value ranges
    m0_range = np.max(m0_values) - np.min(m0_values)
    m2_range = np.max(m2_values) - np.min(m2_values)
    
    sigma0 = sigma_scale * m0_range if m0_range > 0 else 1.0
    sigma2 = sigma_scale * m2_range if m2_range > 0 else 1.0
    
    n = len(particles)
    scores = np.zeros(n)
    
    # Compute score for each particle (Eq. 11)
    for i in range(n):
        for j in range(n):
            if i != j:
                d0 = (m0_values[i] - m0_values[j])**2 / (2 * sigma0**2)
                d2 = (m2_values[i] - m2_values[j])**2 / (2 * sigma2**2)
                scores[i] += np.exp(-d0 - d2) / (2 * np.pi * sigma0 * sigma2 * n)
    
    # Filter particles
    filtered = [p for p, s in zip(particles, scores) if s >= cutoff_score]
    
    return filtered


def detect_particles(image: np.ndarray,
                    frame_idx: int,
                    radius: int = 3,
                    percentile: float = 0.1,
                    absolute_threshold: Optional[float] = None,
                    cutoff_score: float = 0.0,
                    lambda_n: float = 1.0,
                    global_min: Optional[float] = None,
                    global_max: Optional[float] = None,
                    use_gpu: bool = False) -> List[Particle]:
    """
    Detect particles in a single frame using Sbalzarini algorithm.
    
    Parameters
    ----------
    image : ndarray
        Input image (2D array)
    frame_idx : int
        Frame index
    radius : int
        Particle radius in pixels. Should be slightly larger than visible
        particle radius but smaller than inter-particle separation.
    percentile : float
        Percentile threshold (0-100). All local maxima in the upper r-th
        percentile are candidates. Higher = more particles.
        Default 0.1 = top 0.1%.
    absolute_threshold : float, optional
        If provided, use absolute intensity threshold instead of percentile.
    cutoff_score : float
        Score cutoff for non-particle discrimination.
        Higher = more strict. Set to 0 to disable.
    lambda_n : float
        Noise correlation length (typically 1 pixel)
    global_min, global_max : float, optional
        Global intensity min/max for normalization across movie.
        If None, per-frame normalization is used.
    use_gpu : bool
        Use GPU acceleration if available
        
    Returns
    -------
    particles : list of Particle
        Detected particles with refined positions and moments
    """
    # Normalize image to [0, 1]
    if global_min is not None and global_max is not None:
        denom = global_max - global_min
        if denom > 0:
            normalized = (image.astype(float) - global_min) / denom
        else:
            normalized = image.astype(float)
    else:
        img_min, img_max = image.min(), image.max()
        denom = img_max - img_min
        if denom > 0:
            normalized = (image.astype(float) - img_min) / denom
        else:
            normalized = image.astype(float)
    
    # Restore image (background removal + denoising)
    filtered = restore_image(normalized, radius, lambda_n, use_gpu=use_gpu)
    
    # Find local maxima above threshold
    maxima = find_local_maxima(
        filtered, radius, 
        percentile=percentile,
        absolute_threshold=absolute_threshold,
        use_gpu=use_gpu
    )
    
    # Refine positions and compute moments
    particles = []
    for y, x in maxima:
        y_ref, x_ref, m0, m2 = refine_position(filtered, y, x, radius)
        if m0 > 0:
            particles.append(Particle(
                x=x_ref,
                y=y_ref,
                m0=m0,
                m2=m2,
                frame=frame_idx
            ))
    
    # Non-particle discrimination
    if cutoff_score > 0:
        particles = discriminate_particles(particles, cutoff_score)
    
    return particles


def detect_all_frames(movie: np.ndarray,
                     radius: int = 3,
                     percentile: float = 0.1,
                     absolute_threshold: Optional[float] = None,
                     cutoff_score: float = 0.0,
                     lambda_n: float = 1.0,
                     use_gpu: bool = False,
                     verbose: bool = False) -> List[List[Particle]]:
    """
    Detect particles in all frames of a movie.
    
    Parameters
    ----------
    movie : ndarray
        3D array of shape (n_frames, height, width)
    radius : int
        Particle radius
    percentile : float
        Intensity percentile threshold (higher = more permissive)
    absolute_threshold : float, optional
        Absolute intensity threshold (overrides percentile)
    cutoff_score : float
        Non-particle discrimination cutoff
    lambda_n : float
        Noise correlation length
    use_gpu : bool
        Use GPU acceleration if available
    verbose : bool
        Print progress
        
    Returns
    -------
    all_particles : list of list of Particle
        Particles detected in each frame
    """
    n_frames = movie.shape[0]
    
    # Compute global min/max for normalization
    global_min = float(movie.min())
    global_max = float(movie.max())
    
    if use_gpu and HAS_CUPY:
        if verbose:
            print(f"Using GPU acceleration (CuPy)")
    elif use_gpu and not HAS_CUPY:
        if verbose:
            print("CuPy not available, falling back to CPU")
        use_gpu = False
    
    all_particles = []
    for t in range(n_frames):
        if verbose and (t % 100 == 0 or t == n_frames - 1):
            print(f"Processing frame {t+1}/{n_frames}")
        
        particles = detect_particles(
            movie[t],
            frame_idx=t,
            radius=radius,
            percentile=percentile,
            absolute_threshold=absolute_threshold,
            cutoff_score=cutoff_score,
            lambda_n=lambda_n,
            global_min=global_min,
            global_max=global_max,
            use_gpu=use_gpu
        )
        all_particles.append(particles)
    
    if verbose:
        total = sum(len(p) for p in all_particles)
        print(f"Total particles detected: {total} ({total/n_frames:.1f}/frame)")
    
    return all_particles


def preview_detection(image: np.ndarray,
                     radius: int = 3,
                     percentile: float = 0.1,
                     absolute_threshold: Optional[float] = None,
                     cutoff_score: float = 0.0) -> Tuple[np.ndarray, List[Particle]]:
    """
    Preview detection on a single frame, returning both filtered image and particles.
    Useful for parameter tuning.
    
    Parameters
    ----------
    image : ndarray
        Input image
    radius : int
        Particle radius
    percentile : float
        Intensity percentile threshold
    absolute_threshold : float, optional
        Absolute intensity threshold
    cutoff_score : float
        Non-particle discrimination cutoff
        
    Returns
    -------
    filtered : ndarray
        Preprocessed/filtered image
    particles : list of Particle
        Detected particles
    """
    # Normalize
    img_min, img_max = image.min(), image.max()
    denom = img_max - img_min
    if denom > 0:
        normalized = (image.astype(float) - img_min) / denom
    else:
        normalized = image.astype(float)
    
    # Filter
    filtered = restore_image(normalized, radius, 1.0)
    
    # Detect
    particles = detect_particles(
        image, 
        frame_idx=0,
        radius=radius,
        percentile=percentile,
        absolute_threshold=absolute_threshold,
        cutoff_score=cutoff_score
    )
    
    return filtered, particles


# Keep these for backwards compatibility
def laplacian_of_gaussian(image: np.ndarray, sigma: float) -> np.ndarray:
    """Apply Laplacian of Gaussian filter (for comparison only)."""
    filtered = -ndimage.gaussian_laplace(image.astype(float), sigma=sigma)
    return np.maximum(filtered, 0)


def tophat_filter(image: np.ndarray, radius: int) -> np.ndarray:
    """Apply white top-hat transform (for comparison only)."""
    from scipy.ndimage import white_tophat
    size = 2 * radius + 1
    y, x = np.ogrid[-radius:radius+1, -radius:radius+1]
    structure = (x**2 + y**2 <= radius**2).astype(int)
    return white_tophat(image.astype(float), structure=structure)


def bandpass_filter(image: np.ndarray, low_sigma: float, high_sigma: float) -> np.ndarray:
    """Apply bandpass filter (for comparison only)."""
    low_pass = ndimage.gaussian_filter(image.astype(float), sigma=low_sigma)
    high_pass = ndimage.gaussian_filter(image.astype(float), sigma=high_sigma)
    return np.maximum(low_pass - high_pass, 0)
