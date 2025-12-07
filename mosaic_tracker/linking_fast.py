"""
Fast Trajectory Linking Module

Optimized implementation using:
- scipy.optimize.linear_sum_assignment (Hungarian algorithm)
- scipy.spatial.cKDTree for fast neighbor search
- NumPy vectorization for cost computation

Based on Sbalzarini & Koumoutsakos (2005)
"""

import numpy as np
from scipy.optimize import linear_sum_assignment
from scipy.spatial import cKDTree
from dataclasses import dataclass, field
from typing import List, Dict, Optional, Tuple
from .detection import Particle


@dataclass
class Trajectory:
    """A particle trajectory over multiple frames."""
    id: int
    particles: List[Particle] = field(default_factory=list)
    
    @property
    def length(self) -> int:
        return len(self.particles)
    
    @property
    def start_frame(self) -> int:
        return self.particles[0].frame if self.particles else -1
    
    @property
    def end_frame(self) -> int:
        return self.particles[-1].frame if self.particles else -1
    
    @property
    def positions(self) -> np.ndarray:
        return np.array([[p.x, p.y] for p in self.particles])
    
    @property
    def frames(self) -> np.ndarray:
        return np.array([p.frame for p in self.particles])
    
    def __repr__(self):
        return f"Trajectory(id={self.id}, length={self.length}, frames={self.start_frame}-{self.end_frame})"


def compute_cost_matrix_fast(particles_t: List[Particle],
                             particles_t_r: List[Particle],
                             max_displacement: float,
                             use_moments: bool = True) -> Tuple[np.ndarray, List[int], List[int]]:
    """
    Build sparse cost matrix using KD-Tree for fast neighbor search.
    
    Only computes costs for particle pairs within max_displacement.
    
    Parameters
    ----------
    particles_t : list of Particle
        Particles in frame t
    particles_t_r : list of Particle
        Particles in frame t+r  
    max_displacement : float
        Maximum allowed displacement
    use_moments : bool
        Whether to include intensity moments in cost
        
    Returns
    -------
    cost_matrix : ndarray
        Cost matrix for linear_sum_assignment
    row_indices : list
        Mapping from cost matrix rows to particle indices
    col_indices : list
        Mapping from cost matrix cols to particle indices
    """
    n_t = len(particles_t)
    n_t_r = len(particles_t_r)
    
    if n_t == 0 or n_t_r == 0:
        return np.array([[]]), [], []
    
    # Extract positions as arrays
    pos_t = np.array([[p.x, p.y] for p in particles_t])
    pos_t_r = np.array([[p.x, p.y] for p in particles_t_r])
    
    # Build KD-tree for fast neighbor search
    tree = cKDTree(pos_t_r)
    
    # Find neighbors within max_displacement for each particle
    neighbors = tree.query_ball_point(pos_t, r=max_displacement)
    
    # Dummy cost for appearance/disappearance
    dummy_cost = max_displacement ** 2
    
    # Build cost matrix with dummy rows/columns
    # Size: (n_t + n_t_r) x (n_t + n_t_r) for rectangular assignment
    size = n_t + n_t_r
    cost = np.full((size, size), dummy_cost)
    
    # Extract moments if used
    if use_moments:
        m0_t = np.array([p.m0 for p in particles_t])
        m2_t = np.array([p.m2 for p in particles_t])
        m0_t_r = np.array([p.m0 for p in particles_t_r])
        m2_t_r = np.array([p.m2 for p in particles_t_r])
    
    # Fill in actual costs for valid pairs
    for i in range(n_t):
        for j in neighbors[i]:
            # Spatial distance squared
            dx = pos_t[i, 0] - pos_t_r[j, 0]
            dy = pos_t[i, 1] - pos_t_r[j, 1]
            dist_sq = dx*dx + dy*dy
            
            if use_moments:
                # Add moment differences
                dm0_sq = (m0_t[i] - m0_t_r[j])**2
                dm2_sq = (m2_t[i] - m2_t_r[j])**2
                c = dist_sq + dm0_sq + dm2_sq
            else:
                c = dist_sq
            
            cost[i, j] = c
    
    # Lower triangular part: costs for "lost" particles (linking to nothing)
    # Upper triangular part: costs for "new" particles (appearing from nothing)
    # These are already set to dummy_cost
    
    return cost, list(range(n_t)), list(range(n_t_r))


def link_frame_pair_fast(particles_t: List[Particle],
                         particles_t_r: List[Particle],
                         max_displacement: float,
                         use_moments: bool = True) -> Dict[int, int]:
    """
    Link particles between two frames using Hungarian algorithm.
    
    Parameters
    ----------
    particles_t : list of Particle
        Particles in frame t
    particles_t_r : list of Particle
        Particles in frame t+r
    max_displacement : float
        Maximum allowed displacement
    use_moments : bool
        Whether to include intensity moments in cost
        
    Returns
    -------
    links : dict
        Mapping from index in particles_t to index in particles_t_r
        Missing keys indicate particle was lost
    """
    n_t = len(particles_t)
    n_t_r = len(particles_t_r)
    
    if n_t == 0 or n_t_r == 0:
        return {}
    
    # Build cost matrix
    cost, _, _ = compute_cost_matrix_fast(
        particles_t, particles_t_r, max_displacement, use_moments
    )
    
    # Solve assignment problem (Hungarian algorithm)
    row_ind, col_ind = linear_sum_assignment(cost)
    
    # Extract valid links (not to dummy particles)
    dummy_cost = max_displacement ** 2
    links = {}
    
    for i, j in zip(row_ind, col_ind):
        if i < n_t and j < n_t_r:
            # Check if this is a real link (not to dummy)
            if cost[i, j] < dummy_cost:
                links[i] = j
    
    return links


def link_trajectories_fast(all_particles: List[List[Particle]],
                           max_displacement: float,
                           link_range: int = 1,
                           min_length: int = 1,
                           use_moments: bool = True,
                           verbose: bool = False) -> List[Trajectory]:
    """
    Link particles across all frames into trajectories.
    
    Uses optimized Hungarian algorithm with KD-tree acceleration.
    
    Parameters
    ----------
    all_particles : list of list of Particle
        Particles detected in each frame
    max_displacement : float
        Maximum displacement per frame
    link_range : int
        Number of frames to look ahead for gap closing
    min_length : int
        Minimum trajectory length to keep
    use_moments : bool
        Whether to use intensity moments in linking cost
    verbose : bool
        Print progress
        
    Returns
    -------
    trajectories : list of Trajectory
        Linked trajectories
    """
    n_frames = len(all_particles)
    
    if n_frames == 0:
        return []
    
    # Initialize: each particle in frame 0 starts a trajectory
    active_trajectories: Dict[int, Trajectory] = {}  # traj_id -> Trajectory
    particle_to_traj: Dict[Tuple[int, int], int] = {}  # (frame, particle_idx) -> traj_id
    
    next_traj_id = 0
    
    for idx, p in enumerate(all_particles[0]):
        traj = Trajectory(id=next_traj_id, particles=[p])
        active_trajectories[next_traj_id] = traj
        particle_to_traj[(0, idx)] = next_traj_id
        next_traj_id += 1
    
    # Process each frame
    for t in range(n_frames - 1):
        if verbose and (t % 100 == 0 or t == n_frames - 2):
            print(f"Linking frame {t+1}/{n_frames-1}, {len(active_trajectories)} active trajectories")
        
        particles_t = all_particles[t]
        
        # Try linking to frames t+1, t+2, ..., t+link_range
        for r in range(1, min(link_range + 1, n_frames - t)):
            particles_t_r = all_particles[t + r]
            
            if not particles_t or not particles_t_r:
                continue
            
            # Scale max displacement by frame gap
            max_disp_r = max_displacement * r
            
            # Get links
            links = link_frame_pair_fast(
                particles_t, particles_t_r, max_disp_r, use_moments
            )
            
            # Update trajectories
            for i, j in links.items():
                # Check if particle i at frame t belongs to a trajectory
                key_t = (t, i)
                if key_t in particle_to_traj:
                    traj_id = particle_to_traj[key_t]
                    
                    # Check if particle j at frame t+r is already assigned
                    key_t_r = (t + r, j)
                    if key_t_r not in particle_to_traj:
                        # Extend trajectory
                        traj = active_trajectories[traj_id]
                        
                        # Only add if this particle is newer than current end
                        if traj.end_frame < t + r:
                            traj.particles.append(particles_t_r[j])
                            particle_to_traj[key_t_r] = traj_id
        
        # Start new trajectories for unlinked particles in frame t+1
        for idx, p in enumerate(all_particles[t + 1]):
            key = (t + 1, idx)
            if key not in particle_to_traj:
                traj = Trajectory(id=next_traj_id, particles=[p])
                active_trajectories[next_traj_id] = traj
                particle_to_traj[key] = next_traj_id
                next_traj_id += 1
    
    # Filter by minimum length
    trajectories = [
        traj for traj in active_trajectories.values()
        if traj.length >= min_length
    ]
    
    # Sort by trajectory ID
    trajectories.sort(key=lambda t: t.id)
    
    # Reassign IDs
    for i, traj in enumerate(trajectories):
        traj.id = i
    
    if verbose:
        total_particles = sum(traj.length for traj in trajectories)
        print(f"Created {len(trajectories)} trajectories with {total_particles} total particles")
    
    return trajectories


# Keep old function name for compatibility
def link_trajectories(all_particles: List[List[Particle]],
                     max_displacement: float,
                     link_range: int = 1,
                     min_length: int = 1,
                     verbose: bool = False) -> List[Trajectory]:
    """Alias for link_trajectories_fast."""
    return link_trajectories_fast(
        all_particles, max_displacement, link_range, min_length,
        use_moments=True, verbose=verbose
    )


def link_particles_pair(particles_t: List[Particle],
                        particles_t_r: List[Particle],
                        max_displacement: float) -> Dict[int, int]:
    """Alias for link_frame_pair_fast."""
    return link_frame_pair_fast(particles_t, particles_t_r, max_displacement)


def compute_cost(p1: Particle, p2: Particle) -> float:
    """
    Compute linking cost between two particles.
    Kept for compatibility.
    """
    dx = p1.x - p2.x
    dy = p1.y - p2.y
    dist_sq = dx**2 + dy**2
    dm0_sq = (p1.m0 - p2.m0)**2
    dm2_sq = (p1.m2 - p2.m2)**2
    return dist_sq + dm0_sq + dm2_sq
