"""
Trajectory Linking Module

Implementation of the trajectory linking algorithm from:
Sbalzarini & Koumoutsakos (2005) "Feature point tracking and trajectory 
analysis for video imaging in cell biology" J. Struct. Biol. 151(2):182-195

Based on the graph-theoretic approach by Dalziel (1992, 1993) using
the transportation problem (Hitchcock, 1941).
"""

import numpy as np
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
        """Number of particles in trajectory."""
        return len(self.particles)
    
    @property
    def start_frame(self) -> int:
        """First frame of trajectory."""
        return self.particles[0].frame if self.particles else -1
    
    @property
    def end_frame(self) -> int:
        """Last frame of trajectory."""
        return self.particles[-1].frame if self.particles else -1
    
    @property
    def positions(self) -> np.ndarray:
        """Get (x, y) positions as array."""
        return np.array([[p.x, p.y] for p in self.particles])
    
    @property
    def frames(self) -> np.ndarray:
        """Get frame indices."""
        return np.array([p.frame for p in self.particles])
    
    def __repr__(self):
        return f"Trajectory(id={self.id}, length={self.length}, frames={self.start_frame}-{self.end_frame})"


def compute_cost(p1: Particle, p2: Particle) -> float:
    """
    Compute linking cost between two particles.
    
    Cost includes:
    - Squared Euclidean distance
    - Squared difference in m0 (intensity)
    - Squared difference in m2 (spatial extent)
    
    Parameters
    ----------
    p1, p2 : Particle
        Two particles to compute cost between
        
    Returns
    -------
    cost : float
        Total linking cost
    """
    # Spatial distance
    dx = p1.x - p2.x
    dy = p1.y - p2.y
    dist_sq = dx**2 + dy**2
    
    # Intensity moment differences
    dm0_sq = (p1.m0 - p2.m0)**2
    dm2_sq = (p1.m2 - p2.m2)**2
    
    return dist_sq + dm0_sq + dm2_sq


def build_cost_matrix(particles_t: List[Particle],
                     particles_t_r: List[Particle],
                     max_displacement: float) -> Tuple[np.ndarray, float]:
    """
    Build cost matrix for linking particles between frames.
    
    Includes dummy particles for appearance/disappearance.
    Row 0 = dummy in frame t (for new particles)
    Column 0 = dummy in frame t+r (for lost particles)
    
    Parameters
    ----------
    particles_t : list of Particle
        Particles in frame t
    particles_t_r : list of Particle
        Particles in frame t+r
    max_displacement : float
        Maximum allowed displacement
        
    Returns
    -------
    cost_matrix : ndarray
        Cost matrix of shape (n_t + 1, n_t_r + 1)
    dummy_cost : float
        Cost for linking to dummy particle
    """
    n_t = len(particles_t)
    n_t_r = len(particles_t_r)
    
    # Dummy cost is r * L^2 where r is number of frames apart
    # For simplicity we use L^2
    dummy_cost = max_displacement ** 2
    
    # Initialize cost matrix with infinity
    cost = np.full((n_t + 1, n_t_r + 1), np.inf)
    
    # Cost for dummy associations
    cost[0, 1:] = dummy_cost  # New particles
    cost[1:, 0] = dummy_cost  # Lost particles
    cost[0, 0] = 0  # Dummy to dummy (no cost)
    
    # Compute actual costs
    for i, p1 in enumerate(particles_t):
        for j, p2 in enumerate(particles_t_r):
            c = compute_cost(p1, p2)
            # Only allow if displacement is within limit
            dx = p1.x - p2.x
            dy = p1.y - p2.y
            dist = np.sqrt(dx**2 + dy**2)
            
            if dist <= max_displacement:
                cost[i + 1, j + 1] = c
    
    return cost, dummy_cost


def initialize_assignment(cost: np.ndarray) -> np.ndarray:
    """
    Initialize assignment matrix using greedy nearest-neighbor approach.
    
    Parameters
    ----------
    cost : ndarray
        Cost matrix
        
    Returns
    -------
    assignment : ndarray
        Binary assignment matrix (same shape as cost)
    """
    n_rows, n_cols = cost.shape
    assignment = np.zeros_like(cost, dtype=int)
    
    # Track which columns are assigned
    assigned_cols = set()
    
    # For each row (except dummy), find best unassigned column
    for i in range(1, n_rows):
        valid_costs = []
        for j in range(n_cols):
            if j not in assigned_cols or j == 0:  # dummy can be assigned multiple times
                if cost[i, j] < np.inf:
                    valid_costs.append((cost[i, j], j))
        
        if valid_costs:
            # Sort by cost and pick minimum
            valid_costs.sort()
            _, best_j = valid_costs[0]
            assignment[i, best_j] = 1
            if best_j != 0:  # Don't mark dummy as assigned
                assigned_cols.add(best_j)
        else:
            # Assign to dummy
            assignment[i, 0] = 1
    
    # For unassigned columns, link from dummy
    for j in range(1, n_cols):
        if j not in assigned_cols:
            assignment[0, j] = 1
    
    return assignment


def optimize_assignment(cost: np.ndarray, assignment: np.ndarray, 
                       max_iterations: int = 1000) -> np.ndarray:
    """
    Optimize assignment using reduced cost iteration.
    
    At each step, find the association with most negative reduced cost
    and swap assignments accordingly.
    
    Parameters
    ----------
    cost : ndarray
        Cost matrix
    assignment : ndarray
        Initial assignment matrix
        
    Returns
    -------
    assignment : ndarray
        Optimized assignment matrix
    """
    n_rows, n_cols = cost.shape
    assignment = assignment.copy()
    
    for iteration in range(max_iterations):
        best_reduced_cost = 0
        best_swap = None
        
        # Find the swap with most negative reduced cost
        for I in range(1, n_rows):
            for J in range(1, n_cols):
                if assignment[I, J] == 0 and cost[I, J] < np.inf:
                    # Find current assignments: g_IL = 1 and g_KJ = 1
                    L = np.where(assignment[I, :] == 1)[0]
                    K = np.where(assignment[:, J] == 1)[0]
                    
                    if len(L) == 0 or len(K) == 0:
                        continue
                    
                    L = L[0]
                    K = K[0]
                    
                    # Compute reduced cost
                    # z_IJ = phi_IJ - phi_IL - phi_KJ + phi_KL
                    if K > 0:  # Regular particle to regular particle
                        z = cost[I, J] - cost[I, L] - cost[K, J] + cost[K, L]
                    else:  # Appearing particle
                        z = cost[I, J] - cost[I, L] - cost[0, J]
                    
                    if z < best_reduced_cost:
                        best_reduced_cost = z
                        best_swap = (I, J, K, L)
        
        # Also check dummy swaps
        for I in range(1, n_rows):
            if assignment[I, 0] == 0 and cost[I, 0] < np.inf:
                L = np.where(assignment[I, :] == 1)[0]
                if len(L) > 0 and L[0] > 0:
                    L = L[0]
                    z = cost[I, 0] - cost[I, L] + cost[0, L]
                    if z < best_reduced_cost:
                        best_reduced_cost = z
                        best_swap = (I, 0, 0, L)
        
        for J in range(1, n_cols):
            if assignment[0, J] == 0 and cost[0, J] < np.inf:
                K = np.where(assignment[:, J] == 1)[0]
                if len(K) > 0 and K[0] > 0:
                    K = K[0]
                    z = cost[0, J] - cost[K, J] + cost[K, 0]
                    if z < best_reduced_cost:
                        best_reduced_cost = z
                        best_swap = (0, J, K, 0)
        
        if best_swap is None or best_reduced_cost >= 0:
            break
        
        # Perform swap
        I, J, K, L = best_swap
        assignment[I, J] = 1
        if L > 0:
            assignment[I, L] = 0
        if K > 0:
            assignment[K, J] = 0
        if K > 0 and L > 0:
            assignment[K, L] = 1
        elif K == 0:
            assignment[0, J] = 0
        elif L == 0:
            assignment[I, 0] = 0
    
    return assignment


def link_particles_pair(particles_t: List[Particle],
                       particles_t_r: List[Particle],
                       max_displacement: float) -> List[Tuple[int, int]]:
    """
    Link particles between two frames.
    
    Parameters
    ----------
    particles_t : list of Particle
        Particles in frame t
    particles_t_r : list of Particle
        Particles in frame t+r
    max_displacement : float
        Maximum allowed displacement
        
    Returns
    -------
    links : list of (i, j) tuples
        Links between particle indices. i=-1 means new particle,
        j=-1 means lost particle.
    """
    if len(particles_t) == 0 and len(particles_t_r) == 0:
        return []
    
    if len(particles_t) == 0:
        # All particles are new
        return [(-1, j) for j in range(len(particles_t_r))]
    
    if len(particles_t_r) == 0:
        # All particles are lost
        return [(i, -1) for i in range(len(particles_t))]
    
    # Build cost matrix
    cost, _ = build_cost_matrix(particles_t, particles_t_r, max_displacement)
    
    # Initialize and optimize assignment
    assignment = initialize_assignment(cost)
    assignment = optimize_assignment(cost, assignment)
    
    # Extract links
    links = []
    for i in range(1, assignment.shape[0]):
        j = np.where(assignment[i, :] == 1)[0]
        if len(j) > 0:
            j = j[0]
            if j == 0:
                links.append((i - 1, -1))  # Lost particle
            else:
                links.append((i - 1, j - 1))
    
    # New particles
    for j in range(1, assignment.shape[1]):
        if assignment[0, j] == 1:
            links.append((-1, j - 1))
    
    return links


def link_trajectories(all_particles: List[List[Particle]],
                     max_displacement: float,
                     link_range: int = 1,
                     min_length: int = 1) -> List[Trajectory]:
    """
    Link particles across all frames into trajectories.
    
    Parameters
    ----------
    all_particles : list of list of Particle
        Particles detected in each frame
    max_displacement : float
        Maximum displacement per frame
    link_range : int
        Number of future frames to consider for linking.
        Higher values help with temporary occlusion.
    min_length : int
        Minimum trajectory length to keep
        
    Returns
    -------
    trajectories : list of Trajectory
        Linked trajectories
    """
    n_frames = len(all_particles)
    if n_frames == 0:
        return []
    
    # Initialize trajectory ID counter
    next_id = 0
    
    # Track active trajectories: maps particle index in current frame to trajectory
    active_trajs: Dict[int, Trajectory] = {}
    
    # All completed trajectories
    completed: List[Trajectory] = []
    
    # Process first frame - all particles start new trajectories
    for j, p in enumerate(all_particles[0]):
        traj = Trajectory(id=next_id, particles=[p])
        active_trajs[j] = traj
        next_id += 1
    
    # Process subsequent frames
    for t in range(n_frames - 1):
        particles_t = all_particles[t]
        
        # Link with future frames up to link_range
        all_links = {}  # particle_idx -> list of (frame_offset, target_idx, cost)
        
        for r in range(1, min(link_range + 1, n_frames - t)):
            particles_tr = all_particles[t + r]
            links = link_particles_pair(
                particles_t, 
                particles_tr,
                max_displacement * r  # Scale displacement by frame offset
            )
            
            for src, dst in links:
                if src >= 0 and dst >= 0:
                    if src not in all_links:
                        all_links[src] = []
                    # Compute cost for this link
                    c = compute_cost(particles_t[src], particles_tr[dst])
                    all_links[src].append((r, dst, c))
        
        # Determine best link for each particle
        new_active: Dict[int, Trajectory] = {}
        linked_targets: Dict[int, Tuple[int, int]] = {}  # target_idx -> (frame_offset, src_idx)
        
        for src_idx, link_options in all_links.items():
            if src_idx not in active_trajs:
                continue
            
            # Sort by cost and pick best
            link_options.sort(key=lambda x: x[2])
            best_r, best_dst, best_cost = link_options[0]
            
            if best_r == 1:  # Direct link to next frame
                traj = active_trajs[src_idx]
                traj.particles.append(all_particles[t + 1][best_dst])
                
                # Check if this target was already linked
                if best_dst in new_active:
                    # Conflict - keep lower cost
                    existing_traj = new_active[best_dst]
                    if best_cost < compute_cost(
                        existing_traj.particles[-2], 
                        existing_traj.particles[-1]
                    ):
                        # Replace
                        existing_traj.particles.pop()
                        completed.append(existing_traj)
                        new_active[best_dst] = traj
                    else:
                        traj.particles.pop()
                        completed.append(traj)
                else:
                    new_active[best_dst] = traj
            else:
                # Gap in trajectory - particle temporarily lost
                traj = active_trajs[src_idx]
                linked_targets[best_dst] = (best_r, src_idx)
        
        # Handle particles not linked to next frame
        for src_idx, traj in active_trajs.items():
            if src_idx not in all_links or all(r > 1 for r, _, _ in all_links.get(src_idx, [(2, 0, 0)])):
                # No link or only gap links - trajectory ends here
                if traj not in [new_active.get(k) for k in new_active]:
                    completed.append(traj)
        
        # Handle new particles in next frame
        for j, p in enumerate(all_particles[t + 1]):
            if j not in new_active:
                # Check if this particle fills a gap
                if j in linked_targets:
                    r, src_idx = linked_targets[j]
                    if src_idx in active_trajs:
                        traj = active_trajs[src_idx]
                        traj.particles.append(p)
                        new_active[j] = traj
                        continue
                
                # Start new trajectory
                traj = Trajectory(id=next_id, particles=[p])
                new_active[j] = traj
                next_id += 1
        
        active_trajs = new_active
    
    # Add remaining active trajectories
    completed.extend(active_trajs.values())
    
    # Filter by minimum length
    trajectories = [t for t in completed if t.length >= min_length]
    
    # Sort by ID
    trajectories.sort(key=lambda t: t.id)
    
    return trajectories
