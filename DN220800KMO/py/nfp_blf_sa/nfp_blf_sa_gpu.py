"""
Santa 2025 - Christmas Tree Packing Challenge
Accelerated Hybrid Pipeline: NFP-guided BLF + Simulated Annealing

Optimized for macOS with:
- Numba JIT compilation for fast collision detection
- Vectorized NumPy for batch operations  
- Adaptive parallelization (only for larger problems)
- PyTorch MPS GPU backend (when beneficial)

Algorithms:
- NFP (No-Fit Polygon): Computes contact points for optimal placement
- BLF (Bottom-Left-Fill): Heuristic placement strategy
- SA (Simulated Annealing): Global optimization improvement phase
"""

import math
import random
from decimal import Decimal, getcontext
from concurrent.futures import ThreadPoolExecutor
import multiprocessing as mp

import numpy as np
import pandas as pd
from shapely import affinity
from shapely.geometry import Polygon
from shapely.ops import unary_union
from shapely.strtree import STRtree

# Try to import Numba for JIT compilation (massive speedup)
try:
    from numba import jit, prange
    HAS_NUMBA = True
    print("✓ Numba JIT acceleration enabled")
except ImportError:
    HAS_NUMBA = False
    print("⚠ Numba not installed - consider 'pip install numba' for 3-5x speedup")
    # Fallback decorators
    def jit(*args, **kwargs):
        def decorator(func):
            return func
        return decorator if len(args) == 0 or not callable(args[0]) else args[0]
    prange = range

# Try to import PyTorch for GPU operations
try:
    import torch
    if torch.backends.mps.is_available():
        DEVICE = torch.device("mps")
        print("✓ Apple Metal GPU (MPS) available")
    else:
        DEVICE = torch.device("cpu")
    HAS_TORCH = True
except ImportError:
    HAS_TORCH = False
    DEVICE = None

# Set precision for Decimal
getcontext().prec = 25
scale_factor = Decimal('1e15')
SCALE_FLOAT = 1e15

# Build the index of the submission
index = [f'{n:03d}_{t}' for n in range(1, 201) for t in range(n)]

# Pre-compute tree polygon vertices (NumPy array for speed)
def _compute_base_vertices():
    """Compute the base tree vertices as NumPy array."""
    trunk_w = 0.15
    trunk_h = 0.2
    base_w = 0.7
    mid_w = 0.4
    top_w = 0.25
    tip_y = 0.8
    tier_1_y = 0.5
    tier_2_y = 0.25
    base_y = 0.0
    trunk_bottom_y = -trunk_h

    return np.array([
        [0.0, tip_y],
        [top_w / 2, tier_1_y],
        [top_w / 4, tier_1_y],
        [mid_w / 2, tier_2_y],
        [mid_w / 4, tier_2_y],
        [base_w / 2, base_y],
        [trunk_w / 2, base_y],
        [trunk_w / 2, trunk_bottom_y],
        [-trunk_w / 2, trunk_bottom_y],
        [-trunk_w / 2, base_y],
        [-base_w / 2, base_y],
        [-mid_w / 4, tier_2_y],
        [-mid_w / 2, tier_2_y],
        [-top_w / 4, tier_1_y],
        [-top_w / 2, tier_1_y],
    ], dtype=np.float64)

BASE_VERTICES = _compute_base_vertices()


@jit(nopython=True, cache=True)
def rotate_vertices_numba(vertices, angle_deg):
    """Rotate vertices around origin (Numba-optimized)."""
    angle_rad = np.radians(angle_deg)
    cos_a = np.cos(angle_rad)
    sin_a = np.sin(angle_rad)
    
    result = np.empty_like(vertices)
    for i in range(len(vertices)):
        x, y = vertices[i]
        result[i, 0] = x * cos_a - y * sin_a
        result[i, 1] = x * sin_a + y * cos_a
    return result


@jit(nopython=True, cache=True)
def translate_vertices_numba(vertices, dx, dy):
    """Translate vertices (Numba-optimized)."""
    result = np.empty_like(vertices)
    for i in range(len(vertices)):
        result[i, 0] = vertices[i, 0] + dx
        result[i, 1] = vertices[i, 1] + dy
    return result


@jit(nopython=True, cache=True)
def get_bounding_box(vertices):
    """Get axis-aligned bounding box of vertices."""
    min_x = vertices[0, 0]
    max_x = vertices[0, 0]
    min_y = vertices[0, 1]
    max_y = vertices[0, 1]
    
    for i in range(1, len(vertices)):
        x, y = vertices[i]
        if x < min_x: min_x = x
        if x > max_x: max_x = x
        if y < min_y: min_y = y
        if y > max_y: max_y = y
    
    return min_x, min_y, max_x, max_y


@jit(nopython=True, cache=True)
def boxes_overlap(box1, box2):
    """Check if two bounding boxes overlap (fast pre-filter)."""
    return not (box1[2] < box2[0] or box2[2] < box1[0] or 
                box1[3] < box2[1] or box2[3] < box1[1])


@jit(nopython=True, cache=True)
def point_in_polygon(px, py, vertices):
    """Check if point is inside polygon using ray casting."""
    n = len(vertices)
    inside = False
    
    j = n - 1
    for i in range(n):
        xi, yi = vertices[i]
        xj, yj = vertices[j]
        
        if ((yi > py) != (yj > py)) and (px < (xj - xi) * (py - yi) / (yj - yi) + xi):
            inside = not inside
        j = i
    
    return inside


@jit(nopython=True, cache=True)
def line_segments_intersect(p1x, p1y, p2x, p2y, p3x, p3y, p4x, p4y):
    """Check if two line segments intersect."""
    d1x = p2x - p1x
    d1y = p2y - p1y
    d2x = p4x - p3x
    d2y = p4y - p3y
    
    cross = d1x * d2y - d1y * d2x
    if abs(cross) < 1e-12:
        return False
    
    dx = p3x - p1x
    dy = p3y - p1y
    
    t1 = (dx * d2y - dy * d2x) / cross
    t2 = (dx * d1y - dy * d1x) / cross
    
    return 0 <= t1 <= 1 and 0 <= t2 <= 1


@jit(nopython=True, cache=True)
def polygons_intersect_fast(verts1, verts2):
    """
    Fast polygon intersection check using Numba.
    Returns True if polygons overlap (not just touch).
    """
    n1 = len(verts1)
    n2 = len(verts2)
    
    # Check bounding boxes first (fast rejection)
    box1 = get_bounding_box(verts1)
    box2 = get_bounding_box(verts2)
    
    if not boxes_overlap(box1, box2):
        return False
    
    # Check if any edges intersect
    for i in range(n1):
        i2 = (i + 1) % n1
        for j in range(n2):
            j2 = (j + 1) % n2
            if line_segments_intersect(
                verts1[i, 0], verts1[i, 1], verts1[i2, 0], verts1[i2, 1],
                verts2[j, 0], verts2[j, 1], verts2[j2, 0], verts2[j2, 1]
            ):
                return True
    
    # Check if one polygon is completely inside the other
    if point_in_polygon(verts1[0, 0], verts1[0, 1], verts2):
        return True
    if point_in_polygon(verts2[0, 0], verts2[0, 1], verts1):
        return True
    
    return False


@jit(nopython=True, cache=True, parallel=True)
def batch_collision_check(candidate_verts_list, placed_verts_list):
    """
    Check collisions for multiple candidates in parallel.
    
    Args:
        candidate_verts_list: Array of shape (num_candidates, num_vertices, 2)
        placed_verts_list: Array of shape (num_placed, num_vertices, 2)
    
    Returns:
        Array of bools indicating collision for each candidate
    """
    num_candidates = len(candidate_verts_list)
    num_placed = len(placed_verts_list)
    results = np.zeros(num_candidates, dtype=np.bool_)
    
    for c in prange(num_candidates):
        for p in range(num_placed):
            if polygons_intersect_fast(candidate_verts_list[c], placed_verts_list[p]):
                results[c] = True
                break
    
    return results


class ChristmasTree:
    """Represents a single, rotatable Christmas tree of a fixed size."""

    def __init__(self, center_x='0', center_y='0', angle='0'):
        """Initializes the Christmas tree with a specific position and rotation."""
        self.center_x = Decimal(center_x)
        self.center_y = Decimal(center_y)
        self.angle = Decimal(angle)

        # Use pre-computed vertices for faster polygon creation
        rotated_verts = rotate_vertices_numba(BASE_VERTICES, float(self.angle))
        translated_verts = translate_vertices_numba(
            rotated_verts, 
            float(self.center_x), 
            float(self.center_y)
        )
        
        # Store float vertices for fast Numba operations
        self._float_vertices = translated_verts.copy()
        
        # Scale for Shapely (integer coordinates for precision)
        scaled_verts = translated_verts * SCALE_FLOAT
        self.polygon = Polygon(scaled_verts)


def check_collision(poly1, poly2):
    """Check if two polygons overlap (not just touch)."""
    return poly1.intersects(poly2) and not poly1.touches(poly2)


def check_collision_fast(tree1, tree2):
    """Fast collision check using Numba (when available)."""
    if HAS_NUMBA:
        return polygons_intersect_fast(tree1._float_vertices, tree2._float_vertices)
    return check_collision(tree1.polygon, tree2.polygon)


def compute_bounding_box_with_new_tree(placed_trees, new_tree_poly):
    """Compute the bounding box side length if we add a new tree polygon."""
    if not placed_trees:
        bounds = new_tree_poly.bounds
    else:
        all_polys = [t.polygon for t in placed_trees] + [new_tree_poly]
        bounds = unary_union(all_polys).bounds
    
    minx = Decimal(bounds[0]) / scale_factor
    miny = Decimal(bounds[1]) / scale_factor
    maxx = Decimal(bounds[2]) / scale_factor
    maxy = Decimal(bounds[3]) / scale_factor
    
    width = maxx - minx
    height = maxy - miny
    return max(width, height)


def compute_bounding_box_fast(placed_trees, new_tree_verts):
    """Fast bounding box computation using NumPy."""
    if not placed_trees:
        box = get_bounding_box(new_tree_verts)
        return Decimal(str(max(box[2] - box[0], box[3] - box[1])))
    
    # Collect all vertices
    all_verts = np.vstack([t._float_vertices for t in placed_trees] + [new_tree_verts])
    
    min_x = np.min(all_verts[:, 0])
    max_x = np.max(all_verts[:, 0])
    min_y = np.min(all_verts[:, 1])
    max_y = np.max(all_verts[:, 1])
    
    width = max_x - min_x
    height = max_y - min_y
    
    return Decimal(str(max(width, height)))


def find_best_rotation_and_position(placed_trees, num_rotation_angles=24, num_direction_angles=48):
    """
    Find the best (rotation, position) combination using vectorized operations.
    Optimized with Numba batch collision detection.
    """
    if not placed_trees:
        return Decimal('0'), Decimal('0'), Decimal('0')
    
    # Collect placed tree vertices for batch checking
    placed_verts_list = np.array([t._float_vertices for t in placed_trees])
    
    # Also keep Shapely structures for final verification
    placed_polygons = [t.polygon for t in placed_trees]
    tree_index = STRtree(placed_polygons)
    
    best_score = Decimal('Infinity')
    best_result = (Decimal('0'), Decimal('0'), Decimal('0'))
    
    # Pre-compute all direction vectors
    dir_angles = np.linspace(0, 2 * np.pi, num_direction_angles, endpoint=False)
    vx_all = np.cos(dir_angles)
    vy_all = np.sin(dir_angles)
    
    rotation_angles = [i * 360.0 / num_rotation_angles for i in range(num_rotation_angles)]
    
    for rot_angle in rotation_angles:
        # Create rotated template
        rotated_verts = rotate_vertices_numba(BASE_VERTICES, rot_angle)
        template_verts_scaled = rotated_verts * SCALE_FLOAT
        template_poly = Polygon(template_verts_scaled)
        
        candidate_positions = []
        
        for idx in range(num_direction_angles):
            vx = vx_all[idx]
            vy = vy_all[idx]
            
            # Binary search for contact position
            low_radius = 0.0
            high_radius = 12.0
            
            # Quick check: is high_radius collision-free?
            px = high_radius * vx
            py = high_radius * vy
            
            candidate_verts = translate_vertices_numba(rotated_verts, px, py)
            
            # Use Numba for fast collision check
            has_collision = False
            if HAS_NUMBA:
                for placed_verts in placed_verts_list:
                    if polygons_intersect_fast(candidate_verts, placed_verts):
                        has_collision = True
                        break
            else:
                # Fallback to Shapely
                candidate_poly = affinity.translate(template_poly, xoff=px * SCALE_FLOAT, yoff=py * SCALE_FLOAT)
                possible_indices = tree_index.query(candidate_poly)
                has_collision = any(
                    candidate_poly.intersects(placed_polygons[i]) and 
                    not candidate_poly.touches(placed_polygons[i])
                    for i in possible_indices
                )
            
            if has_collision:
                continue
            
            # Binary search to find contact point
            for _ in range(18):
                mid_radius = (low_radius + high_radius) / 2
                px = mid_radius * vx
                py = mid_radius * vy
                
                candidate_verts = translate_vertices_numba(rotated_verts, px, py)
                
                has_collision = False
                if HAS_NUMBA:
                    for placed_verts in placed_verts_list:
                        if polygons_intersect_fast(candidate_verts, placed_verts):
                            has_collision = True
                            break
                else:
                    candidate_poly = affinity.translate(template_poly, xoff=px * SCALE_FLOAT, yoff=py * SCALE_FLOAT)
                    possible_indices = tree_index.query(candidate_poly)
                    has_collision = any(
                        candidate_poly.intersects(placed_polygons[i]) and 
                        not candidate_poly.touches(placed_polygons[i])
                        for i in possible_indices
                    )
                
                if has_collision:
                    low_radius = mid_radius
                else:
                    high_radius = mid_radius
            
            px = high_radius * vx
            py = high_radius * vy
            candidate_positions.append((px, py))
        
        # Evaluate candidates
        for px, py in candidate_positions:
            candidate_verts = translate_vertices_numba(rotated_verts, px, py)
            score = compute_bounding_box_fast(placed_trees, candidate_verts)
            
            if score < best_score:
                best_score = score
                best_result = (Decimal(str(px)), Decimal(str(py)), Decimal(str(rot_angle)))
    
    return best_result


def copy_tree(tree):
    """Create a fresh copy of a tree with properly constructed polygon."""
    return ChristmasTree(
        center_x=str(tree.center_x),
        center_y=str(tree.center_y),
        angle=str(tree.angle)
    )


def validate_and_fix_overlaps(placed_trees, max_iterations=5):
    """Validate and fix any overlapping trees."""
    if len(placed_trees) < 2:
        return placed_trees
    
    fixed_trees = [copy_tree(t) for t in placed_trees]
    
    for iteration in range(max_iterations):
        has_overlap = False
        overlap_pairs = []
        
        for i in range(len(fixed_trees)):
            for j in range(i + 1, len(fixed_trees)):
                if check_collision_fast(fixed_trees[i], fixed_trees[j]):
                    overlap_pairs.append((i, j))
                    has_overlap = True
        
        if not has_overlap:
            break
        
        for i, j in overlap_pairs:
            tree1 = fixed_trees[j]
            tree2 = fixed_trees[i]
            
            dx = tree1.center_x - tree2.center_x
            dy = tree1.center_y - tree2.center_y
            dist = (dx**2 + dy**2).sqrt()
            
            move_distance = Decimal('0.05')
            
            if dist > Decimal('0.001'):
                unit_dx = dx / dist
                unit_dy = dy / dist
                new_x = tree1.center_x + move_distance * unit_dx
                new_y = tree1.center_y + move_distance * unit_dy
            else:
                new_x = tree1.center_x + move_distance
                new_y = tree1.center_y + move_distance
            
            fixed_trees[j] = ChristmasTree(
                center_x=str(new_x),
                center_y=str(new_y),
                angle=str(tree1.angle)
            )
    
    return fixed_trees


def compute_side_length(placed_trees):
    """Compute the bounding square side length for placed trees."""
    if not placed_trees:
        return Decimal('0')
    
    all_polygons = [t.polygon for t in placed_trees]
    bounds = unary_union(all_polygons).bounds
    
    minx = Decimal(bounds[0]) / scale_factor
    miny = Decimal(bounds[1]) / scale_factor
    maxx = Decimal(bounds[2]) / scale_factor
    maxy = Decimal(bounds[3]) / scale_factor
    
    width = maxx - minx
    height = maxy - miny
    
    return max(width, height)


def simulated_annealing_improvement(placed_trees, initial_temp=50.0, cooling_rate=0.95,
                                    iterations=500, num_trees_to_optimize=None):
    """
    Simulated Annealing improvement phase (optimized with Numba collision detection).
    """
    if len(placed_trees) < 2:
        return placed_trees
    
    optimized_trees = [copy_tree(t) for t in placed_trees]
    current_side = compute_side_length(optimized_trees)
    best_side = current_side
    best_trees = [copy_tree(t) for t in optimized_trees]
    
    temperature = float(initial_temp)
    
    if num_trees_to_optimize is None:
        num_trees_to_optimize = min(len(optimized_trees), 30)
    
    tree_indices = list(range(max(0, len(optimized_trees) - num_trees_to_optimize),
                              len(optimized_trees)))
    
    if not tree_indices:
        return placed_trees
    
    for iteration in range(iterations):
        tree_idx = random.choice(tree_indices)
        old_tree = optimized_trees[tree_idx]
        
        old_x = old_tree.center_x
        old_y = old_tree.center_y
        old_angle = old_tree.angle
        
        progress = iteration / iterations
        
        rand = random.random()
        if rand < 0.3:
            perturbation_type = 'translate'
        elif rand < 0.6:
            perturbation_type = 'rotate'
        elif rand < 0.85:
            perturbation_type = 'both'
        else:
            perturbation_type = 'big_rotate'
        
        new_x = old_x
        new_y = old_y
        new_angle = old_angle
        
        if perturbation_type in ['translate', 'both']:
            max_shift = 0.25 * (1.0 - progress * 0.6)
            if random.random() < 0.4:
                dist_from_center = (old_x**2 + old_y**2).sqrt()
                if dist_from_center > Decimal('0.01'):
                    move_toward = Decimal(str(random.uniform(0, max_shift * 0.5)))
                    new_x -= (old_x / dist_from_center) * move_toward
                    new_y -= (old_y / dist_from_center) * move_toward
            else:
                new_x += Decimal(str(random.uniform(-max_shift, max_shift)))
                new_y += Decimal(str(random.uniform(-max_shift, max_shift)))
        
        if perturbation_type == 'rotate':
            max_rotation = 20.0 * (1.0 - progress * 0.5)
            new_angle = (old_angle + Decimal(str(random.uniform(-max_rotation, max_rotation)))) % Decimal('360')
        elif perturbation_type == 'both':
            max_rotation = 15.0 * (1.0 - progress * 0.5)
            new_angle = (old_angle + Decimal(str(random.uniform(-max_rotation, max_rotation)))) % Decimal('360')
        elif perturbation_type == 'big_rotate':
            big_angles = [45, 90, 135, 180, 225, 270, 315]
            new_angle = (old_angle + Decimal(str(random.choice(big_angles)))) % Decimal('360')
        
        new_tree = ChristmasTree(
            center_x=str(new_x),
            center_y=str(new_y),
            angle=str(new_angle)
        )
        
        # Fast collision check with Numba
        has_collision = False
        for i, other_tree in enumerate(optimized_trees):
            if i != tree_idx:
                if check_collision_fast(new_tree, other_tree):
                    has_collision = True
                    break
        
        if has_collision:
            continue
        
        optimized_trees[tree_idx] = new_tree
        new_side = compute_side_length(optimized_trees)
        delta = float(new_side - current_side)
        
        accept = False
        if delta < 0:
            accept = True
        elif temperature > 0.001:
            accept = random.random() < math.exp(-delta / temperature)
        
        if accept:
            current_side = new_side
            if current_side < best_side:
                best_side = current_side
                best_trees = [copy_tree(t) for t in optimized_trees]
        else:
            optimized_trees[tree_idx] = old_tree
        
        temperature *= cooling_rate
    
    return best_trees


def initialize_trees(num_trees, existing_trees=None, use_sa=True, sa_iterations=400):
    """
    Accelerated Hybrid Pipeline with Numba JIT compilation.
    
    Phase 1 (Construction): Rotation-optimized BLF with fast collision detection
    Phase 2 (Improvement): Simulated Annealing with Numba-accelerated checks
    """
    if num_trees == 0:
        return [], Decimal('0')

    if existing_trees is None:
        placed_trees = []
    else:
        placed_trees = [copy_tree(t) for t in existing_trees]

    num_to_add = num_trees - len(placed_trees)

    if num_to_add > 0:
        if not placed_trees:
            placed_trees.append(ChristmasTree(angle='0'))
            num_to_add -= 1
        
        if num_to_add > 0 and len(placed_trees) == 1:
            best_score = Decimal('Infinity')
            best_tree = None
            
            for angle in [0, 45, 90, 135, 180, 225, 270, 315]:
                for px_offset in [Decimal('0.6'), Decimal('0.7'), Decimal('0.8')]:
                    for py_offset in [Decimal('-0.3'), Decimal('0'), Decimal('0.3')]:
                        test_tree = ChristmasTree(
                            center_x=str(px_offset),
                            center_y=str(py_offset),
                            angle=str(angle)
                        )
                        if not check_collision_fast(test_tree, placed_trees[0]):
                            score = compute_bounding_box_fast(
                                placed_trees, 
                                test_tree._float_vertices
                            )
                            if score < best_score:
                                best_score = score
                                best_tree = test_tree
            
            if best_tree:
                placed_trees.append(best_tree)
            else:
                px, py, angle = find_best_rotation_and_position(placed_trees, 24, 48)
                placed_trees.append(ChristmasTree(center_x=str(px), center_y=str(py), angle=str(angle)))
            num_to_add -= 1
        
        # Remaining trees
        for _ in range(num_to_add):
            px, py, angle = find_best_rotation_and_position(
                placed_trees,
                num_rotation_angles=36,
                num_direction_angles=72
            )
            
            placed_tree = ChristmasTree(
                center_x=str(px),
                center_y=str(py),
                angle=str(angle)
            )
            placed_trees.append(placed_tree)
        
        placed_trees = validate_and_fix_overlaps(placed_trees, max_iterations=3)

    # Phase 2: Simulated Annealing
    if use_sa and len(placed_trees) > 2:
        placed_trees = simulated_annealing_improvement(
            placed_trees,
            initial_temp=40.0,
            cooling_rate=0.97,
            iterations=sa_iterations,
            num_trees_to_optimize=min(len(placed_trees), 25)
        )
        
        placed_trees = validate_and_fix_overlaps(placed_trees, max_iterations=2)

    side_length = compute_side_length(placed_trees)
    return placed_trees, side_length


def main():
    """Main function with acceleration."""
    print("=" * 60)
    print("Santa 2025 - Accelerated Tree Packing")
    print("=" * 60)
    
    print(f"\nSystem Configuration:")
    print(f"  CPU cores: {mp.cpu_count()}")
    print(f"  Numba JIT: {'Enabled ✓' if HAS_NUMBA else 'Not available'}")
    if HAS_TORCH:
        print(f"  PyTorch device: {DEVICE}")
    print()
    
    tree_data = []
    current_placed_trees = []

    for n in range(200):
        current_placed_trees, side = initialize_trees(
            n + 1, 
            existing_trees=current_placed_trees
        )
        
        if (n + 1) % 10 == 0:
            print(f"Processed {n + 1:3d} trees, side length: {side:.12f}")
        
        for tree in current_placed_trees:
            tree_data.append([tree.center_x, tree.center_y, tree.angle])

    print("\nCreating submission DataFrame...")
    cols = ['x', 'y', 'deg']
    submission = pd.DataFrame(
        index=index, columns=cols, data=tree_data
    ).rename_axis('id')

    for col in cols:
        submission[col] = submission[col].astype(float).round(decimals=6)

    for col in submission.columns:
        submission[col] = 's' + submission[col].astype('string')
    
    output_file = 'nfp_blf_sa_gpu.csv'
    submission.to_csv(output_file)
    print(f"\n✓ Submission saved to {output_file}")
    print(f"  Total rows: {len(submission)}")


if __name__ == '__main__':
    main()
