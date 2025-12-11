"""
Santa 2025 - Christmas Tree Packing Challenge
Hybrid Pipeline: NFP-guided BLF + Simulated Annealing

Algorithms:
- NFP (No-Fit Polygon): Computes contact points for optimal placement
- BLF (Bottom-Left-Fill): Heuristic placement strategy
- SA (Simulated Annealing): Global optimization improvement phase
"""

import math
import random
from decimal import Decimal, getcontext

import numpy as np
import pandas as pd
from shapely import affinity
from shapely.geometry import Point, Polygon, MultiPolygon
from shapely.ops import unary_union
from shapely.strtree import STRtree

# Set precision for Decimal
getcontext().prec = 25
scale_factor = Decimal('1e15')

# Build the index of the submission, in the format: <trees_in_problem>_<tree_index>
index = [f'{n:03d}_{t}' for n in range(1, 201) for t in range(n)]


class ChristmasTree:
    """Represents a single, rotatable Christmas tree of a fixed size."""

    def __init__(self, center_x='0', center_y='0', angle='0'):
        """Initializes the Christmas tree with a specific position and rotation."""
        self.center_x = Decimal(center_x)
        self.center_y = Decimal(center_y)
        self.angle = Decimal(angle)

        trunk_w = Decimal('0.15')
        trunk_h = Decimal('0.2')
        base_w = Decimal('0.7')
        mid_w = Decimal('0.4')
        top_w = Decimal('0.25')
        tip_y = Decimal('0.8')
        tier_1_y = Decimal('0.5')
        tier_2_y = Decimal('0.25')
        base_y = Decimal('0.0')
        trunk_bottom_y = -trunk_h

        initial_polygon = Polygon(
            [
                # Start at Tip
                (Decimal('0.0') * scale_factor, tip_y * scale_factor),
                # Right side - Top Tier
                (top_w / Decimal('2') * scale_factor, tier_1_y * scale_factor),
                (top_w / Decimal('4') * scale_factor, tier_1_y * scale_factor),
                # Right side - Middle Tier
                (mid_w / Decimal('2') * scale_factor, tier_2_y * scale_factor),
                (mid_w / Decimal('4') * scale_factor, tier_2_y * scale_factor),
                # Right side - Bottom Tier
                (base_w / Decimal('2') * scale_factor, base_y * scale_factor),
                # Right Trunk
                (trunk_w / Decimal('2') * scale_factor, base_y * scale_factor),
                (trunk_w / Decimal('2') * scale_factor, trunk_bottom_y * scale_factor),
                # Left Trunk
                (-(trunk_w / Decimal('2')) * scale_factor, trunk_bottom_y * scale_factor),
                (-(trunk_w / Decimal('2')) * scale_factor, base_y * scale_factor),
                # Left side - Bottom Tier
                (-(base_w / Decimal('2')) * scale_factor, base_y * scale_factor),
                # Left side - Middle Tier
                (-(mid_w / Decimal('4')) * scale_factor, tier_2_y * scale_factor),
                (-(mid_w / Decimal('2')) * scale_factor, tier_2_y * scale_factor),
                # Left side - Top Tier
                (-(top_w / Decimal('4')) * scale_factor, tier_1_y * scale_factor),
                (-(top_w / Decimal('2')) * scale_factor, tier_1_y * scale_factor),
            ]
        )
        rotated = affinity.rotate(initial_polygon, float(self.angle), origin=(0, 0))
        self.polygon = affinity.translate(
            rotated,
            xoff=float(self.center_x * scale_factor),
            yoff=float(self.center_y * scale_factor)
        )


def generate_weighted_angle():
    """
    Generates a random angle with a distribution weighted by abs(sin(2*angle)).
    This helps place more trees in corners, and makes the packing less round.
    """
    while True:
        angle = random.uniform(0, 2 * math.pi)
        if random.uniform(0, 1) < abs(math.sin(2 * angle)):
            return angle


def compute_nfp_contact_points(polygon_a, polygon_b, num_points=200):
    """
    Compute NFP-derived contact points by sliding polygon_b around polygon_a.
    Returns a list of (x, y) positions where polygon_b can be placed touching polygon_a.
    
    This approximates the No-Fit Polygon by sampling contact points.
    """
    contact_points = []
    
    # Handle MultiPolygon by extracting the largest polygon or all polygons
    if isinstance(polygon_a, MultiPolygon):
        # Use the largest polygon by area, or combine all exteriors
        polygons_list = list(polygon_a.geoms)
        if polygons_list:
            # Get the largest polygon
            polygon_a = max(polygons_list, key=lambda p: p.area)
        else:
            # Fallback: use first polygon
            polygon_a = polygons_list[0] if polygons_list else polygon_a
    
    # Get vertices and edges of polygon_a
    if hasattr(polygon_a, 'exterior'):
        coords_a = list(polygon_a.exterior.coords[:-1])  # Exclude duplicate last point
    else:
        # For other geometry types, try to get coordinates
        try:
            coords_a = list(polygon_a.coords)
        except (NotImplementedError, AttributeError):
            # Fallback: use boundary
            coords_a = list(polygon_a.boundary.coords[:-1]) if hasattr(polygon_a, 'boundary') else []
    
    # Handle MultiPolygon for polygon_b as well
    if isinstance(polygon_b, MultiPolygon):
        polygons_list = list(polygon_b.geoms)
        if polygons_list:
            polygon_b = max(polygons_list, key=lambda p: p.area)
        else:
            polygon_b = polygons_list[0] if polygons_list else polygon_b
    
    # Get vertices of polygon_b (centered at origin)
    if hasattr(polygon_b, 'exterior'):
        coords_b = list(polygon_b.exterior.coords[:-1])
    else:
        try:
            coords_b = list(polygon_b.coords)
        except (NotImplementedError, AttributeError):
            coords_b = list(polygon_b.boundary.coords[:-1]) if hasattr(polygon_b, 'boundary') else []
    
    # Compute bounding box for search area
    bounds_a = polygon_a.bounds
    margin = max(bounds_a[2] - bounds_a[0], bounds_a[3] - bounds_a[1]) * 0.5
    
    # Strategy 1: Slide polygon_b vertices along polygon_a edges
    for i, (vx_a, vy_a) in enumerate(coords_a):
        next_i = (i + 1) % len(coords_a)
        vx_a_next, vy_a_next = coords_a[next_i]
        
        # Edge direction
        edge_dx = vx_a_next - vx_a
        edge_dy = vy_a_next - vy_a
        edge_len = math.sqrt(edge_dx**2 + edge_dy**2)
        
        if edge_len > 0:
            # Sample points along this edge
            samples = min(20, max(5, int(edge_len / (margin / 50))))
            for j in range(samples):
                t = j / max(1, samples - 1)
                px = vx_a + t * edge_dx
                py = vy_a + t * edge_dy
                
                # Try placing polygon_b at various angles relative to this point
                for k, (vx_b, vy_b) in enumerate(coords_b):
                    # Offset to make polygon_b touch polygon_a
                    offset_x = px - vx_b
                    offset_y = py - vy_b
                    
                    # Test if this is a valid contact point
                    test_poly = affinity.translate(
                        polygon_b,
                        xoff=offset_x,
                        yoff=offset_y
                    )
                    
                    if polygon_a.touches(test_poly) or polygon_a.distance(test_poly) < 1e-6:
                        contact_points.append((offset_x, offset_y))
    
    # Strategy 2: Sample around polygon_a boundary
    if len(contact_points) < num_points:
        # Generate additional points by sampling angles
        for angle in np.linspace(0, 2 * math.pi, num_points // 4):
            # Find the farthest point in this direction from polygon_a center
            cx = (bounds_a[0] + bounds_a[2]) / 2
            cy = (bounds_a[1] + bounds_a[3]) / 2
            
            # Sample distances
            for dist_mult in np.linspace(0.5, 2.0, 5):
                test_x = cx + dist_mult * margin * math.cos(angle)
                test_y = cy + dist_mult * margin * math.sin(angle)
                
                test_poly = affinity.translate(
                    polygon_b,
                    xoff=test_x,
                    yoff=test_y
                )
                
                # Slide towards polygon_a until contact
                step = margin / 100
                for _ in range(50):
                    if polygon_a.touches(test_poly) or polygon_a.distance(test_poly) < 1e-6:
                        # Extract center position
                        test_bounds = test_poly.bounds
                        center_x = (test_bounds[0] + test_bounds[2]) / 2
                        center_y = (test_bounds[1] + test_bounds[3]) / 2
                        contact_points.append((center_x, center_y))
                        break
                    
                    # Move towards polygon_a
                    dir_x = cx - test_x
                    dir_y = cy - test_y
                    dir_len = math.sqrt(dir_x**2 + dir_y**2)
                    if dir_len > 0:
                        test_x += step * dir_x / dir_len
                        test_y += step * dir_y / dir_len
                        test_poly = affinity.translate(
                            polygon_b,
                            xoff=test_x,
                            yoff=test_y
                        )
                    else:
                        break
    
    # Strategy 3: Use polygon vertices as contact points
    for vx_a, vy_a in coords_a:
        for vx_b, vy_b in coords_b:
            offset_x = vx_a - vx_b
            offset_y = vy_a - vy_b
            contact_points.append((offset_x, offset_y))
    
    # Remove duplicates and limit to num_points
    unique_points = []
    seen = set()
    for x, y in contact_points:
        key = (round(x, 6), round(y, 6))
        if key not in seen:
            seen.add(key)
            unique_points.append((x, y))
            if len(unique_points) >= num_points:
                break
    
    return unique_points[:num_points]


def check_collision(poly1, poly2, tolerance=1e-8):
    """
    Check if two polygons overlap (not just touch).
    Returns True if there's an overlap, False otherwise.
    """
    if not poly1.intersects(poly2):
        return False
    
    # Check if they only touch (no area overlap)
    if poly1.touches(poly2):
        return False
    
    # Check for actual area overlap
    try:
        intersection = poly1.intersection(poly2)
        if intersection.is_empty:
            return False
        
        # Get area of intersection (works for Polygon, MultiPolygon, etc.)
        if hasattr(intersection, 'area'):
            overlap_area = intersection.area
            # Consider it a collision if overlap area is above tolerance
            return overlap_area > tolerance
        elif hasattr(intersection, 'geoms'):
            # MultiPolygon case - sum areas
            total_area = sum(geom.area for geom in intersection.geoms if hasattr(geom, 'area'))
            return total_area > tolerance
        
        # Fallback: if we can't compute area but they intersect and don't touch, assume collision
        return True
    except Exception:
        # If intersection computation fails, be conservative and assume collision
        return True


def blf_placement_with_nfp(tree_to_place, placed_trees, num_contact_points=200):
    """
    Bottom-Left-Fill heuristic using NFP-derived contact points.
    Evaluates approximately num_contact_points positions per tree.
    """
    if not placed_trees:
        return Decimal('0'), Decimal('0')
    
    placed_polygons = [t.polygon for t in placed_trees]
    tree_index = STRtree(placed_polygons)
    
    # Get the union of all placed polygons for NFP computation
    if len(placed_polygons) == 1:
        reference_polygon = placed_polygons[0]
    else:
        reference_polygon = unary_union(placed_polygons)
    
    # Compute NFP contact points
    contact_points = compute_nfp_contact_points(
        reference_polygon,
        tree_to_place.polygon,
        num_points=num_contact_points
    )
    
    best_position = None
    best_score = Decimal('Infinity')
    
    # Evaluate each contact point using BLF criteria (minimize x+y, prefer lower-left)
    for contact_x, contact_y in contact_points:
        # Contact points are offsets in scaled coordinates
        # Convert to Decimal for computation
        px_scaled = Decimal(str(contact_x))
        py_scaled = Decimal(str(contact_y))
        
        # Translate tree to this position (using scaled coordinates)
        candidate_poly = affinity.translate(
            tree_to_place.polygon,
            xoff=float(px_scaled),
            yoff=float(py_scaled)
        )
        
        # Check for collisions using improved collision detection
        possible_indices = tree_index.query(candidate_poly)
        has_collision = any(
            check_collision(candidate_poly, placed_polygons[i])
            for i in possible_indices
        )
        
        if not has_collision:
            # BLF score: minimize x+y (bottom-left preference)
            # Convert to unscaled coordinates for scoring and storage
            px_unscaled = px_scaled / scale_factor
            py_unscaled = py_scaled / scale_factor
            score = px_unscaled + py_unscaled + Decimal('0.1') * (px_unscaled**2 + py_unscaled**2).sqrt()
            
            if score < best_score:
                best_score = score
                best_position = (px_unscaled, py_unscaled)
    
    # If no valid contact point found, fall back to sliding approach
    if best_position is None:
        # Try sliding from various angles
        for _ in range(50):
            angle = generate_weighted_angle()
            vx = Decimal(str(math.cos(angle)))
            vy = Decimal(str(math.sin(angle)))
            
            radius = Decimal('20.0')
            step_in = Decimal('0.5')
            
            collision_found = False
            while radius >= 0:
                px = radius * vx
                py = radius * vy
                
                candidate_poly = affinity.translate(
                    tree_to_place.polygon,
                    xoff=float(px * scale_factor),
                    yoff=float(py * scale_factor)
                )
                
                possible_indices = tree_index.query(candidate_poly)
                if any(check_collision(candidate_poly, placed_polygons[i])
                       for i in possible_indices):
                    collision_found = True
                    break
                radius -= step_in
            
            if collision_found:
                step_out = Decimal('0.05')
                while True:
                    radius += step_out
                    px = radius * vx
                    py = radius * vy
                    
                    candidate_poly = affinity.translate(
                        tree_to_place.polygon,
                        xoff=float(px * scale_factor),
                        yoff=float(py * scale_factor)
                    )
                    
                    possible_indices = tree_index.query(candidate_poly)
                    if not any(check_collision(candidate_poly, placed_polygons[i])
                               for i in possible_indices):
                        score = px + py + Decimal('0.1') * (px**2 + py**2).sqrt()
                        if score < best_score:
                            best_score = score
                            best_position = (px, py)
                        break
    
    if best_position is None:
        # Last resort: place at origin
        return Decimal('0'), Decimal('0')
    
    return best_position[0], best_position[1]


def validate_and_fix_overlaps(placed_trees, max_iterations=10):
    """
    Validate that no trees overlap and fix any overlaps found.
    Returns the fixed list of trees.
    Uses minimal movement to fix overlaps without expanding the bounding box unnecessarily.
    """
    if len(placed_trees) < 2:
        return placed_trees
    
    fixed_trees = []
    for tree in placed_trees:
        new_tree = ChristmasTree(
            center_x=str(tree.center_x),
            center_y=str(tree.center_y),
            angle=str(tree.angle)
        )
        fixed_trees.append(new_tree)
    
    # Check for overlaps and fix them with minimal movement
    for iteration in range(max_iterations):
        has_overlap = False
        placed_polygons = [t.polygon for t in fixed_trees]
        tree_index = STRtree(placed_polygons)
        
        # Check all pairs for overlaps
        overlap_pairs = []
        for i, tree1 in enumerate(fixed_trees):
            possible_indices = tree_index.query(tree1.polygon)
            for j in possible_indices:
                if i < j and check_collision(tree1.polygon, placed_polygons[j]):
                    overlap_pairs.append((i, j))
                    has_overlap = True
        
        if not has_overlap:
            break
        
        # Fix overlaps by moving trees apart minimally
        for i, j in overlap_pairs:
            tree1 = fixed_trees[i]
            tree2 = fixed_trees[j]
            
            # Calculate direction vector from tree2 to tree1
            dx = tree1.center_x - tree2.center_x
            dy = tree1.center_y - tree2.center_y
            dist = (dx**2 + dy**2).sqrt()
            
            # Use smaller, adaptive move distance that decreases with iterations
            base_move = Decimal('0.05') * (Decimal('1.0') - Decimal(str(iteration)) / Decimal(str(max_iterations)))
            move_distance = max(base_move, Decimal('0.01'))
            
            if dist > Decimal('0.001'):
                # Normalize direction
                unit_dx = dx / dist
                unit_dy = dy / dist
                # Only move the later-placed tree (tree1) to minimize disruption
                tree1.center_x += move_distance * unit_dx
                tree1.center_y += move_distance * unit_dy
            else:
                # If trees are at same position, move them apart minimally
                tree1.center_x += move_distance
                tree1.center_y += move_distance
            
            # Update polygon
            base_tree1 = ChristmasTree(angle=str(tree1.angle))
            tree1.polygon = affinity.translate(
                base_tree1.polygon,
                xoff=float(tree1.center_x * scale_factor),
                yoff=float(tree1.center_y * scale_factor)
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
    side_length = max(width, height)
    
    return side_length


def simulated_annealing_improvement(placed_trees, initial_temp=100.0, cooling_rate=0.95, 
                                   iterations=1000, num_trees_to_optimize=None):
    """
    Simulated Annealing improvement phase to refine the layout.
    """
    if len(placed_trees) < 2:
        return placed_trees
    
    # Work with a copy
    optimized_trees = []
    for tree in placed_trees:
        new_tree = ChristmasTree(
            center_x=str(tree.center_x),
            center_y=str(tree.center_y),
            angle=str(tree.angle)
        )
        optimized_trees.append(new_tree)
    
    current_side = compute_side_length(optimized_trees)
    best_side = current_side
    best_trees = [ChristmasTree(
        center_x=str(t.center_x),
        center_y=str(t.center_y),
        angle=str(t.angle)
    ) for t in optimized_trees]
    
    temperature = Decimal(str(initial_temp))
    
    # Determine which trees to optimize (focus on recent additions)
    if num_trees_to_optimize is None:
        num_trees_to_optimize = min(len(optimized_trees), 50)
    
    tree_indices = list(range(max(0, len(optimized_trees) - num_trees_to_optimize), 
                              len(optimized_trees)))
    
    for iteration in range(iterations):
        # Select a random tree to perturb
        if not tree_indices:
            break
        
        tree_idx = random.choice(tree_indices)
        tree = optimized_trees[tree_idx]
        
        # Generate a perturbation
        perturbation_type = random.choice(['translate', 'rotate', 'both'])
        
        old_x = tree.center_x
        old_y = tree.center_y
        old_angle = tree.angle
        
        if perturbation_type in ['translate', 'both']:
            # Small random translation
            max_shift = Decimal('0.5') * (Decimal('1.0') - Decimal(str(iteration / iterations)))
            dx = Decimal(str(random.uniform(-float(max_shift), float(max_shift))))
            dy = Decimal(str(random.uniform(-float(max_shift), float(max_shift))))
            tree.center_x += dx
            tree.center_y += dy
        
        if perturbation_type in ['rotate', 'both']:
            # Small random rotation
            max_rotation = 15.0 * (1.0 - iteration / iterations)
            dangle = Decimal(str(random.uniform(-max_rotation, max_rotation)))
            tree.angle = (tree.angle + dangle) % Decimal('360')
        
        # Recreate polygon with updated position and angle
        base_tree = ChristmasTree(angle=str(tree.angle))
        tree.polygon = affinity.translate(
            base_tree.polygon,
            xoff=float(tree.center_x * scale_factor),
            yoff=float(tree.center_y * scale_factor)
        )
        
        # Check for collisions
        placed_polygons = [t.polygon for t in optimized_trees]
        tree_index = STRtree(placed_polygons)
        
        has_collision = False
        for i, other_tree in enumerate(optimized_trees):
            if i != tree_idx:
                if check_collision(tree.polygon, other_tree.polygon):
                    has_collision = True
                    break
        
        if has_collision:
            # Reject: restore old position
            tree.center_x = old_x
            tree.center_y = old_y
            tree.angle = old_angle
            base_tree = ChristmasTree(angle=str(tree.angle))
            tree.polygon = affinity.translate(
                base_tree.polygon,
                xoff=float(tree.center_x * scale_factor),
                yoff=float(tree.center_y * scale_factor)
            )
        else:
            # Evaluate new side length
            new_side = compute_side_length(optimized_trees)
            delta = new_side - current_side
            
            # Accept or reject based on SA criteria
            if delta < 0 or (temperature > 0 and 
                           random.random() < math.exp(-float(delta) / float(temperature))):
                current_side = new_side
                if current_side < best_side:
                    best_side = current_side
                    best_trees = [ChristmasTree(
                        center_x=str(t.center_x),
                        center_y=str(t.center_y),
                        angle=str(t.angle)
                    ) for t in optimized_trees]
            else:
                # Reject: restore old position
                tree.center_x = old_x
                tree.center_y = old_y
                tree.angle = old_angle
                base_tree = ChristmasTree(angle=str(tree.angle))
                tree.polygon = affinity.translate(
                    base_tree.polygon,
                    xoff=float(tree.center_x * scale_factor),
                    yoff=float(tree.center_y * scale_factor)
                )
        
        # Cool down
        temperature *= Decimal(str(cooling_rate))
        if temperature < Decimal('0.01'):
            temperature = Decimal('0.01')
    
    return best_trees


def initialize_trees(num_trees, existing_trees=None, use_sa=True, sa_iterations=500):
    """
    Hybrid Pipeline: NFP-guided BLF Construction + Simulated Annealing Improvement
    
    Phase 1 (Construction): Uses NFP-guided Bottom-Left-Fill heuristic with ~200
    contact points per tree to generate a valid starting layout.
    
    Phase 2 (Improvement): Uses Simulated Annealing to refine the layout and escape
    local optima.
    """
    if num_trees == 0:
        return [], Decimal('0')

    if existing_trees is None:
        placed_trees = []
    else:
        # Create copies to avoid modifying original
        placed_trees = []
        for tree in existing_trees:
            new_tree = ChristmasTree(
                center_x=str(tree.center_x),
                center_y=str(tree.center_y),
                angle=str(tree.angle)
            )
            placed_trees.append(new_tree)

    num_to_add = num_trees - len(placed_trees)

    if num_to_add > 0:
        # Phase 1: Construction using NFP-guided BLF
        unplaced_trees = [
            ChristmasTree(angle=random.uniform(0, 360)) for _ in range(num_to_add)
        ]
        
        if not placed_trees:
            # Place first tree at origin
            placed_trees.append(unplaced_trees.pop(0))

        for tree_to_place in unplaced_trees:
            # Use BLF with NFP contact points (~200 per tree)
            px, py = blf_placement_with_nfp(
                tree_to_place,
                placed_trees,
                num_contact_points=200
            )
            
            tree_to_place.center_x = px
            tree_to_place.center_y = py
            
            # Update polygon position
            base_tree = ChristmasTree(angle=str(tree_to_place.angle))
            tree_to_place.polygon = affinity.translate(
                base_tree.polygon,
                xoff=float(tree_to_place.center_x * scale_factor),
                yoff=float(tree_to_place.center_y * scale_factor)
            )
            
            placed_trees.append(tree_to_place)
        
        # Only validate occasionally to avoid excessive movement
        # Validate every 10 trees or at the end
        if len(placed_trees) % 10 == 0 or len(placed_trees) == num_trees:
            placed_trees = validate_and_fix_overlaps(placed_trees, max_iterations=5)

    # Phase 2: Simulated Annealing Improvement
    if use_sa and len(placed_trees) > 1:
        # Apply SA improvement, focusing on recently placed trees
        placed_trees = simulated_annealing_improvement(
            placed_trees,
            initial_temp=50.0,
            cooling_rate=0.95,
            iterations=sa_iterations,
            num_trees_to_optimize=min(len(placed_trees), 30)
        )
        
        # Only validate once after SA, with minimal iterations
        placed_trees = validate_and_fix_overlaps(placed_trees, max_iterations=3)

    side_length = compute_side_length(placed_trees)
    return placed_trees, side_length


def main():
    """Main function to generate submission file."""
    print("Starting tree packing solution with NFP-BLF-SA hybrid pipeline...")
    
    tree_data = []
    current_placed_trees = []  # Initialize an empty list for the first iteration

    for n in range(200):
        # Pass the current_placed_trees to initialize_trees
        current_placed_trees, side = initialize_trees(n+1, existing_trees=current_placed_trees)
        
        if (n+1) % 10 == 0:
            print(f"Processed {n+1} trees, side length: {side:.12f}")
        
        for tree in current_placed_trees:
            tree_data.append([tree.center_x, tree.center_y, tree.angle])

    print("Creating submission DataFrame...")
    cols = ['x', 'y', 'deg']
    submission = pd.DataFrame(
        index=index, columns=cols, data=tree_data
    ).rename_axis('id')

    for col in cols:
        submission[col] = submission[col].astype(float).round(decimals=6)

    # To ensure everything is kept as a string, prepend an 's'
    for col in submission.columns:
        submission[col] = 's' + submission[col].astype('string')
    
    output_file = 'sample_submission.csv'
    submission.to_csv(output_file)
    print(f"Submission saved to {output_file}")
    print(f"Total rows: {len(submission)}")


if __name__ == '__main__':
    main()

