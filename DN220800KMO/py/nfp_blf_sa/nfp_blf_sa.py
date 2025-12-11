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

import pandas as pd
from shapely import affinity
from shapely.geometry import Polygon
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


def check_collision(poly1, poly2):
    """
    Check if two polygons overlap (not just touch).
    Returns True if there's an overlap, False otherwise.
    
    Uses the same proven logic as the baseline greedy approach.
    """
    return poly1.intersects(poly2) and not poly1.touches(poly2)


def compute_bounding_box_with_new_tree(placed_trees, new_tree_poly):
    """
    Compute the bounding box side length if we add a new tree polygon.
    Returns the max side length of the bounding box.
    """
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


def find_best_rotation_and_position(placed_trees, num_rotation_angles=24, num_direction_angles=48):
    """
    Try multiple rotation angles for the new tree and find the best 
    (rotation, position) combination that minimizes the bounding box.
    
    Returns (best_x, best_y, best_angle) in unscaled Decimal coordinates.
    """
    if not placed_trees:
        return Decimal('0'), Decimal('0'), Decimal('0')
    
    placed_polygons = [t.polygon for t in placed_trees]
    tree_index = STRtree(placed_polygons)
    
    best_score = Decimal('Infinity')
    best_result = (Decimal('0'), Decimal('0'), Decimal('0'))
    
    # Try different rotation angles
    # Focus on angles that help interlocking: 0, 30, 60, 90, 120, 150, 180, etc.
    rotation_angles = [i * 360 / num_rotation_angles for i in range(num_rotation_angles)]
    
    for rot_angle in rotation_angles:
        # Create template tree with this rotation at origin
        template_tree = ChristmasTree(angle=str(rot_angle))
        
        # Generate candidate positions for this rotation
        candidate_positions = []
        direction_angles = [2 * math.pi * i / num_direction_angles for i in range(num_direction_angles)]
        
        for dir_angle in direction_angles:
            vx = Decimal(str(math.cos(dir_angle)))
            vy = Decimal(str(math.sin(dir_angle)))
            
            # Binary search for contact position
            low_radius = Decimal('0.0')
            high_radius = Decimal('12.0')
            
            # Check if high_radius is collision-free
            px = high_radius * vx
            py = high_radius * vy
            candidate_poly = affinity.translate(
                template_tree.polygon,
                xoff=float(px * scale_factor),
                yoff=float(py * scale_factor)
            )
            possible_indices = tree_index.query(candidate_poly)
            has_collision = any(
                candidate_poly.intersects(placed_polygons[i]) and 
                not candidate_poly.touches(placed_polygons[i])
                for i in possible_indices
            )
            
            if has_collision:
                continue
            
            # Binary search
            for _ in range(18):
                mid_radius = (low_radius + high_radius) / 2
                px = mid_radius * vx
                py = mid_radius * vy
                
                candidate_poly = affinity.translate(
                    template_tree.polygon,
                    xoff=float(px * scale_factor),
                    yoff=float(py * scale_factor)
                )
                
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
        
        # Evaluate each candidate position
        for px, py in candidate_positions:
            # Create the tree at this position
            new_tree_poly = affinity.translate(
                template_tree.polygon,
                xoff=float(px * scale_factor),
                yoff=float(py * scale_factor)
            )
            
            # Score = bounding box side length
            score = compute_bounding_box_with_new_tree(placed_trees, new_tree_poly)
            
            if score < best_score:
                best_score = score
                best_result = (px, py, Decimal(str(rot_angle)))
    
    return best_result


def copy_tree(tree):
    """Create a fresh copy of a tree with properly constructed polygon."""
    return ChristmasTree(
        center_x=str(tree.center_x),
        center_y=str(tree.center_y),
        angle=str(tree.angle)
    )


def validate_and_fix_overlaps(placed_trees, max_iterations=5):
    """
    Validate that no trees overlap and fix any overlaps found.
    Returns the fixed list of trees.
    Uses minimal movement to fix overlaps.
    """
    if len(placed_trees) < 2:
        return placed_trees
    
    # Work with copies
    fixed_trees = [copy_tree(t) for t in placed_trees]
    
    for iteration in range(max_iterations):
        has_overlap = False
        overlap_pairs = []
        
        # Check all pairs for overlaps
        for i in range(len(fixed_trees)):
            for j in range(i + 1, len(fixed_trees)):
                if check_collision(fixed_trees[i].polygon, fixed_trees[j].polygon):
                    overlap_pairs.append((i, j))
                    has_overlap = True
        
        if not has_overlap:
            break
        
        # Fix overlaps by moving the later tree away
        for i, j in overlap_pairs:
            tree1 = fixed_trees[j]  # Move the later-placed tree
            tree2 = fixed_trees[i]
            
            # Calculate direction vector from tree2 to tree1
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
                # Trees at same position - move in arbitrary direction
                new_x = tree1.center_x + move_distance
                new_y = tree1.center_y + move_distance
            
            # Recreate tree at new position
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
    side_length = max(width, height)
    
    return side_length


def simulated_annealing_improvement(placed_trees, initial_temp=50.0, cooling_rate=0.95, 
                                   iterations=500, num_trees_to_optimize=None):
    """
    Simulated Annealing improvement phase to refine the layout.
    Uses small perturbations to escape local optima and find more compact packings.
    
    Enhanced with:
    - Larger rotation perturbations to find better interlocking
    - Move toward center bias
    - Occasional big jumps to escape local minima
    """
    if len(placed_trees) < 2:
        return placed_trees
    
    # Work with copies
    optimized_trees = [copy_tree(t) for t in placed_trees]
    
    current_side = compute_side_length(optimized_trees)
    best_side = current_side
    best_trees = [copy_tree(t) for t in optimized_trees]
    
    temperature = float(initial_temp)
    
    # Determine which trees to optimize (all trees for small sets, recent for large)
    if num_trees_to_optimize is None:
        num_trees_to_optimize = min(len(optimized_trees), 30)
    
    tree_indices = list(range(max(0, len(optimized_trees) - num_trees_to_optimize), 
                              len(optimized_trees)))
    
    if not tree_indices:
        return placed_trees
    
    for iteration in range(iterations):
        # Select a random tree to perturb
        tree_idx = random.choice(tree_indices)
        old_tree = optimized_trees[tree_idx]
        
        # Save old state
        old_x = old_tree.center_x
        old_y = old_tree.center_y
        old_angle = old_tree.angle
        
        # Decrease perturbation magnitude as temperature cools
        progress = iteration / iterations
        
        # Choose perturbation type with rotation bias
        rand = random.random()
        if rand < 0.3:
            perturbation_type = 'translate'
        elif rand < 0.6:
            perturbation_type = 'rotate'
        elif rand < 0.85:
            perturbation_type = 'both'
        else:
            perturbation_type = 'big_rotate'  # Occasional large rotation
        
        new_x = old_x
        new_y = old_y
        new_angle = old_angle
        
        if perturbation_type in ['translate', 'both']:
            max_shift = 0.25 * (1.0 - progress * 0.6)
            # Bias toward center
            if random.random() < 0.4:
                # Move slightly toward center
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
            # Try a major rotation (45, 90, 135, 180 degrees etc.)
            big_angles = [45, 90, 135, 180, 225, 270, 315]
            new_angle = (old_angle + Decimal(str(random.choice(big_angles)))) % Decimal('360')
        
        # Create new tree with perturbed position/angle
        new_tree = ChristmasTree(
            center_x=str(new_x),
            center_y=str(new_y),
            angle=str(new_angle)
        )
        
        # Check for collisions with other trees
        has_collision = False
        for i, other_tree in enumerate(optimized_trees):
            if i != tree_idx:
                if check_collision(new_tree.polygon, other_tree.polygon):
                    has_collision = True
                    break
        
        if has_collision:
            # Reject move - keep old tree
            continue
        
        # Temporarily replace tree to compute new side length
        optimized_trees[tree_idx] = new_tree
        new_side = compute_side_length(optimized_trees)
        delta = float(new_side - current_side)
        
        # Accept or reject based on SA criteria
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
            # Reject: restore old tree
            optimized_trees[tree_idx] = old_tree
        
        # Cool down
        temperature *= cooling_rate
    
    return best_trees


def initialize_trees(num_trees, existing_trees=None, use_sa=True, sa_iterations=400):
    """
    Hybrid Pipeline: Rotation-optimized BLF Construction + Simulated Annealing Improvement
    
    Phase 1 (Construction): For each tree, tries multiple rotation angles and finds
    the best (rotation, position) combination that minimizes the bounding box.
    
    Phase 2 (Improvement): Uses Simulated Annealing to refine the layout and escape
    local optima.
    """
    if num_trees == 0:
        return [], Decimal('0')

    if existing_trees is None:
        placed_trees = []
    else:
        # Create copies to avoid modifying original
        placed_trees = [copy_tree(t) for t in existing_trees]

    num_to_add = num_trees - len(placed_trees)

    if num_to_add > 0:
        # Phase 1: Construction with rotation optimization
        
        if not placed_trees:
            # First tree: try a few rotations to start with best orientation
            # For first tree alone, rotation doesn't matter much, place at origin
            placed_trees.append(ChristmasTree(angle='0'))
            num_to_add -= 1
        
        # Second tree: try interlocking configurations (tips toward each other)
        if num_to_add > 0 and len(placed_trees) == 1:
            best_score = Decimal('Infinity')
            best_tree = None
            
            # Try specific angles that encourage interlocking
            for angle in [0, 45, 90, 135, 180, 225, 270, 315]:
                template = ChristmasTree(angle=str(angle))
                # Try positions close to the first tree
                for px_offset in [Decimal('0.6'), Decimal('0.7'), Decimal('0.8')]:
                    for py_offset in [Decimal('-0.3'), Decimal('0'), Decimal('0.3')]:
                        test_tree = ChristmasTree(
                            center_x=str(px_offset),
                            center_y=str(py_offset),
                            angle=str(angle)
                        )
                        if not check_collision(test_tree.polygon, placed_trees[0].polygon):
                            score = compute_bounding_box_with_new_tree(placed_trees, test_tree.polygon)
                            if score < best_score:
                                best_score = score
                                best_tree = test_tree
            
            if best_tree:
                placed_trees.append(best_tree)
            else:
                # Fallback
                px, py, angle = find_best_rotation_and_position(placed_trees, 24, 48)
                placed_trees.append(ChristmasTree(center_x=str(px), center_y=str(py), angle=str(angle)))
            num_to_add -= 1
        
        # Remaining trees: use full rotation optimization
        for _ in range(num_to_add):
            # Try many rotation angles and find the best placement
            px, py, angle = find_best_rotation_and_position(
                placed_trees,
                num_rotation_angles=36,  # Every 10 degrees
                num_direction_angles=72  # Every 5 degrees
            )
            
            placed_tree = ChristmasTree(
                center_x=str(px),
                center_y=str(py),
                angle=str(angle)
            )
            placed_trees.append(placed_tree)
        
        # Validate at the end to fix any overlaps
        placed_trees = validate_and_fix_overlaps(placed_trees, max_iterations=3)

    # Phase 2: Simulated Annealing Improvement (only for non-trivial cases)
    if use_sa and len(placed_trees) > 2:
        placed_trees = simulated_annealing_improvement(
            placed_trees,
            initial_temp=40.0,
            cooling_rate=0.97,
            iterations=sa_iterations,
            num_trees_to_optimize=min(len(placed_trees), 25)
        )
        
        # Final validation
        placed_trees = validate_and_fix_overlaps(placed_trees, max_iterations=2)

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

