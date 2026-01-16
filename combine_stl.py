#!/usr/bin/env python3
"""
Script to combine multiple STL files into batches of 10 connected parts for JLC3DP printing.
Each batch has parts arranged in a 2x5 grid with 1.5mm thick connecting runners.
"""

import trimesh
import numpy as np
from pathlib import Path
import os

# Configuration
STL_DIR = Path("Adapter STL files")
OUTPUT_DIR = Path("Combined STL files")
RUNNER_THICKNESS = 1.5  # mm - minimum for easy break-off per JLC3DP guidelines
RUNNER_WIDTH = 2.0  # mm
PART_SPACING = 5.0  # mm gap between parts
RUNNER_Z_HEIGHT = 3.0  # mm - place runners on the main body, well above the prongs
RUNNER_Y_OFFSET = 3.0  # mm - offset runners toward the "top" edge (higher Y) to avoid center indents

# Define all 9 batches
BATCHES = {
    "Batch_01_Middle_-1": [("Middle_-1.stl", 10)],
    "Batch_02_Middle_0": [("Middle_0.stl", 10)],
    "Batch_03_Middle_0_deeper": [("Middle_0_deeper.stl", 10)],
    "Batch_04_Middle_1": [("Middle_1.stl", 10)],
    "Batch_05_Middle_2": [("Middle_2.stl", 10)],
    "Batch_06_Left_mixed": [
        ("Left_-1.stl", 4),
        ("Left_0.stl", 4),
        ("Left_1.stl", 2),
    ],
    "Batch_07_Left_Right_mixed": [
        ("Left_1.stl", 2),
        ("Left_2.stl", 4),
        ("Right_-1.stl", 4),
    ],
    "Batch_08_Right_mixed": [
        ("Right_0.stl", 4),
        ("Right_1.stl", 4),
        ("Right_2.stl", 2),
    ],
    "Batch_09_Mixed": [
        ("Right_2.stl", 2),
        ("Middle_-1.stl", 4),
        ("Middle_1.stl", 4),
    ],
}


def load_stl(filename):
    """Load an STL file and return the mesh."""
    filepath = STL_DIR / filename
    if not filepath.exists():
        raise FileNotFoundError(f"STL file not found: {filepath}")
    mesh = trimesh.load(str(filepath))
    return mesh


def find_optimal_attachment_point(mesh):
    """
    Analyze a mesh to find the optimal attachment point for runners.
    Returns (y_offset, z_height, top_surface_z) relative to the mesh center.
    
    Strategy:
    1. Find flat side faces (normals pointing ±X direction)
    2. Filter for faces above Z=2mm (on main body, not prongs)
    3. Select faces on the outer edge (high Y) to avoid center indents
    4. Find the top surface Z at the attachment Y position
    5. Return the Y and Z coordinates for attachment, plus top surface Z
    """
    normals = mesh.face_normals
    
    # Find faces with normals pointing in ±X direction (side faces)
    # We use ±X because runners typically run in the X direction between parts
    x_faces_mask = np.abs(normals[:, 0]) > 0.85  # Faces pointing mostly in X direction
    
    if not x_faces_mask.any():
        # Fallback to default values if no suitable faces found
        print("  Warning: No suitable side faces found, using defaults")
        return (RUNNER_Y_OFFSET, RUNNER_Z_HEIGHT, RUNNER_Z_HEIGHT + 2.0)
    
    # Get centroids of these faces
    x_face_indices = np.where(x_faces_mask)[0]
    x_face_verts = mesh.vertices[mesh.faces[x_face_indices]]
    x_face_centroids = x_face_verts.mean(axis=1)
    
    # Filter for faces above the prongs (Z > 2.0mm)
    z_values = x_face_centroids[:, 2]
    above_prongs_mask = z_values > 2.0
    
    if not above_prongs_mask.any():
        print("  Warning: No faces above prongs, using defaults")
        return (RUNNER_Y_OFFSET, RUNNER_Z_HEIGHT, RUNNER_Z_HEIGHT + 2.0)
    
    # Filter to faces above prongs
    valid_centroids = x_face_centroids[above_prongs_mask]
    
    # Get Y range of the mesh
    y_min, y_max = mesh.bounds[0][1], mesh.bounds[1][1]
    y_range = y_max - y_min
    y_center = (y_max + y_min) / 2
    
    # Find faces in the outer 40% of Y range (toward the edge, away from center indent)
    # We want faces with Y > y_center + 0.2 * y_range
    y_threshold = y_center + 0.2 * y_range
    outer_edge_mask = valid_centroids[:, 1] > y_threshold
    
    if not outer_edge_mask.any():
        # If no faces in outer edge, just use the highest Y faces
        outer_edge_mask = valid_centroids[:, 1] > y_center
    
    if outer_edge_mask.any():
        edge_centroids = valid_centroids[outer_edge_mask]
        # Use the median Y and Z position of these faces
        optimal_y = np.median(edge_centroids[:, 1])
        optimal_z = np.median(edge_centroids[:, 2])
        
        # Convert to offset relative to mesh center
        y_offset = optimal_y - y_center
        
        # Now find the MINIMUM top surface Z across the full Y range
        # This accounts for sloped surfaces and vertical runners between rows
        # Look for faces with normals pointing upward (+Z direction)
        z_up_faces_mask = normals[:, 2] > 0.85  # Faces pointing up
        if z_up_faces_mask.any():
            z_up_indices = np.where(z_up_faces_mask)[0]
            z_up_verts = mesh.vertices[mesh.faces[z_up_indices]]
            z_up_centroids = z_up_verts.mean(axis=1)
            
            # Filter for faces in the outer region where runners attach
            # Use faces with Y > y_center (upper half of the mesh)
            upper_half_mask = z_up_centroids[:, 1] > y_center
            if upper_half_mask.any():
                upper_centroids = z_up_centroids[upper_half_mask]
                # Use MINIMUM Z of top surface in this region (critical for sloped surfaces)
                top_surface_z = np.min(upper_centroids[:, 2])
            else:
                # No top faces in upper half, use attachment Y area
                y_near_mask = np.abs(z_up_centroids[:, 1] - optimal_y) < 2.0
                if y_near_mask.any():
                    near_centroids = z_up_centroids[y_near_mask]
                    top_surface_z = np.min(near_centroids[:, 2])
                else:
                    # Fallback to max Z
                    top_surface_z = mesh.bounds[1][2]
        else:
            # No top faces found, estimate
            top_surface_z = mesh.bounds[1][2]
        
        print(f"  Found attachment point: Y offset = {y_offset:.2f}mm, Z = {optimal_z:.2f}mm, Top surface Z = {top_surface_z:.2f}mm")
        return (y_offset, optimal_z, top_surface_z)
    else:
        print("  Warning: No suitable outer edge faces, using defaults")
        return (RUNNER_Y_OFFSET, RUNNER_Z_HEIGHT, RUNNER_Z_HEIGHT + 2.0)


def create_runner(start_pos, end_pos, z_level):
    """
    Create a rectangular runner (connecting bar) between two positions.
    The runner is placed at z_level (just above the prongs, on the main body).
    """
    # Create a box for the runner
    length = np.linalg.norm(np.array(end_pos[:2]) - np.array(start_pos[:2]))
    
    # Runner dimensions
    runner = trimesh.creation.box(
        extents=[length + RUNNER_WIDTH, RUNNER_WIDTH, RUNNER_THICKNESS]
    )
    
    # Calculate angle for rotation
    direction = np.array(end_pos[:2]) - np.array(start_pos[:2])
    angle = np.arctan2(direction[1], direction[0])
    
    # Rotate and position the runner
    rotation = trimesh.transformations.rotation_matrix(angle, [0, 0, 1])
    runner.apply_transform(rotation)
    
    # Position at midpoint between start and end, at specified z_level
    midpoint = (np.array(start_pos[:2]) + np.array(end_pos[:2])) / 2
    runner.apply_translation([midpoint[0], midpoint[1], z_level + RUNNER_THICKNESS / 2])
    
    return runner


def get_mesh_dimensions(mesh):
    """Get the bounding box dimensions of a mesh."""
    bounds = mesh.bounds
    return bounds[1] - bounds[0]  # max - min for each axis


def arrange_parts_in_grid(meshes, attachment_offsets, common_z, rows=2, cols=5):
    """
    Arrange meshes in a grid pattern, offsetting each part so its attachment
    point aligns with the fixed runner grid positions.
    
    Args:
        meshes: List of meshes to arrange
        attachment_offsets: List of (y_offset, z_height, top_surface_z) for each mesh
        common_z: Common Z height for all runners
        rows: Number of rows in grid
        cols: Number of columns in grid
    
    Returns:
        arranged: List of positioned meshes
        grid_positions: List of (x, y) grid positions where runners connect
        cell_size: (cell_width, cell_height) tuple
    """
    if len(meshes) > rows * cols:
        raise ValueError(f"Too many meshes ({len(meshes)}) for {rows}x{cols} grid")
    
    # Find the maximum dimensions across all meshes
    max_dims = np.array([0.0, 0.0, 0.0])
    for mesh in meshes:
        dims = get_mesh_dimensions(mesh)
        max_dims = np.maximum(max_dims, dims)
    
    # Cell size includes part spacing
    cell_width = max_dims[0] + PART_SPACING
    cell_height = max_dims[1] + PART_SPACING
    
    # Calculate runner top Z
    runner_top_z = common_z + RUNNER_THICKNESS / 2
    
    arranged = []
    grid_positions = []
    
    for i, mesh in enumerate(meshes):
        row = i // cols
        col = i % cols
        
        # Calculate grid cell center
        grid_x = col * cell_width
        grid_y = row * cell_height
        
        # Store the grid position (where runners will connect)
        grid_positions.append((grid_x, grid_y))
        
        # Get attachment offset for this mesh
        y_offset, z_height, top_surface_z = attachment_offsets[i]
        
        # Copy mesh and translate to position
        positioned_mesh = mesh.copy()
        
        # First center the mesh at origin
        centroid = positioned_mesh.bounds.mean(axis=0)
        positioned_mesh.apply_translation(-centroid)
        
        # Calculate Z adjustments:
        # 1. Base adjustment to align attachment point with common_z
        z_adjustment = common_z - z_height
        
        # 2. Additional adjustment if runner would protrude above top surface
        # After base adjustment, top surface will be at: top_surface_z + z_adjustment
        # Runner top is at: runner_top_z
        # If runner_top_z > (top_surface_z + z_adjustment), push part up
        adjusted_top_surface = top_surface_z + z_adjustment
        if runner_top_z > adjusted_top_surface:
            protrusion = runner_top_z - adjusted_top_surface
            z_adjustment += protrusion + 0.1  # Add 0.1mm margin
            print(f"    Part {i}: Pushing up by {protrusion:.2f}mm to clear runner")
        
        # Apply all translations
        positioned_mesh.apply_translation([grid_x, grid_y - y_offset, -positioned_mesh.bounds[0][2] + z_adjustment])
        
        arranged.append(positioned_mesh)
    
    return arranged, grid_positions, (cell_width, cell_height)


def create_connected_batch(meshes, original_meshes):
    """
    Create a single connected mesh from multiple meshes.
    Creates a fixed runner grid, then positions parts so their attachment 
    points align with the grid.
    
    Args:
        meshes: List of mesh copies to arrange
        original_meshes: List of original meshes (before positioning) for analysis
    """
    if len(meshes) == 0:
        raise ValueError("No meshes to combine")
    
    # Calculate optimal attachment points for each mesh
    print("  Analyzing meshes for optimal attachment points...")
    attachment_points = []
    for i, orig_mesh in enumerate(original_meshes):
        y_offset, z_height, top_surface_z = find_optimal_attachment_point(orig_mesh)
        attachment_points.append((y_offset, z_height, top_surface_z))
    
    # Find a common Z height for all runners (use minimum to avoid sticking out)
    z_heights = [z for _, z, _ in attachment_points]
    common_z = min(z_heights)
    print(f"  Using common runner Z height: {common_z:.2f}mm")
    
    # Arrange parts in grid, offsetting each part so its attachment aligns with grid
    arranged, grid_positions, cell_size = arrange_parts_in_grid(meshes, attachment_points, common_z)
    
    # Create runners at fixed grid positions
    runners = []
    cols = 5
    
    for i, grid_pos in enumerate(grid_positions):
        row = i // cols
        col = i % cols
        
        # Connect to right neighbor (horizontal runner)
        if col < cols - 1 and i + 1 < len(grid_positions):
            next_grid_pos = grid_positions[i + 1]
            runner = create_runner(grid_pos, next_grid_pos, common_z)
            runners.append(runner)
        
        # Connect to bottom neighbor (vertical runner)
        if row < 1 and i + cols < len(grid_positions):
            below_grid_pos = grid_positions[i + cols]
            runner = create_runner(grid_pos, below_grid_pos, common_z)
            runners.append(runner)
    
    # Combine all meshes and runners
    all_meshes = arranged + runners
    combined = trimesh.util.concatenate(all_meshes)
    
    return combined


def process_batch(batch_name, part_specs):
    """
    Process a single batch: load parts, combine, and save.
    part_specs is a list of (filename, count) tuples.
    """
    print(f"\nProcessing {batch_name}...")
    
    # Load all required meshes
    meshes = []
    original_meshes = []  # Keep original copies for analysis
    for filename, count in part_specs:
        print(f"  Loading {count}x {filename}")
        base_mesh = load_stl(filename)
        for _ in range(count):
            meshes.append(base_mesh.copy())
            original_meshes.append(base_mesh.copy())  # Store original for attachment point analysis
    
    print(f"  Total parts: {len(meshes)}")
    
    # Combine meshes using dynamic attachment points
    combined = create_connected_batch(meshes, original_meshes)
    
    # Save combined mesh
    output_path = OUTPUT_DIR / f"{batch_name}.stl"
    combined.export(str(output_path))
    print(f"  Saved: {output_path}")
    
    return combined


def main():
    """Main function to process all batches."""
    print("=" * 60)
    print("STL Combiner for JLC3DP Connected Parts")
    print("=" * 60)
    
    # Create output directory
    OUTPUT_DIR.mkdir(exist_ok=True)
    print(f"\nOutput directory: {OUTPUT_DIR}")
    
    # Verify all source files exist
    print("\nVerifying source files...")
    all_files = set()
    for part_specs in BATCHES.values():
        for filename, _ in part_specs:
            all_files.add(filename)
    
    missing = []
    for filename in all_files:
        if not (STL_DIR / filename).exists():
            missing.append(filename)
    
    if missing:
        print("ERROR: Missing STL files:")
        for f in missing:
            print(f"  - {f}")
        return 1
    
    print(f"  All {len(all_files)} source files found!")
    
    # Process each batch
    total_parts = 0
    for batch_name, part_specs in BATCHES.items():
        process_batch(batch_name, part_specs)
        total_parts += sum(count for _, count in part_specs)
    
    print("\n" + "=" * 60)
    print(f"COMPLETE! Generated {len(BATCHES)} combined STL files")
    print(f"Total parts: {total_parts}")
    print(f"Output location: {OUTPUT_DIR}/")
    print("=" * 60)
    
    return 0


if __name__ == "__main__":
    exit(main())
