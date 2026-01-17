#!/usr/bin/env python3
"""
Script to combine multiple STL files into batches of 10 connected parts for JLC3DP printing.
Each batch has parts arranged in a 2x5 grid with 1.5mm thick connecting runners.
"""

import trimesh
import numpy as np
from pathlib import Path

# Configuration
STL_DIR = Path("Adapter STL files")
OUTPUT_DIR = Path("Combined STL files")
RUNNER_THICKNESS = 1.5  # mm - minimum for easy break-off per JLC3DP guidelines
RUNNER_WIDTH = 2.0  # mm
PART_SPACING = 5.0  # mm gap between parts
RUNNER_Z_HEIGHT = 3.0  # mm - place runners on the main body, well above the prongs

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


def find_attachment_point_local(mesh):
    """
    Find the attachment point for runners in the mesh's LOCAL coordinate frame.
    Returns (x, y, z) - the absolute position in mesh-local coordinates.

    Strategy: Use simple geometry based on bounding box percentages.
    - X: center of mesh (runners connect at center X)
    - Y: 70% from min toward max (toward back edge, away from center indent)
    - Z: 3mm above mesh bottom (on main body, above the prongs)
    """
    bounds = mesh.bounds
    x_min, y_min, z_min = bounds[0]
    x_max, y_max, z_max = bounds[1]

    # X: center of mesh
    x_attach = (x_min + x_max) / 2

    # Y: 70% from min toward max (toward "back" edge, away from center indent)
    y_attach = y_min + 0.7 * (y_max - y_min)

    # Z: fixed height above mesh bottom (on main body, above prongs)
    # Use RUNNER_Z_HEIGHT as the offset from bottom
    z_attach = z_min + RUNNER_Z_HEIGHT

    return (x_attach, y_attach, z_attach)


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


def arrange_parts_in_grid(meshes, rows=2, cols=5):
    """
    Arrange meshes in a grid pattern so their attachment points land on a regular grid.

    The attachment point of each mesh is moved to a target grid position using a
    single translation. This ensures runners connect at consistent positions.

    Args:
        meshes: List of meshes to arrange
        rows: Number of rows in grid
        cols: Number of columns in grid

    Returns:
        arranged: List of positioned meshes
        attachment_positions: List of (x, y, z) final attachment positions for each part
        cell_size: (cell_width, cell_height) tuple
        common_z: The Z height used for all attachment points
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

    # Use a fixed Z height for all runners (RUNNER_Z_HEIGHT above final mesh bottom)
    # We'll position parts so their bottoms are at Z=0, then attachment is at RUNNER_Z_HEIGHT
    common_z = RUNNER_Z_HEIGHT

    arranged = []
    attachment_positions = []

    for i, mesh in enumerate(meshes):
        row = i // cols
        col = i % cols

        # Target grid position for attachment point
        target_x = col * cell_width
        target_y = row * cell_height
        target_z = common_z

        # Find attachment point in mesh's local coordinates
        local_x, local_y, local_z = find_attachment_point_local(mesh)

        # Get mesh bounds for Z floor calculation
        z_min = mesh.bounds[0][2]

        # Calculate translation:
        # - X and Y: move local attachment point to target grid position
        # - Z: place mesh bottom at Z=0, so attachment point (at z_min + RUNNER_Z_HEIGHT)
        #      ends up at RUNNER_Z_HEIGHT = common_z
        translation = (
            target_x - local_x,
            target_y - local_y,
            -z_min  # This puts mesh bottom at Z=0, attachment at RUNNER_Z_HEIGHT
        )

        # Apply single translation
        positioned_mesh = mesh.copy()
        positioned_mesh.apply_translation(translation)

        arranged.append(positioned_mesh)
        attachment_positions.append((target_x, target_y, target_z))

    return arranged, attachment_positions, (cell_width, cell_height), common_z


def create_connected_batch(meshes):
    """
    Create a single connected mesh from multiple meshes.
    Arranges parts in a grid and connects them with runners at their attachment points.

    Args:
        meshes: List of meshes to arrange and connect
    """
    if len(meshes) == 0:
        raise ValueError("No meshes to combine")

    # Arrange parts in grid - each part's attachment point lands on the grid
    arranged, attachment_positions, cell_size, common_z = arrange_parts_in_grid(meshes)
    print(f"  Using runner Z height: {common_z:.2f}mm")

    # Create runners connecting attachment points
    runners = []
    cols = 5

    for i, attach_pos in enumerate(attachment_positions):
        row = i // cols
        col = i % cols

        # Connect to right neighbor (horizontal runner)
        if col < cols - 1 and i + 1 < len(attachment_positions):
            next_attach_pos = attachment_positions[i + 1]
            runner = create_runner(attach_pos, next_attach_pos, common_z)
            runners.append(runner)

        # Connect to bottom neighbor (vertical runner)
        if row < 1 and i + cols < len(attachment_positions):
            below_attach_pos = attachment_positions[i + cols]
            runner = create_runner(attach_pos, below_attach_pos, common_z)
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
    for filename, count in part_specs:
        print(f"  Loading {count}x {filename}")
        base_mesh = load_stl(filename)
        for _ in range(count):
            meshes.append(base_mesh.copy())

    print(f"  Total parts: {len(meshes)}")

    # Combine meshes with runners
    combined = create_connected_batch(meshes)

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
