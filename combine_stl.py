#!/usr/bin/env python3
"""
Script to combine multiple STL files into batches of 10 connected parts for JLC3DP printing.
Each batch has parts arranged in a 2x5 grid with 1.5mm thick connecting runners.
"""

import json
import trimesh
import numpy as np
from pathlib import Path

# Configuration
STL_DIR = Path("Adapter STL files")
OUTPUT_DIR = Path("Combined STL files")
OFFSETS_FILE = Path("part_offsets.json")
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


def load_offsets():
    """Load part offsets from JSON file. Returns empty dict if file doesn't exist."""
    if OFFSETS_FILE.exists():
        with open(OFFSETS_FILE) as f:
            return json.load(f)
    return {}


def get_offset(offsets, part_name):
    """Get offset for a part, defaulting to zeros."""
    return offsets.get(part_name, {'x': 0.0, 'y': 0.0, 'z': 0.0})


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

    Strategy: Find where back face, side faces, AND top surface all exist,
    so runner doesn't jut out past any face or into indent areas.
    - X: center of mesh (runners connect at center X)
    - Y: constrained by back face, side face, AND top surface extents
    - Z: minimum Z of back face above prongs (where flat surface exists)
    """
    bounds = mesh.bounds
    x_min, y_min, z_min = bounds[0]
    x_max, y_max, z_max = bounds[1]
    normals = mesh.face_normals

    # X: center of mesh
    x_attach = (x_min + x_max) / 2

    # Filter threshold for above prongs
    prong_threshold = z_min + 2.5

    # Find back faces (+Y normal)
    back_mask = normals[:, 1] > 0.9

    if back_mask.any():
        back_face_indices = np.where(back_mask)[0]
        back_faces = mesh.faces[back_face_indices]
        back_verts = mesh.vertices[back_faces.flatten()]

        # Filter to vertices above prongs
        back_above_prongs = back_verts[back_verts[:, 2] > prong_threshold]

        if len(back_above_prongs) > 0:
            # Z: Use minimum Z of back face above prongs
            z_attach = np.min(back_above_prongs[:, 2])

            # Find side faces (±X normal) at the runner Z level
            left_mask = normals[:, 0] < -0.9
            right_mask = normals[:, 0] > 0.9
            side_mask = left_mask | right_mask

            # Get max Y extent where side faces exist at the runner Z level
            side_y_max = y_max  # Default to mesh bound
            if side_mask.any():
                side_verts = mesh.vertices[mesh.faces[np.where(side_mask)[0]].flatten()]
                z_tolerance = RUNNER_THICKNESS + 1.0
                side_at_z = side_verts[
                    (side_verts[:, 2] >= z_attach - 0.5) &
                    (side_verts[:, 2] <= z_attach + z_tolerance)
                ]
                if len(side_at_z) > 0:
                    side_y_max = np.max(side_at_z[:, 1])

            # Find top surface (+Z normal) Y extent
            # This prevents placing runner in back indent areas with no top
            top_mask = normals[:, 2] > 0.9
            top_y_max = y_max  # Default to mesh bound
            if top_mask.any():
                top_verts = mesh.vertices[mesh.faces[np.where(top_mask)[0]].flatten()]
                top_above_prongs = top_verts[top_verts[:, 2] > prong_threshold]
                if len(top_above_prongs) > 0:
                    top_y_max = np.max(top_above_prongs[:, 1])

            # Back face max Y
            back_y_max = np.max(back_above_prongs[:, 1])

            # Y attachment: use the MINIMUM of all three constraints
            # This ensures runner is on solid material with top surface above it
            effective_y_max = min(back_y_max, side_y_max, top_y_max)

            # Offset inward by RUNNER_WIDTH/2 so runner doesn't extend past
            y_attach = effective_y_max - RUNNER_WIDTH / 2

            return (x_attach, y_attach, z_attach)

    # Fallback to bounding box estimation if back face analysis fails
    y_attach = y_max - 1.0  # 1mm inside back edge
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


def arrange_parts_in_grid(meshes, part_names=None, offsets=None, rows=2, cols=5):
    """
    Arrange meshes in a grid pattern so their attachment points land on a regular grid.

    The attachment point of each mesh is moved to a target grid position using a
    single translation. This ensures runners connect at consistent positions.

    Args:
        meshes: List of meshes to arrange
        part_names: List of part names (for offset lookup), or None for no offsets
        offsets: Dict of part offsets from load_offsets(), or None for no offsets
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

    # First pass: find attachment points for all meshes to determine common Z
    # Each mesh has its own optimal Z based on where its back face starts
    local_attachments = []
    for i, mesh in enumerate(meshes):
        local_x, local_y, local_z = find_attachment_point_local(mesh)
        z_min = mesh.bounds[0][2]
        # The attachment Z relative to mesh bottom
        relative_z = local_z - z_min

        # Apply X/Y offsets to local attachment point
        z_offset = 0.0
        if part_names and offsets:
            offset = get_offset(offsets, part_names[i])
            local_x += offset['x']
            local_y += offset['y']
            z_offset = offset['z']

        local_attachments.append((local_x, local_y, local_z, z_min, relative_z, z_offset))

    # Use the MAXIMUM relative Z as common_z so all runners are at a valid height
    # This ensures no runner is below any part's back face
    common_z = max(att[4] for att in local_attachments)

    arranged = []
    attachment_positions = []

    for i, mesh in enumerate(meshes):
        row = i // cols
        col = i % cols

        # Target grid position for attachment point
        target_x = col * cell_width
        target_y = row * cell_height
        target_z = common_z

        local_x, local_y, local_z, z_min, relative_z, z_offset = local_attachments[i]

        # Calculate translation:
        # - X and Y: move local attachment point to target grid position
        # - Z: place mesh so its attachment point (at relative_z from bottom)
        #      ends up at common_z, plus any Z offset
        translation = (
            target_x - local_x,
            target_y - local_y,
            common_z - relative_z - z_min + z_offset
        )

        # Apply single translation
        positioned_mesh = mesh.copy()
        positioned_mesh.apply_translation(translation)

        arranged.append(positioned_mesh)
        attachment_positions.append((target_x, target_y, target_z))

    return arranged, attachment_positions, (cell_width, cell_height), common_z


def create_connected_batch(meshes, part_names=None, offsets=None):
    """
    Create a single connected mesh from multiple meshes.
    Arranges parts in a grid and connects them with runners at their attachment points.

    Args:
        meshes: List of meshes to arrange and connect
        part_names: List of part names (for offset lookup), or None for no offsets
        offsets: Dict of part offsets from load_offsets(), or None for no offsets
    """
    if len(meshes) == 0:
        raise ValueError("No meshes to combine")

    # Arrange parts in grid - each part's attachment point lands on the grid
    arranged, attachment_positions, cell_size, common_z = arrange_parts_in_grid(
        meshes, part_names=part_names, offsets=offsets
    )
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


def process_batch(batch_name, part_specs, offsets=None):
    """
    Process a single batch: load parts, combine, and save.
    part_specs is a list of (filename, count) tuples.
    offsets is a dict of part offsets from load_offsets(), or None for no offsets.
    """
    print(f"\nProcessing {batch_name}...")

    # Load all required meshes and track part names
    meshes = []
    part_names = []
    for filename, count in part_specs:
        print(f"  Loading {count}x {filename}")
        part_name = filename.replace('.stl', '')
        base_mesh = load_stl(filename)
        for _ in range(count):
            meshes.append(base_mesh.copy())
            part_names.append(part_name)

    print(f"  Total parts: {len(meshes)}")

    # Combine meshes with runners (applying offsets if provided)
    combined = create_connected_batch(meshes, part_names=part_names, offsets=offsets)

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

    # Load offsets if available
    offsets = load_offsets()
    if offsets:
        print(f"\nLoaded offsets for {len(offsets)} part types from {OFFSETS_FILE}")

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
        process_batch(batch_name, part_specs, offsets=offsets)
        total_parts += sum(count for _, count in part_specs)
    
    print("\n" + "=" * 60)
    print(f"COMPLETE! Generated {len(BATCHES)} combined STL files")
    print(f"Total parts: {total_parts}")
    print(f"Output location: {OUTPUT_DIR}/")
    print("=" * 60)
    
    return 0


if __name__ == "__main__":
    exit(main())
