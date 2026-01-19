#!/usr/bin/env python3
"""
Interactive 3D UI for fine-tuning part positions in STL batches.

Controls:
  0-9     Select part to adjust
  X/Y/Z   Select axis to adjust
  Up/Down Nudge selected part along axis (+/- 0.5mm)
  Shift+Up/Down Fine nudge (+/- 0.1mm)
  R       Reset current part offsets to zero
  S       Save offsets and regenerate STL
  N/P     Next/Previous batch
  V       Cycle through preset views
  Q       Quit
"""

import json
import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
from mpl_toolkits.mplot3d.art3d import Poly3DCollection
from pathlib import Path
import trimesh

# Import batch definitions from combine_stl
from combine_stl import (
    BATCHES, STL_DIR, OUTPUT_DIR, RUNNER_WIDTH, RUNNER_THICKNESS,
    load_stl, find_attachment_point_local, create_runner, get_mesh_dimensions,
    PART_SPACING
)

OFFSETS_FILE = Path("part_offsets.json")

# Preset views: (elevation, azimuth, name)
VIEWS = [
    (30, 45, "Isometric"),
    (90, -90, "Top"),
    (0, 0, "Back"),
    (0, -90, "Side"),
]


class TuningUI:
    def __init__(self):
        self.offsets = self.load_offsets()
        self.batch_names = list(BATCHES.keys())
        self.current_batch_idx = 0
        self.selected_part = 0
        self.current_axis = 'y'  # Default to Y (depth adjustment most common)
        self.current_view_idx = 0
        self.view_elev = VIEWS[0][0]
        self.view_azim = VIEWS[0][1]

        # Track part info for current batch
        self.part_names = []  # Name of each part in current batch
        self.meshes = []      # Original meshes

        # Setup matplotlib figure
        self.fig = plt.figure(figsize=(14, 10))
        self.ax = self.fig.add_subplot(111, projection='3d')

        # Connect keyboard events
        self.fig.canvas.mpl_connect('key_press_event', self.on_key)

        # Initial render
        self.load_batch()
        self.render()

    def load_offsets(self):
        """Load offsets from JSON file."""
        if OFFSETS_FILE.exists():
            with open(OFFSETS_FILE) as f:
                return json.load(f)
        return {}

    def save_offsets(self):
        """Save offsets to JSON file."""
        with open(OFFSETS_FILE, 'w') as f:
            json.dump(self.offsets, f, indent=2)
        print(f"Saved offsets to {OFFSETS_FILE}")

    def get_offset(self, part_name):
        """Get offset for a part, defaulting to zeros."""
        return self.offsets.get(part_name, {'x': 0.0, 'y': 0.0, 'z': 0.0})

    def set_offset(self, part_name, axis, value):
        """Set offset for a part on a specific axis."""
        if part_name not in self.offsets:
            self.offsets[part_name] = {'x': 0.0, 'y': 0.0, 'z': 0.0}
        self.offsets[part_name][axis] = value

    def load_batch(self):
        """Load meshes for the current batch."""
        batch_name = self.batch_names[self.current_batch_idx]
        part_specs = BATCHES[batch_name]

        self.meshes = []
        self.part_names = []

        for filename, count in part_specs:
            part_name = filename.replace('.stl', '')
            base_mesh = load_stl(filename)
            for _ in range(count):
                self.meshes.append(base_mesh.copy())
                self.part_names.append(part_name)

        # Reset selection if out of range
        if self.selected_part >= len(self.meshes):
            self.selected_part = 0

    def arrange_with_offsets(self):
        """Arrange parts in grid with current offsets applied."""
        rows, cols = 2, 5

        # Find max dimensions for cell size
        max_dims = np.array([0.0, 0.0, 0.0])
        for mesh in self.meshes:
            dims = get_mesh_dimensions(mesh)
            max_dims = np.maximum(max_dims, dims)

        cell_width = max_dims[0] + PART_SPACING
        cell_height = max_dims[1] + PART_SPACING

        # First pass: find attachment points to determine common Z
        local_attachments = []
        for i, mesh in enumerate(self.meshes):
            local_x, local_y, local_z = find_attachment_point_local(mesh)
            z_min = mesh.bounds[0][2]
            relative_z = local_z - z_min

            # Apply offset to local attachment
            offset = self.get_offset(self.part_names[i])
            local_x += offset['x']
            local_y += offset['y']

            local_attachments.append((local_x, local_y, local_z, z_min, relative_z, offset['z']))

        # Common Z is max relative_z
        common_z = max(att[4] for att in local_attachments)

        arranged = []
        attachment_positions = []

        for i, mesh in enumerate(self.meshes):
            row = i // cols
            col = i % cols

            target_x = col * cell_width
            target_y = row * cell_height
            target_z = common_z

            local_x, local_y, local_z, z_min, relative_z, z_offset = local_attachments[i]

            translation = (
                target_x - local_x,
                target_y - local_y,
                common_z - relative_z - z_min + z_offset
            )

            positioned_mesh = mesh.copy()
            positioned_mesh.apply_translation(translation)

            arranged.append(positioned_mesh)
            attachment_positions.append((target_x, target_y, target_z))

        # Create runners
        runners = []
        for i, attach_pos in enumerate(attachment_positions):
            row = i // cols
            col = i % cols

            if col < cols - 1 and i + 1 < len(attachment_positions):
                next_attach_pos = attachment_positions[i + 1]
                runner = create_runner(attach_pos, next_attach_pos, common_z)
                runners.append(runner)

            if row < 1 and i + cols < len(attachment_positions):
                below_attach_pos = attachment_positions[i + cols]
                runner = create_runner(attach_pos, below_attach_pos, common_z)
                runners.append(runner)

        return arranged, runners, attachment_positions

    def render(self):
        """Render the current batch with 3D visualization."""
        # Save current view before clearing
        if hasattr(self.ax, 'elev') and self.ax.elev is not None:
            self.view_elev = self.ax.elev
            self.view_azim = self.ax.azim

        self.ax.clear()

        # Arrange parts with offsets
        arranged, runners, attachment_positions = self.arrange_with_offsets()

        # Render each part
        for i, mesh in enumerate(arranged):
            color = 'orangered' if i == self.selected_part else 'steelblue'
            alpha = 0.9 if i == self.selected_part else 0.6
            self._render_mesh(mesh, color, alpha)

            # Add part number label
            centroid = mesh.centroid
            self.ax.text(centroid[0], centroid[1], mesh.bounds[1][2] + 2,
                        str(i), fontsize=10, ha='center', color='black',
                        fontweight='bold' if i == self.selected_part else 'normal')

        # Render runners
        for runner in runners:
            self._render_mesh(runner, 'gray', 0.8)

        # Set axis limits
        all_meshes = arranged + runners
        if all_meshes:
            combined = trimesh.util.concatenate(all_meshes)
            bounds = combined.bounds
            margin = 5
            self.ax.set_xlim(bounds[0][0] - margin, bounds[1][0] + margin)
            self.ax.set_ylim(bounds[0][1] - margin, bounds[1][1] + margin)
            self.ax.set_zlim(bounds[0][2] - margin, bounds[1][2] + margin)

        self.ax.set_xlabel('X')
        self.ax.set_ylabel('Y')
        self.ax.set_zlabel('Z')

        # Restore view angle
        self.ax.view_init(elev=self.view_elev, azim=self.view_azim)

        # Update title with status
        batch_name = self.batch_names[self.current_batch_idx]
        part_name = self.part_names[self.selected_part] if self.part_names else "N/A"
        offset = self.get_offset(part_name)

        # Main title with batch info
        self.ax.set_title(f"{batch_name}", fontsize=12, fontweight='bold')

        # Add status text in figure coordinates (always visible, not affected by 3D rotation)
        # Clear any previous status text
        for txt in self.fig.texts:
            txt.remove()

        # Part selection status
        self.fig.text(0.02, 0.95, f"Part: {self.selected_part} ({part_name})",
                     fontsize=12, fontweight='bold', color='orangered',
                     transform=self.fig.transFigure)

        # Axis selection - highlight the selected axis
        axis_colors = {'x': 'black', 'y': 'black', 'z': 'black'}
        axis_colors[self.current_axis] = 'blue'
        axis_weights = {'x': 'normal', 'y': 'normal', 'z': 'normal'}
        axis_weights[self.current_axis] = 'bold'

        self.fig.text(0.02, 0.90, f"Axis: ", fontsize=11, transform=self.fig.transFigure)
        self.fig.text(0.07, 0.90, "X", fontsize=11, color=axis_colors['x'],
                     fontweight=axis_weights['x'], transform=self.fig.transFigure)
        self.fig.text(0.09, 0.90, " / ", fontsize=11, transform=self.fig.transFigure)
        self.fig.text(0.11, 0.90, "Y", fontsize=11, color=axis_colors['y'],
                     fontweight=axis_weights['y'], transform=self.fig.transFigure)
        self.fig.text(0.13, 0.90, " / ", fontsize=11, transform=self.fig.transFigure)
        self.fig.text(0.15, 0.90, "Z", fontsize=11, color=axis_colors['z'],
                     fontweight=axis_weights['z'], transform=self.fig.transFigure)

        # Current offsets - highlight non-zero values
        def fmt_offset(val, axis):
            color = 'green' if val != 0 else 'gray'
            weight = 'bold' if axis == self.current_axis else 'normal'
            return val, color, weight

        self.fig.text(0.02, 0.85, "Offsets:", fontsize=11, transform=self.fig.transFigure)
        x_val, x_col, x_wt = fmt_offset(offset['x'], 'x')
        y_val, y_col, y_wt = fmt_offset(offset['y'], 'y')
        z_val, z_col, z_wt = fmt_offset(offset['z'], 'z')

        self.fig.text(0.10, 0.85, f"X={x_val:+.1f}", fontsize=11, color=x_col,
                     fontweight=x_wt, transform=self.fig.transFigure)
        self.fig.text(0.19, 0.85, f"Y={y_val:+.1f}", fontsize=11, color=y_col,
                     fontweight=y_wt, transform=self.fig.transFigure)
        self.fig.text(0.28, 0.85, f"Z={z_val:+.1f}", fontsize=11, color=z_col,
                     fontweight=z_wt, transform=self.fig.transFigure)

        # Help text at bottom
        self.fig.text(0.5, 0.02,
                     "[0-9]=part  [X/Y/Z]=axis  [↑↓]=nudge  [S]=save  [N/P]=batch  [V]=view  [Q]=quit",
                     fontsize=9, ha='center', transform=self.fig.transFigure, color='dimgray')

        self.fig.canvas.draw_idle()

    def _render_mesh(self, mesh, color, alpha):
        """Render a single mesh as polygons."""
        # Sample faces to keep rendering fast
        step = max(1, len(mesh.faces) // 1500)
        faces_to_plot = mesh.faces[::step]

        polygons = [mesh.vertices[face] for face in faces_to_plot]
        collection = Poly3DCollection(polygons, alpha=alpha, linewidth=0.1, edgecolor='gray')
        collection.set_facecolor(color)
        self.ax.add_collection3d(collection)

    def on_key(self, event):
        """Handle keyboard events."""
        if event.key == 'q':
            plt.close(self.fig)
            return

        elif event.key in '0123456789':
            part_num = int(event.key)
            if part_num < len(self.meshes):
                self.selected_part = part_num

        elif event.key in ('x', 'y', 'z'):
            self.current_axis = event.key

        elif event.key == 'up':
            self._nudge(0.5)
        elif event.key == 'down':
            self._nudge(-0.5)
        elif event.key == 'shift+up':
            self._nudge(0.1)
        elif event.key == 'shift+down':
            self._nudge(-0.1)

        elif event.key == 'r':
            # Reset current part offsets
            part_name = self.part_names[self.selected_part]
            self.offsets[part_name] = {'x': 0.0, 'y': 0.0, 'z': 0.0}
            print(f"Reset offsets for {part_name}")

        elif event.key == 's':
            self.save_offsets()
            self._regenerate_stl()

        elif event.key == 'n':
            self.current_batch_idx = (self.current_batch_idx + 1) % len(self.batch_names)
            self.load_batch()
        elif event.key == 'p':
            self.current_batch_idx = (self.current_batch_idx - 1) % len(self.batch_names)
            self.load_batch()

        elif event.key == 'v':
            self.current_view_idx = (self.current_view_idx + 1) % len(VIEWS)
            self.view_elev, self.view_azim, view_name = VIEWS[self.current_view_idx]
            print(f"View: {view_name}")

        self.render()

    def _nudge(self, delta):
        """Nudge the selected part along the current axis."""
        part_name = self.part_names[self.selected_part]
        offset = self.get_offset(part_name)
        new_value = offset[self.current_axis] + delta
        self.set_offset(part_name, self.current_axis, new_value)
        print(f"{part_name} {self.current_axis.upper()}: {new_value:.1f}mm")

    def _regenerate_stl(self):
        """Regenerate the current batch STL with offsets."""
        batch_name = self.batch_names[self.current_batch_idx]
        print(f"Regenerating {batch_name}...")

        arranged, runners, _ = self.arrange_with_offsets()
        all_meshes = arranged + runners
        combined = trimesh.util.concatenate(all_meshes)

        OUTPUT_DIR.mkdir(exist_ok=True)
        output_path = OUTPUT_DIR / f"{batch_name}.stl"
        combined.export(str(output_path))
        print(f"Saved: {output_path}")

    def run(self):
        """Start the UI."""
        plt.show()


def main():
    print("=" * 60)
    print("STL Position Tuning UI")
    print("=" * 60)
    print(__doc__)

    ui = TuningUI()
    ui.run()


if __name__ == "__main__":
    main()
