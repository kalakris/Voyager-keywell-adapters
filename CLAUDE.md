# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Repository Overview

This repository contains 3D-printable adapters that curve ZSA Voyager keyboard keycaps into a bowl shape. The main assets are 3MF and STL files for printing, along with a Python utility for batch combining STL files.

## STL Combiner Script

The `combine_stl.py` script combines individual STL files into batches of 10 connected parts for JLC3DP printing, connected by break-off runners.

### Running the Script

```bash
# Create virtual environment (uses trimesh library)
python3 -m venv venv
source venv/bin/activate
pip install trimesh numpy

# Run the combiner
python combine_stl.py
```

Output goes to `Combined STL files/` directory.

### Architecture

The script:
1. Loads individual adapter STL files from `Adapter STL files/`
2. Analyzes each mesh to find optimal runner attachment points (avoiding prongs and center indents)
3. Arranges parts in a 2x5 grid with connecting runners at a common Z height
4. Exports combined meshes for batch printing

Key configuration in the script:
- `BATCHES` dict defines which parts go into each of the 9 batches
- `RUNNER_THICKNESS`, `RUNNER_WIDTH`, `PART_SPACING` control runner geometry
- `find_optimal_attachment_point()` analyzes mesh normals to avoid fragile areas

## File Structure

- `Adapter STL files/` - Individual adapter STLs named by position: `{Left|Middle|Right}_{-1|0|1|2}.stl`
- `Combined STL files/` - Generated batch files (not tracked)
- `*.3mf` - Complete print-ready files for personal printing
