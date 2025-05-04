# q3d: Advanced Quaternion 3D Simulator

A dynamic and interactive 3D visualization tool for exploring quaternions, 3D rotations, and animation. Built with Python, Pygame, and PyOpenGL.

## Features
- Visualize and interact with 3D objects (cube, sphere) using quaternions
- Multiple objects with independent orientation, scale, and position
- Quaternion SLERP animation and keyframe recording/playback
- Orientation trails and quaternion function plotting
- Dynamic camera controls (orbit, pan, zoom)
- Direct quaternion/Euler editing
- Scene save/load (JSON)
- On-screen OpenGL text overlay for help and values
- Object duplication and deletion

## Requirements
- Python 3.8+
- pygame
- PyOpenGL
- numpy
- scipy (for Euler/quaternion conversion)

Install dependencies with:
```bash
pip install pygame PyOpenGL numpy scipy
```

## Usage
Run the simulator:
```bash
python main.py
```

## Controls
- **TAB**: Switch shape (cube/sphere)
- **N**: New object
- **M**: Next object
- **D**: Duplicate object
- **DEL**: Delete object
- **K**: Record keyframe
- **P**: Play keyframes
- **1/2/3**: Preset spins (X/Y/Z)
- **R**: Reset orientation/scale/position
- **SPACE**: Animate to target
- **T**: Random target orientation
- **W/A/S/D/Z/X**: Move object
- **+/-**: Scale object
- **Arrow keys/Q/E**: Rotate object
- **S**: Save scene
- **L**: Load scene
- **E**: Edit quaternion/Euler (toggle mode, use arrows to edit)
- **F**: Toggle quaternion function plotter
- **Mouse**: Left-drag to rotate camera, right-drag to pan, scroll to zoom
- **ESC**: Quit

## License
MIT License

---

Created by xStFtx. Contributions welcome!
