# SLAM Algorithms

This repository contains Python implementations of popular SLAM (Simultaneous Localization and Mapping) algorithms: FastSLAM and GraphSLAM.

## Algorithms

### FastSLAM

FastSLAM is a particle filter-based SLAM algorithm that maintains a set of particles to estimate the robot's pose and the map of the environment. It combines a particle filter for pose estimation with a particle filter for map estimation. The algorithm is particularly useful for solving SLAM problems with non-linear motion and measurement models.

**File**: FastSLAM.py

### GraphSLAM

GraphSLAM is a graph-based SLAM algorithm that represents the robot's trajectory and the map of the environment as a graph. It formulates SLAM as a graph optimization problem, where the nodes represent poses and landmarks, and the edges represent the constraints between them. GraphSLAM optimizes the graph to find the most likely trajectory and map given the sensor measurements and motion commands.

**File**: GraphSLAM.py

## Dependencies

The code in this repository requires the following dependencies:

- NumPy: Library for numerical computing
- Matplotlib: Library for data visualization

To install the dependencies, run the following command:

```bash
pip install numpy matplotlib
```

## Usage

- Clone the repository:

```bash
git clone https://github.com/your-username/slam-algorithms.git
```

- Navigate to the repository:

```bash
cd slam-algorithms
```

- Run the desired SLAM algorithm:

	- For FastSLAM:

```bash
python FastSLAM.py
```

	- For GraphSLAM:
```bash
python GraphSLAM.py
```

## Examples

The repository includes examples of how to use the FastSLAM and GraphSLAM algorithms with synthetic data. The examples demonstrate the estimation of the robot's pose and the map of the environment based on motion inputs and sensor observations.
