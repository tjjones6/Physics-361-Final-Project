# Reduced-Order Modeling with Deep Learning: Flow Reconstruction for External Aerodynamics and Thermal Convection

## Overview

This repository contains the materials associated with the project titled:

> **Deep Learning on Reduced Manifolds: Data-Driven Reconstruction of Turbulent Thermal-Fluid Systems via POD and Neural Networks**

The project combines high-fidelity computational fluid dynamics (CFD), Proper Orthogonal Decomposition (POD), and deep neural networks (DNNs) to develop real-time surrogate models of turbulent thermal-fluid flows. The goal is to enable rapid and accurate prediction of complex fluid dynamics with significantly reduced computational costs.

<p align="center">
  <video src="Animations/cylinder_particle_advection.mp4" width="600" loop controls></video>
</p>

---

## Project Structure

```
├── README.md                # Project documentation and instructions
├── Paper/                   # Final report and related documentation
│   └── Reduced_Order_Modeling_of_Turbulent_Fluid_Systems.pdf
├── Code/                    # Python source code and scripts
│   ├── CFD/                 # CFD simulation scripts
│   ├── POD/                 # Scripts for Proper Orthogonal Decomposition
│   ├── ML/                  # Machine learning surrogate models (MLP)
│   └── PostProcessing/      # FTLE, streamlines, and visualization scripts
├── Data/                    # Simulation data (large datasets stored externally)
│   └── (HDF5 or NPZ data files)
└── Animations_Figures/      # Visual results: plots, animations, and figures
```

---

## How to Use this Repository

### Requirements

- Python >= 3.8
- NumPy, SciPy, Matplotlib
- scikit-learn (MLPRegressor)
- h5py (for dataset management)

You can install required packages via:

```sh
pip install numpy scipy matplotlib scikit-learn h5py
```

### Running the Project

**1. CFD Simulations:**

To generate CFD data:

```sh
cd Code/CFD
python viscous_cylinder_simulation.py
```

The data will be saved in `Data/`.

**2. POD Analysis:**

Run the POD analysis to generate modes and amplitude files:

```sh
cd Code/POD
python compute_pod.py
```

Results will be stored as NumPy arrays for ML training.

**3. Machine Learning Model Training:**

Train the deep neural network surrogate:

```sh
cd Code/ML
python train_dnn_surrogate.py
```

**4. Post-Processing and Visualization:**

For visualizing reconstructed flow fields, animations, and FTLE:

```sh
cd Code/PostProcessing
python visualize_results.py
```

---

## Key Results

- Achieved >99% flow energy reconstruction with fewer than 20 POD modes.
- ML surrogate yields under 2% mean squared reconstruction error.
- Real-time predictions (~200× faster than CFD).

Visual results (FTLE fields, POD mode animations) are available in `Animations_Figures/`.

---

## Future Work

- Incorporate viscous boundary layer modeling (Immersed Boundary methods).
- Extend neural networks to temporal modeling with LSTMs or Neural ODEs.
- Implement uncertainty quantification (Bayesian methods).

---

## Contact

For questions, feedback, or collaboration, contact:

- **Tyler Jones**
- [tjjones6@wisc.edu]
- University of Wisconsin-Madison | B.S. Applied Mathematics, Engineering, and Physics (AMEP)

---

**Date:** May 2025  
**Version:** 1.0.0
