# 3D-PINN-Aerodynamics-

# Physics-Informed Neural Network for Multi-Phase Boundary Flows

A PyTorch implementation of a 3D Physics-Informed Neural Network (PINN) designed to model non-linear transient fluid flows, advection-diffusion boundaries, and wake interactions. The model ingests structural masks alongside high-fidelity volumetric data to output conservation-compliant velocity distributions and scalar smoke transports.

---

# Architectural Overview

Traditional deep learning models treat fluid mechanics as pure image-to-image translations, causing non-physical mass loss and gradient explosions near sharp structural boundaries. This architecture addresses those constraints through an end-to-end 3D Convolutional Network embedded with an inline **Hybrid Physics Loss Layer**.




## Physics-Informed Formulation

The network trains by optimizing a unified objective function across structural data parameters and partial differential equations governing transport and mass continuity:

\[\mathcal{L}_{\text{total}} = \mathcal{L}_{\text{data}} + \lambda_1 \mathcal{L}_{\text{divergence}} + \lambda_2 \mathcal{L}_{\text{transport}}\]

### 1. Incompressibility Constraint (Mass Conservation)
To enforce physical mass conservation, the fluid velocities are driven toward a zero-divergence field via spatial central differences:

\[\mathcal{L}_{\text{divergence}} = \frac{1}{N}\sum \left( \frac{\partial u}{\partial x} + \frac{\partial v}{\partial y} + \frac{\partial w}{\partial z} \right)^2\]

### 2. Scalar Advection-Diffusion Transport
The dynamic propagation of smoke density \(\phi\) is bound to physical advection and diffusion equations using calculated velocity parameters:

\[\mathcal{L}_{\text{transport}} = \frac{1}{N}\sum \left( \left[ u\frac{\partial \phi}{\partial x} + v\frac{\partial \phi}{\partial y} + w\frac{\partial \phi}{\partial z} \right] - D\nabla^2 \phi \right)^2\]

Where \(D\) represents the specified isotropic diffusion coefficient.

---

##  Optimization & Boundary Stabilization Safeguards

When deep neural networks encounter vertical structural discontinuities (\(0.0 \rightarrow 1.0\) masks), calculated gradient variables approach infinity, resulting in runtime crashes. This repository includes two distinct stabilization techniques to safely resolve optimization parameters:

* **Gaussian Boundary Regularization:** Structural shapes are modulated using an optimized Gaussian filter (\(\sigma = 1.2\)). Blurring sharp boundaries translates step discontinuities into continuous profiles, providing stable, differentiable topologies for spatial derivative steps.
* **Inline Norm Circuit Breaker:** Implements a strict gradient clipping ceiling (`max_norm=1.0`) directly preceding optimization adjustments to eliminate structural weight explosion issues caused by transient high-velocity wave interactions.

---

##  Getting Started

### Prerequisites
Ensure your hardware cluster contains a CUDA-compliant GPU for volumetric tensor optimization loops. 

```bash
# Clone the repository
git clone https://github.com
cd 3D-PINN-Aerodynamics

# Install dependencies
pip install -r requirements.txt
```

### Data Workspace Configuration
This training routine expects multi-dimensional array slices conforming to the BLASTNet format. Mount your local dataset workspace matching the structure below:
```text
../input/blastnet-momentum128-3d-sr-dataset/
└── your-dataset-subfolders/
    ├── frame_0001.npy
    └── frame_0002.npy
```

### Execution
Run the optimization script:
```bash
python train_pinn.py
```
