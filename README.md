# PIGNN-1D-Blood-Flow-Simulation

This repository contains the code and data for the paper:

**"Physics-Informed Graph Neural Networks to Solve 1-D Equations of Blood Flow"** (https://www.sciencedirect.com/science/article/pii/S0169260724004206)

## Overview

This project implements a Physics-Informed Graph Neural Network (PIGNN) framework to solve 1-D equations governing blood flow in arterial networks. The model predicts blood velocity and cross-sectional area waveforms using sparse velocity measurements and incorporates physical laws (mass and momentum conservation) into the training loss.

Key features:
- Solves nonlinear 1-D blood flow equations.
- Graph-based architecture modeling vessel topology.
- ARMA Graph Convolutional Network layers.
- Physics-informed loss using conservation laws.
- Supports in silico and in vivo data.

## Installation
1. **Clone the repository:**
   ```bash
   git clone https://github.com/ahmetsenemse/PIGNN-1D-Blood-Flow-Simulation.git
   cd PIGNN-1D-Blood-Flow-Simulation

2. python -m venv venv
   source venv/bin/activate     # On Windows: venv\Scripts\activate
   pip install -r requirements.txt

3. cd PIGNN_1_artery
   python GNN.py


## Citation
@article{Sen2024PIGNN,
  title={Physics-Informed Graph Neural Networks to Solve 1-D Equations of Blood Flow},
  author={Ahmet Sen, Elnaz Ghajar-Rahimi, Miquel Aguirre, Laurent Navarro, Craig J. Goergen, Stephane Avril},
  journal={Computer Methods and Programs in Biomedicine},
  year={2024},
  doi={10.1016/j.cmpb.2024.108427}
}

For questions or contributions, please contact Ahmet Sen (ahmet.1.sen@kcl.ac.uk).

 
