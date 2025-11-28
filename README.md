# Project Synapse: A Generative Foundation Model for Biomolecular Dynamics

**Project Synapse** is an AI-driven system designed to move beyond static protein structures (like AlphaFold) to simulate dynamic interactions and generate novel therapeutics.

> *"Structure is not Function. Biology is a 4D phenomenon."*

## 🧬 Core Modules

This repository contains the MVP (Minimum Viable Product) implementation of the Synapse engine:

### [cite_start]1. The Eye (Data & Vision) [cite: 9, 31, 32]
- **`01_data_loader.py`**: Automated pipeline to fetch 3D structures from PDB.
- **`02_featurizer.py`**: Converts raw atomic coordinates into Graph-based tensors for AI processing.

### [cite_start]2. The Brain (Geometric AI) [cite: 41, 42, 45]
- **`03_model.py`**: Implements a **Graph Neural Network (GNN)** to understand spatial relationships between atoms.
- **`04_train.py`**: Trains the model to predict atomic dynamics (B-factors) from geometry alone.
- **`05_inference_viz.py`**: Generates 3D heatmaps of protein flexibility compatible with PyMOL.

### [cite_start]3. The Designer (Generative Chemistry) [cite: 12, 13, 47]
- **`06_drug_generator.py`**: An LSTM-based language model that learns the grammar of chemistry (SMILES) to generate novel molecules.
- **`08_synapse_core.py`**: Integration with **ChemBERTa** (Transformer architecture) for advanced molecular property prediction.

### [cite_start]4. Virtual Lab (Interaction) [cite: 11, 46]
- **`07_virtual_lab.py`**: A fusion network that predicts binding affinity between a target protein and a generated drug.

## 🚀 Quick Start

1. Install dependencies:
```bash
conda create -n synapse python=3.10
pip install torch transformers rdkit biopython

Run the "Miracle Test" to see AI chemical intuition:

Bash

python src/09_miracle_test.py
🔬 Tech Stack
PyTorch (Core AI Engine)

Transformers (Pre-trained LLMs)

RDKit (Cheminformatics)

BioPython (Structural Biology)


Developed by Alan Jafari - Neurovix AI# Project-Synapse
