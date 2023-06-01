#!/bin/bash

# Make sure to set the script as executable using the command chmod +x setup.sh before running it.
# Execute setup.sh from the ~ directory
# Remove the existing results directory
if [ -d "~/DiffDock/results" ]
then
    rm -rf /content/DiffDock/results
fi

# Clone DiffDock if it doesn't exist
if [ ! -d "~/DiffDock" ]
then
    cd ~
    git clone https://github.com/gcorso/DiffDock.git
    cd ~/DiffDock
    git checkout a6c5275
fi

# Install necessary Python packages
pip install ipython-autotime pyg==0.7.1 pyyaml==6.0 scipy==1.7.3 networkx==2.6.3 biopython==1.79 rdkit-pypi==2022.03.5 e3nn==0.5.0 spyrmsd==0.5.2 biopandas==0.4.1 torch --quiet

# Install PyTorch Geometric dependencies
pip install torch-scatter torch-sparse torch-cluster torch-spline-conv torch-geometric --quiet

# Clone and install ESM if not present
if [ ! -d "~/DiffDock/esm" ]
then
    cd ~/DiffDock
    git clone https://github.com/facebookresearch/esm
    cd ~/DiffDock/esm
    git checkout ca8a710
    pip install -e .
    cd ~/DiffDock
fi

# Fetch script for multiple docking:
wget https://raw.githubusercontent.com/gkoorsen/DiffDock-autodocking/main/multi_complex_DiffDock.py -P ~/DiffDock/
