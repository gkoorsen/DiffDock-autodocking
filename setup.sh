#!/bin/bash

# Make sure to set the script as executable using the command chmod +x setup.sh before running it.
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
pip install ipython-autotime pyg==0.7.1 pyyaml==6.0 scipy==1.7.3 networkx==2.6.3 biopython==1.79 rdkit-pypi==2022.03.5 e3nn==0.5.0 spyrmsd==0.5.2 biopandas==0.4.1 torch==1.12.1+cu113 --quiet

# Check for torch_geometric and install if not present
python -c "
import torch
try:
    import torch_geometric
except ModuleNotFoundError:
    !pip uninstall torch-scatter torch-sparse torch-geometric torch-cluster  --y
    !pip install torch-scatter -f https://data.pyg.org/whl/torch-{torch.__version__}.html --quiet
    !pip install torch-sparse -f https://data.pyg.org/whl/torch-{torch.__version__}.html --quiet
    !pip install torch-cluster -f https://data.pyg.org/whl/torch-{torch.__version__}.html --quiet
    !pip install git+https://github.com/pyg-team/pytorch_geometric.git  --quiet 
"

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
