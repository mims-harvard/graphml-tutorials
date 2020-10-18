#!/bin/bash
set -e
eval "$(conda shell.bash hook)"
conda activate base
conda config --set channel_priority false
conda env create -f environment.yml
conda activate graphml-venv
conda install python=3.8.5
conda install -c conda-forge jupyterlab

case $(uname | tr '[:upper:]' '[:lower:]') in
  linux*)
    echo "LINUX"
    conda install pytorch torchvision cudatoolkit -c pytorch
    ;;
  darwin*)
    echo "OSX"
    conda install pytorch torchvision -c pytorch
    ;;
  *)
      exit 1
    ;;
esac
conda info --envs
TORCH=$(python -c "import torch; print(torch.__version__)")
CUDA=$(python -c "import torch; print(torch.version.cuda)")

# Checks CUDA variant and installs DGL and Sets CUDA variable
if [ "$CUDA" = "None" ]; then
    CUDA="cpu";
    conda install -c dglteam dgl;
elif [ "$CUDA" = "10.1" ]; then
    CUDA="cu101";
    conda install -c dglteam dgl-cuda10.1;
elif [ "$CUDA" = "10.0" ]; then
    CUDA="cu100";
    conda install -c dglteam dgl-cuda10.0;
fi

# Below is to ensure we have the correct version of Pytorch Installed
pip install torch-scatter==latest+"$CUDA" -f https://pytorch-geometric.com/whl/torch-"$TORCH".html > /dev/null
pip install torch-sparse==latest+"$CUDA" -f https://pytorch-geometric.com/whl/torch-"$TORCH".html > /dev/null
pip install torch-cluster==latest+"$CUDA" -f https://pytorch-geometric.com/whl/torch-"$TORCH".html > /dev/null
pip install torch-spline-conv==latest+"$CUDA" -f https://pytorch-geometric.com/whl/torch-"$TORCH".html > /dev/null
pip install torch-geometric > /dev/null

echo "Installation Successful"
echo "Activate Conda Environment with: conda activate graphml-venv"

