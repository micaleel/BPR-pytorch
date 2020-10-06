#!/bin/sh

if [ "$(uname)" == "Darwin" ]; then
    echo "MacOS Binaries dont support CUDA, install from source if necessary"
else
    eval "$(conda shell.bash hook)"
    conda activate torch-bpr
    conda install pytorch cudatoolkit=10.2 -c pytorch
fi
