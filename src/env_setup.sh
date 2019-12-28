conda create -y -n $1 python=3.6.8
conda activate $1
conda install -y pytorch torchvision cudatoolkit=9.0 -c pytorch
pip install tf-nightly
conda install -y -c conda-forge matplotlib
conda install -y -c conda-forge opencv
pip install openexr
conda install -y pillow
