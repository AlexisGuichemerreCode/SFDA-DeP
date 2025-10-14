#!/bin/bash
# Run script: ./make_venv.sh NAME_OF_YOUR_VENV

env=$1

module load StdEnv/2023
module load python/3.12.4
module load cuda/12.6
module load swig/4.1.1
module load gcc/13.3
module load opencv/4.11.0
module load scipy-stack/2025a

cdir=$(pwd)

echo "Create CC env $env"
mkdir -p ~/Venvs
rm -fr  ~/Venvs/$env
virtualenv --no-download ~/Venvs/$env
source ~/Venvs/$env/bin/activate

echo "Installing..."
pip install  --upgrade pip

pip install -r reqs_cc.txt

pip install --no-cache-dir torch==2.6.0 torchvision==0.21.0 torchaudio==2.6.0 --extra-index-url https://download.pytorch.org/whl/cu117

pip install pretrainedmodels zipp timm kornia
pip install efficientnet-pytorch==0.7.1

pip install seaborn

pip install gdown
pip install pykeops==2.1.2

cd $cdir


module load StdEnv/2023
module load python/3.12.4
module load cuda/12.6
module load swig/4.1.1
module load gcc/13.3
module load opencv/4.11.0
module load scipy-stack/2025a


cd $cdir/dlib/crf/crfwrapper/bilateralfilter
swig -python -c++ bilateralfilter.i
pip install .

cd $cdir
cd dlib/crf/crfwrapper/colorbilateralfilter
swig -python -c++ colorbilateralfilter.i
pip install .


echo "Done creating and installing virt.env: $env."