# INSTALL BEFORE:
# Run in shell:

```
conda create -n ba_venv6 python=3.11 -y
conda activate ba_venv6
conda install gdal pip -y
pip install torch "numpy<2" rasterio matplotlib tqdm torchvision seaborn scikit-learn python-dotenv psutil plottable torchmetrics
```
For me 06.2026 locally on MacOS it was important to
- use python 3.11
- install gdal through conda, not pip
- install "numpy<2" for compatibility with other modules that were compiled using an older numpy version

OR on HPC:
```
python3 -m venv venv
source venv/bin/activate
module load cuda/12.8
python -m pip install numpy rasterio matplotlib tqdm seaborn scikit-learn python-dotenv psutil plottable torchmetrics

pip3 install torch torchvision --index-url https://download.pytorch.org/whl/cu126
```
- On the HPC gdal doesnt give issues but the CUDA compatibility does
- Combination that worked for me on most HPC GPUS:
  - load cuda 12.8 module
  - but install torch with cuda 12.6
  - Otherwise first try https://pytorch.org/get-started


Known Package Import Error:
> ImportError: dlopen(... ba_venv7/lib/python3.11/site-packages/rasterio/_base.cpython-311-darwin.so, 0x0002): Symbol not found: _BIO_ADDR_copy
pip cache remove rasterio
pip install rasterio --no-cache-dir --no-binary rasterio


# DOWNLOAD DATASETS

## MDAS Dataset
# Download zip from https://huggingface.co/datasets/torchgeo/mdas/tree/main
# needs -L to follow redirects
curl -L -o mdas.zip "https://huggingface.co/datasets/torchgeo/mdas/resolve/main/Augsburg_data_4_publication.zip?download=true"
python -m zipfile -e mdas.zip BA_MDAS/
rm "mdas.zip"


## DFC 2018 Dataset
# Make an account on https://ieee-dataport.org
# copy link on File at https://ieee-dataport.org/open-access/2018-ieee-grss-data-fusion-challenge-fusion-multispectral-lidar-and-hyperspectral-data"
> ! CHANGES EVERY TIME !
# Run in terminal:

curl -o dfc18.zip "https://ieee-dataport.s3.amazonaws.com < PUT YOUR CORRECT LINK IN HERE >"
unzip "dfc18.zip" -d "BA_DFC2018/"
rm "dfc18.zip"


# .env FILE
Needs .env file in root with e.g.:

```
ALL_DATA_FILE_ROOT="./data/"
ENV_ACTIVATE_COMMAND="conda activate <MY_CONDA_VENV>"

# OR

ALL_DATA_FILE_ROOT = "/scratch/<MY_SCRATCH_FOLDER>/"
ENV_ACTIVATE_COMMAND="source <MY_PYTHON_VENV>/bin/activate"

```
