# # BA

import os
import random
from datetime import datetime

from dotenv import load_dotenv, dotenv_values 
import matplotlib.pyplot as plt
import matplotlib.colors as colors
import numpy as np

import rasterio as rio
from rasterio import windows
from rasterio import plot
from tqdm import tqdm # nice process bar
import torch
import torch.nn as nn

import psutil

# load .env file
load_dotenv() 


# Randomness for prod
generator = torch.Generator()
NP_RAND = np.random.default_rng() 

""" # For Testing Only: Set random seed
# FIXME
RAND_SEED = 42
NP_RAND = np.random.default_rng(seed=RAND_SEED)
torch.manual_seed(RAND_SEED)
generator.manual_seed(RAND_SEED) """

# ------

ALL_DATA_FILE_ROOT = os.getenv("ALL_DATA_FILE_ROOT")
print("ALL_DATA_FILE_ROOT:", ALL_DATA_FILE_ROOT)
if ALL_DATA_FILE_ROOT is None:
    raise EnvironmentError("ALL_DATA_FILE_ROOT not set in .env file!")

PLOTS_SAVE_DIR="plots/"

""" # EXAMPLE .env file:
ALL_DATA_FILE_ROOT="./data/"
ENV_ACTIVATE_COMMAND="conda activate ba_venv5"
"""


""" 
# UNUSED doesnt alarm ever ?!
torch.autograd.set_detect_anomaly(True)  """


TRAIN = 0
TEST = 1

ENSEMBLES_SAVE_DIR = "./ensembles/"

# Saved to with torch.save()
ENSEMBLE_METADATA_FILENAME = "meta.pth"
ENSEMBLE_FULL_TEST_FILENAME = "full_test.pth"
MODEL_METADATA_TRAIN_SUFFIX = "_train_stats.pth"
MODEL_METADATA_TEST_SUFFIX = "_test_stats.pth"
ENSEMBLE_VIZ_CUSTOM_MODEL_LABELS = "custom_model_labels.txt"

# -- Tiled images:
data_tile_filename = '{}.tif'
SIZE = 128
# (<- 1 img now were 100/16 = 6.25 images before downscaling and where SIZE = 512)

# (Had before: 512 <- now too big since downscaled pixels x 10,
#   so 100 imgs before is 1 img now at that same size)

# SIZE = 256 (<- 1 img now were 100/4 = 12.5 images before downscaling and where SIZE = 512)

"""
# CRS origin is in bottom left!

gt_bounds: BoundingBox(left=272056.0, bottom=3289689.0, right=274440.0, top=3290290.0)
RGB:       BoundingBox(left=272056.0, bottom=3289689.0, right=272652.0, top=3290290.0)
...
"""

# Model CONSTANTS --------------------------------------------------
# LEARNING_RATE
# default AdamW learning rate is 1e-3 = 0.001
#   epoch 4 loss 2.999 meh
# has been ok:
#  -> 3e-4, without adaptation
#  -> 1e-4
# diverging loss (> 1000): 1e-1, 1e-2
# start higher to reduce
DEFAULT_LEARNING_RATE = 1e-3

MAX_NORM_FOR_CLIPPING = 2.0


# Train CONSTANTS --------------------------------------------------

# if MAX_EPOCHS is None then using EARLY_STOPPING
MAX_EPOCHS = None
EARLY_STOPPING_PATIENCE = 20


# RUNTIME CONFIG ---------------------------------------------------
DEVICE = "cpu"
BATCH_SIZE = 4
NUM_WORKERS = 2
PIN_MEMORY = False

# Check run using CUDA GPU acceleration
if torch.cuda.is_available():
    # A100 GPU -> 1:15 per epoch (new)
    # L4 GPU -> 3:00 per epoch (new)
    DEVICE = "cuda"
    # has been good: 20
    BATCH_SIZE = 20
    # NUM_WORKERS = 10 # 12
    # Get recommended number of workers based on CPU cores available - 1
    NUM_WORKERS = max(1, len(os.sched_getaffinity(0)) - 1)
    PIN_MEMORY = True

if torch.backends.mps.is_available():
    # Intel or Apple Silicon GPU
    DEVICE = "mps"
    BATCH_SIZE = 4
    NUM_WORKERS = 0 # 4

# PERSISTANT_WORKERS: keep worker pool alive across epochs instead of respawning it every epoch
PERSISTANT_WORKERS = NUM_WORKERS > 0

print("Running on, with workers:", DEVICE, NUM_WORKERS)

# Using ** shortshand
DATALOADER_ARGS = dict(batch_size=BATCH_SIZE, num_workers=NUM_WORKERS, pin_memory=PIN_MEMORY,
                        persistent_workers=PERSISTANT_WORKERS)


# Part of the Config: "Subjective" functions -------------------------
def normalize_band(band):
    # normalize -> [0, 1]
    band = np.asarray(band, dtype=np.float32)
    
    # ignore NaN
    band_min = np.nanmin(band)
    band_max = np.nanmax(band)
    
    return (band - band_min) / (band_max - band_min)


def clamp_percentile_band(band, percentile=2, lowClamp=True, highClamp=True):
    ''' Take only nth percentile range
    -> without resulting images can be close to fully black/white, since too extreme values '''
    low = percentile if lowClamp else 0
    high = (100 - percentile) if highClamp else 100

    lo, hi = np.percentile(band, (low, high))
    return np.clip(band, lo, hi)



# Some Helpers


# Remove dir contents, from all dir paths given
def dir_remove_all_files(dir_paths):
    if isinstance(dir_paths, str):
        dir_paths = [dir_paths]

    for dir_path in dir_paths:
        for filename in os.listdir(dir_path):
            file_path = os.path.join(dir_path, filename)
            if os.path.isfile(file_path):
                os.remove(file_path)

        print(f"Cleared dir: {dir_path}")


def get_unique_save_name(save_dir, base_name, extension=None):
    save_name = base_name + (extension if extension else "")

    counter = 1
    while os.path.exists(save_dir + save_name):
        save_name = f"{base_name}_{counter}" + (extension if extension else "")
        counter += 1

    return save_name


def print_memory_usage():
    # Process memory (RSS = actual RAM used, VMS = virtual memory)
    process = psutil.Process(os.getpid())
    mem_info = process.memory_info()
    rss_gb = mem_info.rss / 1024**3
    vms_gb = mem_info.vms / 1024**3

    # System memory (free/total/percentage)
    virtual_mem = psutil.virtual_memory()
    free_gb = virtual_mem.free / 1024**3
    total_gb = virtual_mem.total / 1024**3
    used_percent = virtual_mem.percent

    print(
        f"[RAM] Process: RSS={rss_gb:.2f} GB | VMS={vms_gb:.2f} GB | "
        f"System: Free={free_gb:.2f} GB / {total_gb:.2f} GB ({100 - used_percent:.1f}% free)"
    )

    print(f"GPU Memory: Allocated={torch.cuda.memory_allocated()/1024**3:.2f} GB | Reserved={torch.cuda.memory_reserved()/1024**3:.2f} GB")
