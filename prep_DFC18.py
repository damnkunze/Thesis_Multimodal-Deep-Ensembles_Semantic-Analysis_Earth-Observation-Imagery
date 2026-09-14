
import os
import torch

from src.base import *
from src.show import *

from src.prep_merge_cut_layer_norm import *
from src.prep_scale_editGT import *
from src.prep_tile import *

from src.DFC18_Dataset import * 

# ==============================================================================================
# =========================================== CONFIG ===========================================
# ==============================================================================================

TARGET_GSD = 0.5 # 0.5m
''' # GSDs:
- Mask: 0.5m
- RGB data: 0.05m
- HSI data: 1m
- LIDAR intensity data: 0.5m 
- LIDAR DSM data: 0.5m 
'''

FILE_ROOT       = DFC18Dataset.FILE_ROOT
TRAIN_FILE_ROOT = DFC18Dataset.DATA_FILES[TRAIN]
TEST_FILE_ROOT  = DFC18Dataset.DATA_FILES[TEST]

TRAIN_DATA_DIR = FILE_ROOT + 'ImageryAndTrainingGT/2018IEEE_Contest/Phase2/'
data_dir_RGB = TRAIN_DATA_DIR + 'Final RGB HR Imagery/'
data_dir_LIDAR = TRAIN_DATA_DIR + "Lidar GeoTiff Rasters/"

TRAIN_FILES_PREPROCESSED = TRAIN_FILE_ROOT + 'preprocessed_data_merged/'
TEST_FILES_PREPROCESSED = TEST_FILE_ROOT + 'preprocessed_data_merged/'

# BOTH TRAIN & TEST ====================================
# |- HSI
hsi_full_image = TRAIN_DATA_DIR + "FullHSIDataset/20170218_UH_CASI_S4_NAD83.pix"
hsi_spectral_only = TRAIN_FILES_PREPROCESSED + 'full_HSI_spectral_only.pix'
data_HSI_path_scaled = TRAIN_FILES_PREPROCESSED + 'full_HSI_data_GSD_0.5.pix'

# |- LIDAR (Intensities)
data_tifs_LIDAR_inten = ["Intensity_C1/UH17_GI1F051.tif", "Intensity_C2/UH17_GI2F051.tif", "Intensity_C3/UH17_GI3F051.tif"]
lidar_data_layered = TRAIN_FILES_PREPROCESSED + 'full_LIDAR_data_layered.tif'

# |- LIDAR (DSM)
data_tif_DSM_unnormalized = data_dir_LIDAR + "DSM_C12/UH17c_GEF051.tif"
data_tif_DSM = TRAIN_FILES_PREPROCESSED + 'full_DSM_data.tif'


# TRAIN =============================================
# |- DATA
# |-- RGB
data_tifs = ['UH_NAD83_272056_3289689.tif', 'UH_NAD83_272652_3289689.tif', 'UH_NAD83_273248_3289689.tif', 'UH_NAD83_273844_3289689.tif']
# Sort left to right using x param (UH_NAD83_272056_3289689 -> 272056)
data_tifs_sorted = sorted(data_tifs, key=lambda x: x.split('_')[2])

# |- GT
gt_file = TRAIN_DATA_DIR + 'TrainingGT/2018_IEEE_GRSS_DFC_GT_TR.tif'
# train image dimensions: W: 11920, H: 12020

# |- PRE SCALED (scaled before merging, faster)
RGB_data_scaled_dir = TRAIN_FILES_PREPROCESSED + 'RGB_input_imgs_GSD_0.5/'

# |- MERGED / CUT PART
## UNUSED merged_data_path = TRAIN_FILES_PREPROCESSED + 'merged_RGB_data_GSD_0.05.tif'
hsi_train_part_cut = TRAIN_FILES_PREPROCESSED + 'cut_HSI_train_part.pix'
lidar_train_part_cut = TRAIN_FILES_PREPROCESSED + 'cut_LIDAR_train_part.tif'
dsm_train_part_cut = TRAIN_FILES_PREPROCESSED + 'cut_DSM_train_part.tif'

# |- SCALED
merged_data_path_scaled = TRAIN_FILES_PREPROCESSED + 'merged_RGB_data_GSD_0.5.tif'

# |- TILED
# tile ground truth and data images in SIZE x SIZE images
# filenames: numbered eg: 0_data.tif, 0_gt.tif
tiled_RGB_data_dir      = TRAIN_FILE_ROOT + DFC18Dataset.MODALITIES_FOLDERS[DFC18Dataset.RGB]
tiled_HSI_data_dir      = TRAIN_FILE_ROOT + DFC18Dataset.MODALITIES_FOLDERS[DFC18Dataset.HSI]
tiled_LIDAR_data_dir    = TRAIN_FILE_ROOT + DFC18Dataset.MODALITIES_FOLDERS[DFC18Dataset.LIDAR]
tiled_DSM_data_dir      = TRAIN_FILE_ROOT + DFC18Dataset.MODALITIES_FOLDERS[DFC18Dataset.DSM]
tiled_gt_dir            = TRAIN_FILE_ROOT + DFC18Dataset.GT_FOLDER


# TEST ==============================================
# |- DATA
# All RGB files in data_dir_RGB (not sorted but rasterio does its job)
data_files_for_test_sorted = list(filter(lambda f: f.endswith('.tif'), os.listdir(data_dir_RGB)))

""" # Restrict to tiles that are not train
# first upper row then lower row
data_files_for_test_sorted = ['UH_NAD83_271460_3290290.tif', 'UH_NAD83_272056_3290290.tif', 'UH_NAD83_272652_3290290.tif', 'UH_NAD83_273248_3290290.tif', 'UH_NAD83_273844_3290290.tif', 'UH_NAD83_274440_3290290.tif', 'UH_NAD83_275036_3290290.tif',
                            'UH_NAD83_271460_3289689.tif', 'UH_NAD83_274440_3289689.tif', 'UH_NAD83_275036_3289689.tif'] """

# |- GT
test_gt_file = FILE_ROOT + "/TestingGT/TestingGT/Test_Labels.tif"

# |- PRE SCALED (scaled before merging, faster)
RGB_test_data_scaled_dir = TEST_FILES_PREPROCESSED + 'RGB_input_imgs_GSD_0.5/'

# |- MERGED / CUT PART
## UNUSED merged_data_path_for_test = TEST_FILES_PREPROCESSED + 'merged_RGB_data_test_GSD_0.05.tif'
## UNUSED hsi_test_part_cut = TEST_FILES_PREPROCESSED + 'cut_HSI_test_part.pix'
## UNUSED lidar_test_part_cut = TEST_FILES_PREPROCESSED + 'cut_LIDAR_test_part.tif'

# |- SCALED
merged_data_path_for_test_scaled = TEST_FILES_PREPROCESSED + 'merged_RGB_data_test_GSD_0.5.tif'

# |- TILED
tiled_test_RGB_data_dir     = TEST_FILE_ROOT + DFC18Dataset.MODALITIES_FOLDERS[DFC18Dataset.RGB]
tiled_test_HSI_data_dir     = TEST_FILE_ROOT + DFC18Dataset.MODALITIES_FOLDERS[DFC18Dataset.HSI]
tiled_test_LIDAR_data_dir   = TEST_FILE_ROOT + DFC18Dataset.MODALITIES_FOLDERS[DFC18Dataset.LIDAR]
tiled_test_DSM_data_dir     = TEST_FILE_ROOT + DFC18Dataset.MODALITIES_FOLDERS[DFC18Dataset.DSM]
tiled_test_gt_dir           = TEST_FILE_ROOT + DFC18Dataset.GT_FOLDER


# Create any nonexisitng folders
# And make sure all DIR VARIABLE values end with "/"

all_folders = [FILE_ROOT, TRAIN_FILE_ROOT, TRAIN_FILES_PREPROCESSED, tiled_RGB_data_dir, tiled_LIDAR_data_dir, tiled_DSM_data_dir,
            RGB_data_scaled_dir, tiled_HSI_data_dir, tiled_gt_dir, TEST_FILE_ROOT, TEST_FILES_PREPROCESSED, RGB_test_data_scaled_dir,
            tiled_test_RGB_data_dir, tiled_test_HSI_data_dir, tiled_test_LIDAR_data_dir, tiled_test_DSM_data_dir,
            tiled_test_gt_dir]

if os.path.exists(FILE_ROOT):
    for folder in all_folders:
        if not folder.endswith("/"):
            raise ValueError(f"Folder path variable should end with '/': {folder}")

        if not os.path.exists(folder):
            os.makedirs(folder)
            print("Created dir:", folder)

else: 
    raise EnvironmentError("FILE_ROOT does not exist:", FILE_ROOT)

# All data images
in_image_paths_train = [merged_data_path_scaled, hsi_train_part_cut, lidar_train_part_cut,  dsm_train_part_cut]
in_image_paths_test  = [merged_data_path_for_test_scaled, data_HSI_path_scaled, lidar_data_layered, data_tif_DSM]

out_images_dirs_train = [tiled_RGB_data_dir, tiled_HSI_data_dir, tiled_LIDAR_data_dir, tiled_DSM_data_dir]
out_images_dirs_test  = [tiled_test_RGB_data_dir, tiled_test_HSI_data_dir, tiled_test_LIDAR_data_dir, tiled_test_DSM_data_dir]


# Attach GT as last image
all_in_paths_train  = in_image_paths_train   + [gt_file]
all_in_paths_test   = in_image_paths_test    + [test_gt_file]
all_out_dirs_train  = out_images_dirs_train  + [tiled_gt_dir]
all_out_dirs_test   = out_images_dirs_test   + [tiled_test_gt_dir]


if __name__ == "__main__":

    # ==============================================================================================
    # ============================================ PREP ============================================
    # ==============================================================================================

    print("=" * 50)
    print("Preparing DFC18 Dataset...")
    print("=" * 50)

    # 1. SCALE + NORMALIZE (HSI, RGB) --------------------------------------------------------------
    ## Downscale merged data images to GSD = 0.5m
    ## Normalize HSI, RGB (LIDAR is done in layering)

    ## HSI full: Both Train & Test
    ## Drop non-spectral bands 49 (off-nadir angle) & 50 (DEM height) first
    remove_bands_from_image(hsi_full_image, hsi_spectral_only, bands_to_remove=[49, 50])
    scale_to_GSD_opt_normalize(hsi_spectral_only, data_HSI_path_scaled, TARGET_GSD, is_HSI=True)


    ## RGB
    ## Scale each un-merged input RGB image individually
    # Clear before
    to_clear = [RGB_data_scaled_dir, RGB_test_data_scaled_dir]
    dir_remove_all_files(to_clear)

    ### Train data
    scale_each_image_to_GSD(data_dir_RGB, data_tifs_sorted, RGB_data_scaled_dir, TARGET_GSD, is_RGB=True)

    ### Test data
    scale_each_image_to_GSD(data_dir_RGB, data_files_for_test_sorted, RGB_test_data_scaled_dir, TARGET_GSD, is_RGB=True)

    # 2. MERGE, LAYER, CUT -----------------------------------------------------

    ## MERGE (RGB)

    ### Train data 
    data_tifs_sorted_paths = [RGB_data_scaled_dir + f for f in os.listdir(RGB_data_scaled_dir)]

    merge_data_tiles(data_tifs_sorted_paths, merged_data_path_scaled)
    inspect_image_meta(merged_data_path_scaled)

    ### Test data
    data_tifs_sorted_paths_for_test = [RGB_test_data_scaled_dir + f for f in os.listdir(RGB_test_data_scaled_dir)]

    merge_data_tiles(data_tifs_sorted_paths_for_test, merged_data_path_for_test_scaled)
    inspect_image_meta(merged_data_path_for_test_scaled)


    ## LAYER (LIDAR) + NORMALIZE, CORRECT NoData
    # Normalize & Replace NoData Value with `None`
    lidar_paths = [data_dir_LIDAR + p for p in data_tifs_LIDAR_inten]

    layer_data_images(lidar_paths, lidar_data_layered, new_no_data=0) 
    check_img_normalization(lidar_data_layered)


    ## (DSM) NORMALIZE, CORRECT
    # Normalize & Replace NoData Value with `None`
    normalize_and_correct(data_tif_DSM_unnormalized, data_tif_DSM, new_no_data=0)
    # check_img_normalization(data_tif_DSM)

    ## CUT Train Part (HSI, LIDAR, DSM)

    ## Test is full input image (including Train Part), just not gonna be evaluated on it since Test GT doesnt include Train part

    ### HSI
    cut_image_with_bounds_from_other(data_HSI_path_scaled, gt_file, hsi_train_part_cut)

    ### LIDAR
    cut_image_with_bounds_from_other(lidar_data_layered, gt_file, lidar_train_part_cut)

    ### DSM
    cut_image_with_bounds_from_other(data_tif_DSM, gt_file, dsm_train_part_cut)


    # 3. CHECK PREPROCESSED DATA -----------------------------------------------------

    # Check Train & Test Data Images
    for path in in_image_paths_train + in_image_paths_test:
        inspect_image_meta(path)
        check_img_normalization(path)
        # show_full_image(path)

    # Check GT
    for gt_path in [gt_file, test_gt_file]:
        inspect_image_meta(gt_path)
        check_img_normalization(gt_path, min_val=0, max_val=DFC18Dataset.NUM_CLASSES - 1)


    # 4. TILE ALL ------------------------------------------------------------
    ## Clear before Tile
    dir_remove_all_files(all_out_dirs_train) 
    dir_remove_all_files(all_out_dirs_test) 

    ### TRAIN SET
    tile_images_aligned_from_all_sensors(all_in_paths_train, all_out_dirs_train)

    ### TEST SET
    tile_images_aligned_from_all_sensors(all_in_paths_test, all_out_dirs_test)
    # Results ==============================================================

    # Get lowest amount of tiles in tiles files 
    # (eg HSI always has x2)
    NUM_TRAIN_TILES = min([len(os.listdir(f)) for f in all_out_dirs_train]) 
    NUM_TEST_TILES  = min([len(os.listdir(f)) for f in all_out_dirs_test]) 

    # Check Tiled data
    check_num = 5

    rand_indxs_train = [0] # list(np.random.choice(range(NUM_TRAIN_TILES), size=check_num, replace=False))
    rand_indxs_test  = list(np.random.choice(range(NUM_TEST_TILES),  size=check_num, replace=False))

    # Train data
    print(f"Showing {DFC18Dataset.NAME} train tile at index/indices {rand_indxs_train}:")
    show_dataset_images(DFC18Dataset, TRAIN, rand_indxs_train)