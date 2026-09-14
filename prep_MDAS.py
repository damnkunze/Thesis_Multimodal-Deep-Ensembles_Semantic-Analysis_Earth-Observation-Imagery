import os
import torch
import shutil

from src.base import *
from src.show import *

from src.prep_merge_cut_layer_norm import *
from src.prep_scale_editGT import *
from src.prep_tile import *

from src.MDAS_Dataset import * 

# ==============================================================================================
# =========================================== CONFIG ===========================================
# ==============================================================================================

# Take 50% tiles and move to Train Folders
train_split_p = 0.5

""" # OLD 
# Take 20% and move to Train Folders
# # train_split_p = 0.2 """

test_split_p = 1 - train_split_p

TARGET_GSD = 2.2
# Is GSD of GT

FILE_ROOT       = MDAS_Dataset.FILE_ROOT
TRAIN_FILE_ROOT = MDAS_Dataset.DATA_FILES[TRAIN]
TEST_FILE_ROOT  = MDAS_Dataset.DATA_FILES[TEST]

TRAIN_DATA_DIR = FILE_ROOT + 'Augsburg_data_4_publication/entire_city/'
FILES_PREPROCESSED = FILE_ROOT + 'preprocessed_data/'

TRAIN_FILE_ROOT = FILE_ROOT + 'forTrain/'
TEST_FILE_ROOT = FILE_ROOT + 'forTest/'


# BOTH TRAIN & TEST ====================================
# | DATA

# |-- RGB
data_RGB_unscaled = TRAIN_DATA_DIR + '3K_RGB.tif'
data_RGB_4_bands  = FILES_PREPROCESSED + 'RGB_4_bands.tif'
data_RGB          = FILES_PREPROCESSED + 'RGB.tif'

# |-- HSI
data_HSI_unnormalized = TRAIN_DATA_DIR + "EeteS_EnMAP_10m.tif"
data_HSI              = FILES_PREPROCESSED + 'HSI.tif'

# |-- MSI
data_MSI_unscaled = TRAIN_DATA_DIR + 'Sentinel-2.tif'
data_MSI          = FILES_PREPROCESSED + 'MSI.tif'

# |-- SAR
data_SAR_unscaled = TRAIN_DATA_DIR + 'Sentinel-1.tif'
data_SAR          = FILES_PREPROCESSED + 'SAR.tif'

# |-- DSM
data_DSM_unscaled = TRAIN_DATA_DIR + '3K_DSM.tif'
data_DSM          = FILES_PREPROCESSED + 'DSM.tif'

# | GT
gt_file_landuse = TRAIN_DATA_DIR + 'OSM_label/osm_landuse.tif'
gt_file_water   = TRAIN_DATA_DIR + 'OSM_label/osm_water.tif'
# UNUSED
# gt_file_buildings = TRAIN_DATA_DIR + 'OSM_label/osm_buildings.tif'

gt_file_combined     = FILES_PREPROCESSED + 'GT_combined.tif'
gt_file_cut          = FILES_PREPROCESSED + 'GT_cut.tif'
gt_file              = FILES_PREPROCESSED + 'GT.tif'


# (Train / Test Split happens after tiling)

# | TILED
# tile ground truth and data images in SIZE x SIZE images
# filenames: numbered eg: 0_data.tif, 0_gt.tif
from src.MDAS_Dataset import *

tiled_RGB_data_dir = TRAIN_FILE_ROOT + MDAS_Dataset.MODALITIES_FOLDERS[MDAS_Dataset.RGB]
tiled_HSI_data_dir = TRAIN_FILE_ROOT + MDAS_Dataset.MODALITIES_FOLDERS[MDAS_Dataset.HSI]
tiled_MSI_data_dir = TRAIN_FILE_ROOT + MDAS_Dataset.MODALITIES_FOLDERS[MDAS_Dataset.MSI]
tiled_SAR_data_dir = TRAIN_FILE_ROOT + MDAS_Dataset.MODALITIES_FOLDERS[MDAS_Dataset.SAR]
tiled_DSM_data_dir = TRAIN_FILE_ROOT + MDAS_Dataset.MODALITIES_FOLDERS[MDAS_Dataset.DSM]
tiled_gt_dir       = TRAIN_FILE_ROOT + MDAS_Dataset.GT_FOLDER


# TEST ==============================================

# | TILED
tiled_test_RGB_data_dir = TEST_FILE_ROOT + MDAS_Dataset.MODALITIES_FOLDERS[MDAS_Dataset.RGB]
tiled_test_HSI_data_dir = TEST_FILE_ROOT + MDAS_Dataset.MODALITIES_FOLDERS[MDAS_Dataset.HSI]
tiled_test_MSI_data_dir = TEST_FILE_ROOT + MDAS_Dataset.MODALITIES_FOLDERS[MDAS_Dataset.MSI]
tiled_test_SAR_data_dir = TEST_FILE_ROOT + MDAS_Dataset.MODALITIES_FOLDERS[MDAS_Dataset.SAR]
tiled_test_DSM_data_dir = TEST_FILE_ROOT + MDAS_Dataset.MODALITIES_FOLDERS[MDAS_Dataset.DSM]
tiled_test_gt_dir       = TEST_FILE_ROOT + MDAS_Dataset.GT_FOLDER


# Create any nonexisitng folders
# And make sure all DIR VARIABLE values end with "/"

all_folders = [FILE_ROOT, TRAIN_FILE_ROOT, FILES_PREPROCESSED, tiled_RGB_data_dir, 
                tiled_HSI_data_dir, tiled_MSI_data_dir, tiled_SAR_data_dir, tiled_DSM_data_dir, tiled_gt_dir, 
                TEST_FILE_ROOT, tiled_test_RGB_data_dir, tiled_test_HSI_data_dir, 
                tiled_test_MSI_data_dir, tiled_test_SAR_data_dir, tiled_test_DSM_data_dir, tiled_test_gt_dir]

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
in_image_paths = [data_RGB, data_HSI, data_MSI, data_SAR, data_DSM]

out_images_dirs_train = [tiled_RGB_data_dir, tiled_HSI_data_dir, tiled_MSI_data_dir, tiled_SAR_data_dir, tiled_DSM_data_dir]
out_images_dirs_test = [tiled_test_RGB_data_dir, tiled_test_HSI_data_dir, tiled_test_MSI_data_dir, tiled_test_SAR_data_dir, tiled_test_DSM_data_dir]

# Attach GT as last image
all_in_paths         = in_image_paths         + [gt_file]
all_out_dirs_train   = out_images_dirs_train  + [tiled_gt_dir]
all_out_dirs_test    = out_images_dirs_test   + [tiled_test_gt_dir]

# In Tiling, First use tmp names
data_tile_filename_tmp = f"tmp_" + data_tile_filename # "tmp_{}.tif"

if __name__ == "__main__":

    # ==============================================================================================
    # ============================================ PREP ============================================
    # ==============================================================================================
    
    print("=" * 50)
    print("Preparing MDAS Dataset...")
    print("=" * 50)

    
    # 1. EDIT GT

    ## COMBINE GT maps
    # landuse + water

    # In order of precedence (later: on top of before), ignoring no-data.
    combine_GTs([gt_file_landuse, gt_file_water], gt_file_combined, ignore_class=-2147483647)


    ## CUT 1px thats too much
    ### All others, e.g. MSI is correct size
    cut_image_with_bounds_from_other(gt_file_combined, data_MSI_unscaled, gt_file_cut)


    ## CLEAN CLASSES
    ## Convert values, skipping & merging some

    classes_conversion = { 
        # osm_landuse.tif
        -2147483647: 0,     # no label
        7201: 1,            # forest
        7202: 0,            
        7203: 2,            # residential
        7204: 3,            # commercial
        7205: 4,            # low plant
        7206: 0,            
        7207: 5,            # allotments
        7208: 4,            # low plant
        7209: 3,            # commercial
        7211: 6,            # recreation ground
        7212: 3,            # commercial
        7214: 0,            
        7215: 0,            
        7217: 4,            # low plant
        7218: 4,            # low plant
        7219: 4,            # low plant
        
        # osm_water.tif
        8200: 7,            # water
        8202: 7,            # water
        8221: 7             # water
    }

    convert_GT_classes(gt_file_cut, gt_file, classes_conversion)


    # 2. SCALE + NORMALIZE (ALL) --------------------------------------------------------------
    ## Up-/ Downscale data images to TARGET_GSD

    ## RGB
    scale_to_GSD_opt_normalize(data_RGB_unscaled, data_RGB_4_bands, TARGET_GSD, is_RGB=True)

    # MSI
    scale_to_GSD_opt_normalize(data_MSI_unscaled, data_MSI, TARGET_GSD, is_MSI=True)

    # SAR (also clamp)
    scale_to_GSD_opt_normalize(data_SAR_unscaled, data_SAR, TARGET_GSD, is_SAR=True, 
                        clamp_percentile=0.5)


    ## DSM
    scale_to_GSD_opt_normalize(data_DSM_unscaled, data_DSM, TARGET_GSD, is_DSM=True, 
                                no_data_val=-32767)
    check_img_normalization(data_DSM)


    # HSI
    ## Dont scale <- too big (24GB)
    normalize_all_bands_file(data_HSI_unnormalized, data_HSI)

    # Remove Band: Correct RGB 4 bands to 3 bands
    remove_bands_from_image(data_RGB_4_bands, data_RGB, bands_to_remove=[4])

    # 3. CHECK PREPROCESSED DATA -----------------------------------------------------

    # Check Train & Test Data Images
    for path in in_image_paths:
        inspect_image_meta(path)
        check_img_normalization(path)
        # show_full_image(path)

    # Check GT
    inspect_image_meta(gt_file)
    check_img_normalization(gt_file, min_val=0, max_val=MDAS_Dataset.NUM_CLASSES - 1)
   


    # 4. TILE ALL ------------------------------------------------------------

    # First save both train & test in test folders

    # Clear before Tile
    dir_remove_all_files(all_out_dirs_train)

    ### FULL SET
    ### |- RGB, HSI, LIDAR, GT
    # Dont load full HSI img into RAM
    tile_images_aligned_from_all_sensors(all_in_paths, all_out_dirs_train, read_from_disk=[MDAS_Dataset.HSI], 
                                            filename_format=data_tile_filename_tmp)
   

    # Get lowest amount of tiles in tiles files 
    # (eg HSI always has x2)
    NUM_ALL_TILES = min([len(os.listdir(f)) for f in all_out_dirs_train]) 

    NUM_TEST_TILES = int(test_split_p * NUM_ALL_TILES)
    NUM_TRAIN_TILES = int(train_split_p * NUM_ALL_TILES)


    # 5. SPLIT TRAIN / TEST -----------------------------------------------------
    
    # 1 Tile into train folder with tmp filename
    # 2 Copy all tiles to test folder with final name
    # 3 Remove files selected for test from train folder
    # 4 Remove gt file from test, for tiles that are selected for train
    #   -> This way during test can still predict on train tiles but will not influence metrics
    #      - The dataloader when not finding the GT file creates an empty matrix as GT
    #   -> Relevant for showing full area
    # 5 Rename train tiles to be numbered from 0 to num_tiles - 1 with final filename


    dir_remove_all_files(all_out_dirs_test) 
    # -> Instead: all_out_dirs_test

    all_tile_indices = range(NUM_ALL_TILES)

    print(f"Splitting Train / Test ({train_split_p} / {test_split_p}) -> {NUM_TRAIN_TILES} / {NUM_TEST_TILES} tiles.")

    # For reproducible train / test split 
    #  -> reproducible CLASS_COUNTS and CLASS_WEIGHTS
    # When changing this, recalculate in MDAS_Dataset.py
    SEED_TRAIN_TEST_SPLIT = 42
    NP_RAND_TRAIN_TEST_SPLIT = np.random.default_rng(seed=SEED_TRAIN_TEST_SPLIT)

    test_tiles_indices = NP_RAND_TRAIN_TEST_SPLIT.choice(all_tile_indices, size=NUM_TEST_TILES, replace=False)
    
    # 2 Copy all tiles to test folders with final name
    for tile_idx in all_tile_indices:
        for train_dir, test_dir in zip(all_out_dirs_train, all_out_dirs_test):
            src = train_dir + data_tile_filename_tmp.format(tile_idx)
            dst = test_dir +  data_tile_filename.format(tile_idx)
            shutil.copyfile(src, dst)
    
    # 3 Remove files selected for test from train folders
    for tile_idx in test_tiles_indices:
        for train_dir in all_out_dirs_train:
            os.remove(train_dir + data_tile_filename_tmp.format(tile_idx))

    train_tiles_indices = np.setdiff1d(all_tile_indices, test_tiles_indices)

    # 4 Remove gt file from test, for tiles that are selected for train
    for tile_idx in train_tiles_indices:
        os.remove(tiled_test_gt_dir + data_tile_filename.format(tile_idx))

    
    """ # OLD way: but original order of test tiles is lot
    for tile_idx in test_tiles_indices:
        for train_dir, test_dir in zip(all_out_dirs_train, all_out_dirs_test):
            os.rename(train_dir + data_tile_filename_tmp.format(tile_idx), test_dir + data_tile_filename_tmp.format(tile_idx))
    """

    print(f"Moved {NUM_TEST_TILES} tiles to Test folders.")


    # 5 Rename train tiles to be numbered from 0 to num_tiles - 1 with final filename
    def rename_tiles_to_sequential_numbers(tiles_dirs):
        for tiles_dir in tiles_dirs:

            tile_files = os.listdir(tiles_dir)
            for new_idx, tile_file in enumerate(tile_files):
                old_path = tiles_dir + tile_file
                new_path = tiles_dir + data_tile_filename.format(new_idx)
                os.rename(old_path, new_path)

    rename_tiles_to_sequential_numbers(all_out_dirs_train)
    
    # DONT FOR TEST, keep in original order

    print("Renamed train tiles to be numbered from 0 to num_tiles - 1.")



    # Results ==============================================================

    """ # 6. CHECK TILES ------------------------------------------------------------
    check_num = 5

    rand_indxs_train = list(np.random.choice(range(NUM_TRAIN_TILES), size=check_num, replace=False))
    rand_indxs_test  = list(np.random.choice(range(NUM_TEST_TILES),  size=check_num, replace=False))
    # At edge: 

    # Train data
    print(f"Showing {MDAS_Dataset.NAME} train tile at index/indices {rand_indxs_train}:")
    show_dataset_images(MDAS_Dataset, TRAIN, rand_indxs_train)
    """

    """ # Test data
    for idx in rand_indxs_test:
        show_dataset_images(MDAS_Dataset, TEST, idx) """


    """ show_tile_file_at_index(idx, [tiled_test_RGB_data_dir],
                    mask_dir=tiled_test_gt_dir, showColorbar=(idx == rand_indxs_test[0]))

    # show Colorbar only for first img
    ''' show_tile_file_at_index(idx, [tiled_test_RGB_data_dir, tiled_test_HSI_data_dir],
                            mask_dir=tiled_test_gt_dir, showColorbar=(idx == rand_indxs_test[0])) ''' """
