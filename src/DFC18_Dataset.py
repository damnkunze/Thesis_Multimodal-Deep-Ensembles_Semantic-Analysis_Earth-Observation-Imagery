import os
import torch
import rasterio as rio

from torch.utils.data.dataset import Dataset
from torch.utils.data import DataLoader
from torchvision.transforms import v2 as transforms
from torchvision.tv_tensors import Image, Mask

from src.base import *
from src.Channels import *
from src.class_counts_weights import *


def _identity_transform(*x):
    """Passthrough transform (picklable version of lambda x: x)"""
    return x


# TODO: Abstract into a class DatasetManager(Dataset) 
#  to not double code in DFC18Dataset and in MDAS_Dataset


class DFC18Dataset(Dataset):
    NAME = "DFC18Dataset"
    NAME_SHORT = "DFC18"

    # Root for this dataset
    FILE_ROOT = ALL_DATA_FILE_ROOT + "BA_DFC2018/"

    DATA_FILES = {
        TRAIN: FILE_ROOT + 'forTrain/',
        TEST: FILE_ROOT + 'forTest/'
    }

    # Modalities
    RGB = 0
    HSI = 1
    LIDAR = 2
    DSM = 3

    # Always: Modalities have to be in the same order
    MODALITIES = [RGB, HSI, LIDAR, DSM]
    MODALITIES_NAMES = ['RGB', 'HSI', 'LIDAR INT', 'LIDAR DSM']

    MODALITIES_FOLDERS = { RGB: "data_RGB/", HSI: "data_HSI/", 
                            LIDAR: "data_LIDAR/", DSM: "data_DSM/"}
    GT_FOLDER = "gt/"

    MODALITIES_CHANNEL_COUNTS = {RGB: 3, HSI: 48, LIDAR: 3, DSM: 1}
    MODALITIES_TOTAL_CHANNELS = sum(MODALITIES_CHANNEL_COUNTS.values())

    CLASS_NAMES = ['Unclassified', 'Healthy grass', 'Stressed grass', 'Artificial turf','Evergreen trees', 'Deciduous trees', 'Bare earth', 'Water','Residential buildings', 'Non-residential buildings', 'Roads','Sidewalks', 'Crosswalks', 'Major thoroughfares', 'Highways','Railways', 'Paved parking lots', 'Unpaved parking lots', 'Cars','Trains', 'Stadium seats']
    
    # 20 data classes + class 0 for "unclassified" (gonna be ignored later)
    NUM_CLASSES = len(CLASS_NAMES)
    
    # All numbers with name (0, 'Unclassified'), ...
    CLASS_NAMES_NUMBERED = list(zip(range(NUM_CLASSES), CLASS_NAMES))
    
    # [1 - 20] exluding unclassified 0
    CLASS_NUMBERS = list(range(1, NUM_CLASSES)) 

    # Count pixels, takes long on CPU
    CLASS_COUNTS_PATH = FILE_ROOT + f"CLASS_COUNTS.pth"
    CLASS_COUNTS = None

    # For Class Balancing:
    # Reload CLASS_COUNTS
    if os.path.exists(CLASS_COUNTS_PATH):
        CLASS_COUNTS = torch.load(CLASS_COUNTS_PATH, weights_only=False, map_location=DEVICE)

        CLASS_WEIGHTS = get_class_weigths_mean_count_by_count(CLASS_COUNTS).to(DEVICE)

        # Alternative: But too heavy weights for small classes
        # CLASS_WEIGHTS = get_class_weigths_inv_freq(CLASS_COUNTS).to(DEVICE)


    # For visualization
    tab20_cmap = plt.colormaps['tab20']
    tab20b_cmap = plt.colormaps['tab20b']
    all_colors = np.vstack((tab20_cmap(np.linspace(0, 1, 20)), tab20b_cmap(np.linspace(0, 1, 20))))
    CMAP_GT = colors.ListedColormap(all_colors[:21])
    CMAP_GT.colors[0] = (1, 1, 1, 1) # class zero = white

    VIS_TILES_PER_ROW = 65
    VIS_MODALITIES_NEED_STRETCHING = [HSI, LIDAR, DSM]

    def __init__(self, train_or_test, channels_modulation=None, augment=True):
        """
        Assuming data files are numbered from 0 to self.len - 1

        Expects to be set:
        SIZE
        data_tile_filename
        """

        root_path = self.DATA_FILES[train_or_test]

        self.data_paths = [root_path + self.MODALITIES_FOLDERS[m] for m in self.MODALITIES]
        self.masks_path = root_path + "gt/"

        self.channels_modulation = channels_modulation

        # If Modulation not given, just return normal data
        if self.channels_modulation is not None:
            # Has to be size of MODALITIES_TOTAL_CHANNELS
            if len(self.channels_modulation) != self.MODALITIES_TOTAL_CHANNELS:
                raise ValueError(f"channels_modulation should be (C, K) with C = {self.MODALITIES_TOTAL_CHANNELS}, but C = {len(self.channels_modulation)}")
       
        # Count is min number of datapoints per modalities
        self.data_imgs_count = min([len(os.listdir(dp)) for dp in self.data_paths])
        self.masks_count = len(os.listdir(self.masks_path))

        self.len = min(self.data_imgs_count, self.masks_count)
        
        if self.len == 0:
            raise ValueError("No tiles found. Make sure to run `prep` at least once before")

        # Dont augment testset
        if augment:
            # Data Augmentation for more variation between epochs
            self.transform = transforms.Compose([
                transforms.RandomHorizontalFlip(p=0.5),
                transforms.RandomVerticalFlip(p=0.5)])
            # RandomRotateTransform() (code in backup)

        else:
            self.transform = _identity_transform
        
        # Calculate trainset CLASS_COUNTS
        # TODO Refactor to not need re-run
        if train_or_test == TRAIN \
                and self.CLASS_COUNTS is None:
            # Count pixels, takes long
            # Save after to run only once
            print("\nCounting pixels for CLASS_COUNTS & CLASS_WEIGHTS, this will take a moment...\n")
            train_dataloader = DataLoader(self, **DATALOADER_ARGS, shuffle=False)
            self.CLASS_COUNTS = get_class_counts(train_dataloader, self.NUM_CLASSES, DEVICE) 

            torch.save(self.CLASS_COUNTS, self.CLASS_COUNTS_PATH)

            raise(Exception("\n\n\nPLEASE RE-RUN. Classes have been counted and counts saved.\n"))


    def __getitem__(self, index):
        # -- Images
        data_tensors = []

        # Get each modality the same way
        for data_path in self.data_paths:
            data_tile = data_path + data_tile_filename.format(index)

            with rio.open(data_tile) as img_src:
                img = img_src.read() # C x H x W

            # Images are already saved normalized, so no preprocessing here
            img_tensor = torch.from_numpy(img).float()

            # Mark as image for transform
            data_tensors.append(Image(img_tensor))


        # -- GT
        gt_tile = self.masks_path + data_tile_filename.format(index)

        if os.path.exists(gt_tile):
            with rio.open(gt_tile) as mask_src:
                mask = mask_src.read() # (1, H, W)

        # if gt tile does not exist, fake an empty one
        else:
            mask = np.zeros((1, SIZE, SIZE), dtype=np.uint8)

        # mask as class ints (has to be long for CrossEntropyLoss)
        mask_tensor = torch.from_numpy(mask).long().squeeze() # H, W

        ''' # Test Show before transform
        convert_and_plot_img_data(data_tensors, mask=mask_tensor, showColorbar=False) '''

        # -- Transform
        # Pass all data and mask together for same random transformation
        # (using spread operator *)
        transformed = self.transform(*data_tensors, Mask(mask_tensor)) # M mods + 1 mask, C, H, W
        transformed_data = list(transformed[:-1]) # M mods, C, H, W
        transformed_mask = transformed[-1] # H, W

        ''' # Test Show
        convert_and_plot_img_data(transformed_data, mask=transformed_mask, showColorbar=False) '''

        # -- Concatenate modalities to be passed together to NN
        # (has to be as list)
        combined_data = torch.cat(transformed_data)
        # C (concatenated channels), H, W

        # -- Channel Modulation
        # Multiply all channels with channels_modulation
        # ((C, H, W).T @ (C, K)).T -> (K, H, W)
        # Instead of .T use permute() since is 3d tensor
        if self.channels_modulation is not None:
            combined_data = Channel_Modulation.apply_modulation(combined_data, self.channels_modulation)

        # Check shapes
        if len(combined_data.shape) != 3 or combined_data.shape[1] != SIZE \
                or combined_data.shape[2] != SIZE:
            raise ValueError(f"Imgs at index {index} have inconsistent shape: {combined_data.shape}")

        if transformed_mask.shape != (SIZE, SIZE):
            raise ValueError(f"Mask at index {index} has inconsistent shape: {transformed_mask.shape}")

        return combined_data, transformed_mask


    def __len__(self):
        return self.len
