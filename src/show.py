import matplotlib.colors as mcolors

from src.base import *

from src.DFC18_Dataset import *
from src.MDAS_Dataset import *

# RAW INPUT IMAGES -----------------------------------------------

def show_full_image(path, cmap=None):
    with rio.open(path) as src:
        fig, ax = plt.subplots(figsize=(8, 8))
        ax.set_xticks([])
        ax.set_yticks([])

        if cmap is None: 
                plot.show(src, ax=ax)
        else: 
            plot.show(src, ax=ax, cmap=cmap)

        plt.show()

def inspect_image_meta(path):
    with rio.open(path) as src:
        print("-" * 10)
        print(path.split("/")[-1])
        print("width", src.width, "height", src.height)
        print("res", src.res)
        print("bounds", src.bounds)

def get_image_shape(path):
    with rio.open(path) as src:
        img = src.read()
        print("shape", img.shape)


def check_img_normalization(path_or_img, moreVals=0, min_val=0, max_val=1):

    if isinstance(path_or_img, str):
        with rio.open(path_or_img) as src:
            img = src.read()

    else:
        img = path_or_img

    if moreVals: return print(np.unique(img))

    # Ignore NoData: None vals
    img = img[~np.isnan(img)]

    img_min = np.min(img)
    img_max = np.max(img)

    # print filename    
    print("\n", path_or_img.split("/")[-1] if isinstance(path_or_img, str) else "Image data")
    print("NORMALIZATION:\n", "min", img_min, "max", img_max)
    print("->", ("good" if (img_min >= min_val and img_max <= max_val) else "NOT NORMALIZED"))




# HSI Show funcs
def lossy_convert_to_RGB(bands):
    # Select 3 random bands to show as RGB
    max_band = bands.shape[0] - 1
    select_bands = torch.randint(low=0, high=max_band, size=(3,))
    print(f"Showing random bands: {select_bands.tolist()} from (0 - {max_band})")

    data = torch.stack([bands[select_bands[i]] for i in range(len(select_bands))])

    return data

    ''' # UNUSED already normalized
    data = np.array([normalize_hsi(bands[select_bands[i]]) for i in range(len(select_bands))])
    data = np.stack(data)
    return data '''


def show_raw_hsi_data(hsi_data):
    data = lossy_convert_to_RGB(hsi_data)

    fig, ax = plt.subplots(figsize=(12, 12))
    ax.set_xticks([])
    ax.set_yticks([])
    plot.show(data, ax=ax)
    plt.show()

    ''' # faster: read only needed bands
    # bands in range 1 - 50
    band1 = normalize_hsi(src_data.read(1))
    band2 = normalize_hsi(src_data.read(25))
    band3 = normalize_hsi(src_data.read(50))
    plot.show(np.stack((band1, band2, band3))), ax=ax)
    plt.show()
    '''

def show_hsi_image_file(hsi_image):
    with rio.open(hsi_image) as src_data:
        show_raw_hsi_data(src_data.read())


# show images funcs for testing
def show_rgb_image_file(rgb_image):
    with rio.open(rgb_image) as src_data:
        fig, ax = plt.subplots(figsize=(8, 8))
        ax.set_xticks([])
        ax.set_yticks([])
        # have to be explicitly .read() to be displayed as RGB 
        plot.show(src_data.read(), ax=ax)
        plt.show()


def stretch_to_0_1(img, per_band=True):
    if per_band:
        img_np = img.numpy()
        band_min = np.nanmin(img_np, axis=(1, 2), keepdims=True)  # (C, 1, 1)
        band_max = np.nanmax(img_np, axis=(1, 2), keepdims=True)

        band_min = torch.from_numpy(band_min)
        band_max = torch.from_numpy(band_max)

        denom = (band_max - band_min).clamp(min=1e-8)
        print(f"Stretching each band from [{band_min.flatten().tolist()}] to [{band_max.flatten().tolist()}] to [0, 1]")
        return (img - band_min) / denom
    else:
        img_np = img.numpy()
        lo = np.nanmin(img_np)
        hi = np.nanmax(img_np)
        if hi == lo:
            return torch.zeros_like(img)
        print(f"Stretching from [{lo:.4f}, {hi:.4f}] to [0, 1]")
        return (img - lo) / (hi - lo)


def plot_img_data(all_imgs_data, Dataset_class, masks=None, pred=None, showColorbar=True, save=False):
    num_rows = len(all_imgs_data)
    num_cols = len(all_imgs_data[0])
    num_cols += masks is not None and any(m is not None for m in masks)
    num_cols += pred is not None
    num_cols = max(num_cols, 2)

    fig, axs = plt.subplots(num_rows, num_cols, figsize=(16, 2 * num_rows), squeeze=False)

    for ax in axs.ravel():
        ax.set_xticks([])
        ax.set_yticks([])

    num_classes = Dataset_class.NUM_CLASSES
    bounds = np.arange(num_classes + 1) - 0.5
    norm = mcolors.BoundaryNorm(bounds, num_classes)

    im_mask = None
    for row, imgs_data in enumerate(all_imgs_data):
        for i, image in enumerate(imgs_data):
            img_np = image.cpu().numpy().astype(np.float32)
            img_np = np.clip(img_np, 0, 1)
            axs[row][i].imshow(img_np)
            if row == 0:
                axs[row][i].set_title(f'{Dataset_class.MODALITIES_NAMES[Dataset_class.MODALITIES[i]]} input')

        if pred is not None:
            ax_pred = axs[row][num_cols - 2 if masks is not None else num_cols - 1]
            ax_pred.imshow(pred, cmap=Dataset_class.CMAP_GT, norm=norm)
            if row == 0:
                ax_pred.set_title('Predicted mask')

        if masks is not None and masks[row] is not None:
            im_mask = axs[row][num_cols - 1].imshow(masks[row], cmap=Dataset_class.CMAP_GT, norm=norm)
            if row == 0:
                axs[row][num_cols - 1].set_title('Ground Truth Mask')

    if showColorbar and im_mask is not None:
        # fig.subplots_adjust(right=0.85)  # make room on the right
        fig.subplots_adjust(left=0.02, top=0.95, bottom=0.02, right=0.85, hspace=0.05, wspace=0.05)

        cbar_scale = num_classes / 21
        cbar_ax = fig.add_axes([0.87, 0.15 + (1 - cbar_scale) * 0.1, 0.02, 0.5 + cbar_scale * 0.2])
        cbar = fig.colorbar(im_mask, cax=cbar_ax, ticks=range(num_classes))
        cbar.ax.set_yticklabels(Dataset_class.CLASS_NAMES, fontsize=10)
        cbar.ax.yaxis.set_tick_params(which='minor', length=0)
        cbar.set_label('Class')

    # no tight_layout — it conflicts with add_axes

    if save:
        save_name = get_unique_save_name(PLOTS_SAVE_DIR, "img_mask_pred", extension=".png")
        save_path = f"{PLOTS_SAVE_DIR}{save_name}"
        print(f"Saving img/mask/pred to {save_path}")
        
        plt.savefig(save_path, dpi=300, bbox_inches='tight')

    else:
        plt.show()


def convert_and_plot_img_data(images_list_or_grid, Dataset_class, masks=None, mask=None, 
        pred=None, showColorbar=True, stretch_modalities=False, save=False):
    # Normalize to grid format (list of rows)
    # Single row: images_list is a flat list of images -> wrap in outer list
    # Multi row: images_list is already a list of lists
    if mask is not None and masks is None:
        masks = [mask]

    if not isinstance(images_list_or_grid, list):
        images_list_or_grid = [images_list_or_grid]

    # Detect if flat list (single row) or grid (list of lists)
    if not isinstance(images_list_or_grid[0], list):
        images_list_or_grid = [images_list_or_grid]

    if masks is None:
        masks = [None] * len(images_list_or_grid)

    all_imgs_data = []
    all_masks_converted = []

    for images_list, row_mask in zip(images_list_or_grid, masks):
        imgs_data = []

        for i, img in enumerate(images_list):
            if isinstance(img, np.ndarray):
                img = torch.from_numpy(img)

            img = img.detach().clone()
            img_bands = img.shape[0]

            if img_bands == 1:
                img = img.repeat(3, 1, 1)
            elif img_bands == 2:
                zero = torch.zeros_like(img[0:1])
                img = torch.cat([img, zero], dim=0)
            elif img_bands > 3:
                img = lossy_convert_to_RGB(img)

            if stretch_modalities and Dataset_class.MODALITIES[i] in Dataset_class.VIS_MODALITIES_NEED_STRETCHING:
                img = stretch_to_0_1(img)

            img = img.permute(1, 2, 0)
            imgs_data.append(img)

        all_imgs_data.append(imgs_data)

        if row_mask is not None:
            if isinstance(row_mask, np.ndarray):
                row_mask = torch.from_numpy(row_mask)
            row_mask = row_mask.detach().clone().long().squeeze().cpu()
            print("GT classes:", np.unique(row_mask))
        all_masks_converted.append(row_mask)

    if pred is not None:
        pred = pred.squeeze()
        
        # Check if already argmaxed
        if len(pred.shape) == 3 and pred.shape[0] == Dataset_class.NUM_CLASSES:
            pred = torch.argmax(pred, dim=0)
            # (NUM_CLASSES, SIZE, SIZE) -> (SIZE, SIZE)

        pred = pred.cpu().numpy()
        print("Pred classes:", np.unique(pred))

    plot_img_data(all_imgs_data, Dataset_class, masks=all_masks_converted, pred=pred, showColorbar=showColorbar, save=save)


def show_dataset_images(Dataset_class, train_or_test, indices):
    if not isinstance(indices, list):
        indices = [indices]

    all_images_lists = []
    all_masks = []

    for index in indices:
        data_paths = [Dataset_class.DATA_FILES[train_or_test] + Dataset_class.MODALITIES_FOLDERS[m] for m in Dataset_class.MODALITIES]
        masks_path = Dataset_class.DATA_FILES[train_or_test] + "gt/"

        images_list = []
        for data_path in data_paths:
            data_tile = data_path + data_tile_filename.format(index)
            with rio.open(data_tile) as img_src:
                img = img_src.read()

            images_list.append(img)

        gt_tile = masks_path + data_tile_filename.format(index)
        with rio.open(gt_tile) as mask_src:
            mask = mask_src.read()

        all_images_lists.append(images_list)
        all_masks.append(mask)

    convert_and_plot_img_data(all_images_lists, Dataset_class, masks=all_masks, 
            stretch_modalities=True)
