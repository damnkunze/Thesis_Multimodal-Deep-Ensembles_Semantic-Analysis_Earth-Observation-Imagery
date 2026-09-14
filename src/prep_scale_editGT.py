import contextlib

from src.base import *
from src.prep_merge_cut_layer_norm import *

# ## Scale (full HSI, all RGB imgs)
from rasterio.warp import reproject


def scale_to_GSD_opt_normalize(data_path, out_path, target_gsd, is_mask=False, is_RGB=False, is_HSI=False, 
                                is_MSI=False, is_SAR=False, is_DSM=False, clamp_percentile=None, no_data_val=None):

    with rio.open(data_path) as src_data:
        src_data_gsd = src_data.res[0]
        gsd_scaling_factor = target_gsd / src_data_gsd
        print(f"Reprojecting from GSD {src_data_gsd} to GSD {target_gsd} -> factor {gsd_scaling_factor} ...")

        # New smaller or bigger data, written in place
        new_height = int(src_data.height / gsd_scaling_factor)
        new_width = int(src_data.width / gsd_scaling_factor)
        print("old H x W:", src_data.height, src_data.width)
        print("new H x W:", new_height, new_width)

        # CUBIC looks best on tiled images (better than bilinear)
        # nearest resampling is very bad for images
        resampling = rio.enums.Resampling.cubic

        # Only for masks use nearest
        if is_mask:
            resampling = rio.enums.Resampling.nearest
        print(f'Resampling using "{rio.enums.Resampling(resampling).name}"')

        data = src_data.read(out_shape=(src_data.count, new_height, new_width), resampling=resampling)

        performed_transform = src_data.transform * src_data.transform.scale(
            (src_data.width / new_width),
            (src_data.height / new_height)
        )

        print("data.shape", data.shape)

        data_meta = src_data.meta.copy()
        data_meta['transform'] = performed_transform
        data_meta['height'] = new_height
        data_meta['width'] = new_width

        # important since normalization gives vals [0 - 1]
        data_meta["dtype"] = rio.float32

        
        # NORMALIZE
        if is_RGB:
            data = data.astype(float) / 255.0
            print("RGB normalized")

        elif is_HSI or is_MSI:
            # run normalize_band() on each band
            data = normalize_all_bands(data)
            print(("HSI" if is_HSI else "MSI") + " normalized")

        elif is_DSM:
            if no_data_val is not None:
                # use np.nan ?
                # Replace NoData Value with `None` but np float complatible: np.nan
                data = np.where(data == no_data_val, np.nan, data)
                print("DSM corrected NoData:", no_data_val)

            data = normalize_band(data)
            print("DSM normalized")

        elif is_SAR:
            if clamp_percentile is not None:
                data = clamp_percentile_band(data, percentile=clamp_percentile, lowClamp=False)

            data = normalize_all_bands(data)
            print("SAR clamped and normalized")


        with rio.open(out_path, 'w', **data_meta) as out_data_file:
            out_data_file.write(data)

        print("Saved scaled to ", out_path)


def scale_each_image_to_GSD(in_dir, in_filenames, out_dir, gsd, **args):
    # **args to catch all extra args and pass them on

    print("Clearing output dir:", out_dir)
    # !rm {out_dir}/*
    os.makedirs(out_dir, exist_ok=True)

    in_paths = [in_dir + f for f in in_filenames]
    out_paths = [out_dir + f for f in in_filenames]

    for in_path, out_path in zip(in_paths, out_paths):
        scale_to_GSD_opt_normalize(in_path, out_path, gsd, **args)
        # show_full_image(out_path)


def remove_bands_from_image(in_path, out_path, bands_to_remove):
    """ eg bands_to_remove = [4] to remove band 4 (1-based indexing) 
    
    Used in
    # MDAS RGB 4 bands to 3 bands
    # DFC18 HSI 50 bands to 48 bands

    """
    with rio.open(in_path) as src:
        data = src.read()

        print("Original bands:", data.shape[0])
        print("Removing bands:", bands_to_remove)

        # Remove specified bands (adjusting for 0-based indexing)
        bands_to_keep = [i for i in range(data.shape[0]) if (i + 1) not in bands_to_remove]
        cleaned_data = data[bands_to_keep, :, :]

        print("New bands:", cleaned_data.shape[0])

        # Save cleaned image with same meta
        data_meta = src.meta.copy()
        data_meta['count'] = cleaned_data.shape[0]

    with rio.open(out_path, 'w', **data_meta) as out_file:
        out_file.write(cleaned_data)

    print("Saved cleaned image to ", out_path)


# Used by MDAS
def convert_GT_classes(gt_file_in, gt_file_out, classes_conversion):
    """ classes_conversion is dict map from old to new classes """

    with rio.open(gt_file_in) as src_gt:
        gt_data = src_gt.read()

        print("Unique values in GT before cleaning:", np.unique(gt_data))

        cleaned_gt_data = np.zeros_like(gt_data, dtype=np.uint8)

        for original_class, new_class in classes_conversion.items():
            cleaned_gt_data[gt_data == original_class] = new_class

        print("Unique values in GT after cleaning:", np.unique(cleaned_gt_data))

        # Save cleaned with same meta
        gt_meta = src_gt.meta.copy()

    with rio.open(gt_file_out, 'w', **gt_meta) as out_gt_file:
        out_gt_file.write(cleaned_gt_data)

        print("Saved cleaned GT to ", gt_file_out)


def combine_GTs(gt_files_in, gt_file_out, ignore_class):
    """ ignore_class: no label class. Do not to overwrite with it """

    # Open flexible number of gts
    with contextlib.ExitStack() as stack:
        # Like individual "with rio.open(path)" statements: stack.enter_context()
        src_files = [stack.enter_context(rio.open(path)) for path in gt_files_in]

        out = src_files[0].read()
        gt_meta = src_files[0].meta.copy()

        for gt_src in src_files[1:]:
            data = gt_src.read() 

            if out.shape != data.shape:
                raise ValueError("GT shape mismatch:", out.shape, data.shape)

            # ignoring ignore_class
            # where(cond, a, b) chooses a if cond else b
            out = np.where(data != ignore_class, data, out)

    with rio.open(gt_file_out, 'w', **gt_meta) as out_gt_file:
        out_gt_file.write(out)

    print("Saved cleaned GT to ", gt_file_out)


# Check normalized HSI
# histogram of count of values in data_HSI_path_scaled

''' with rio.open(data_HSI_path_scaled) as src:
    hsi_data = np.empty(src.count * src.height * src.width, dtype=np.float32)
    offset = 0
    s = src.height * src.width

    for i in range(src.count):
        band = clamp_percentile_band(src.read(i + 1), percentile=5, lowClamp=False)

        hsi_data[offset * s : offset * s + s] = band.flatten()
        offset += 1


plt.figure(figsize=(10, 6))
plt.hist(hsi_data, bins=50, color='skyblue', edgecolor='black')
plt.title('Histogram of HSI Pixel Values (Normalized)')
plt.xlabel('Pixel Value clamped (Normalized)')
plt.ylabel('Frequency')
plt.grid(True)
plt.show() '''