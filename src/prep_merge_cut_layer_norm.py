from src.base import *


# Merge RGB data tiles -------------------------------------------------
from rasterio import merge

def merge_data_tiles(data_tifs_sorted_paths, out_path):
    """Merge all data images into one"""

    merged, output_transform = merge.merge(data_tifs_sorted_paths)

    # Copy meta from first img
    with rio.open(data_tifs_sorted_paths[0]) as src:
        src_meta = src.meta.copy()  # transform & crs same
        src_meta['height'] = merged.shape[1]
        src_meta['width'] = merged.shape[2]
        src_meta['transform'] = output_transform

    # Save merged data with same CRS and transform as GT
    with rio.open(out_path, 'w', **src_meta) as dst:
        dst.write(merged)

    print(f"Merged data image saved to: {out_path}")



# ## Layer (LIDAR) -----------------------------------------------------

# Layer (merge) LIDAR data tiles channel wise + NORMALIZE + remove NoData

def layer_data_images(data_tifs_one_channeled_paths, out_path, new_no_data=np.nan):
    """ Channelwise / "On top of each other """
    channels = len(data_tifs_one_channeled_paths)


    # Take dimensions from first image
    with rio.open(data_tifs_one_channeled_paths[0]) as src_first:
        res_image = np.empty((channels, src_first.height, src_first.width), dtype=src_first.dtypes[0])

        meta = src_first.meta.copy()
        meta['count'] = channels

    # Copy images 1 - 3
    for i, img_path in enumerate(data_tifs_one_channeled_paths):

        with rio.open(img_path) as src_data:
            data = src_data.read()

            no_data_val = src_data.nodata

            # CORRECT NoData
            if no_data_val is not None:
                # Replace NoData Value with `None` but np float complatible: np.nan
                data = np.where(data == no_data_val, new_no_data, data)
                meta['nodata'] = np.nan
                print(f"Corrected NoData: {no_data_val} in {img_path}")

            # NORMALIZE
            data = normalize_band(data)

            # LAYER
            channel_index = i
            res_image[channel_index] = data

    with rio.open(out_path, "w", **meta) as src_out:
        src_out.write(res_image)

    print(f"Merged data image of {channels} channels saved to: {out_path}")


    
# ## MDAS: HSI
def normalize_all_bands_file(in_path, out_path):
    with rio.open(in_path) as src_data:
        data = src_data.read()

        normalized = normalize_all_bands(data)

        meta = src_data.meta.copy()
        
        # normalize converts to float32
        meta["dtype"] = rio.float32

    with rio.open(out_path, "w", **meta) as src_out:
        src_out.write(normalized)

    print(f"Saved normalized to:", out_path)



def normalize_all_bands(data):
    """ run normalize_band() on each band """

    normalized = np.empty_like(data, dtype=np.float32)

    for c in range(data.shape[0]):
        normalized[c] = normalize_band(data[c])

    return normalized


# DFC18: DSM
def normalize_and_correct(img_path, out_path, new_no_data=np.nan):
    # Default percentile: 2%, only high clamp
    with rio.open(img_path) as src_data:
        data = src_data.read()
        meta = src_data.meta.copy()

        no_data_val = src_data.nodata
        if no_data_val is not None:
            # Replace NoData Value with `None` but np float complatible: np.nan
            data = np.where(data == no_data_val, new_no_data, data)
            meta['nodata'] = np.nan
            print(f"Corrected NoData: {no_data_val} in {img_path}")

        data_normalized = normalize_all_bands(data)


    with rio.open(out_path, "w", **meta) as src_out:
        src_out.write(data_normalized)

    print(f"Normalized and clamped DSM, saved to {out_path}")



# ## Cut Train Part (HSI, LIDAR) ---------------------------------


def cut_image_with_bounds_from_other(img1_path, img2_path, out_path):
    with rio.open(img1_path) as src_img1, \
         rio.open(img2_path) as src_img2:

        # Calculate window based on GT bounds but using imgs transform
        window_from_gt_bounds = windows.from_bounds(*src_img2.bounds, src_img1.transform)

        print("src_img1.bounds", src_img1.bounds)
        print("src_img2.bounds", src_img2.bounds)
        print("window_from_gt_bounds", window_from_gt_bounds)

        train_data_part = src_img1.read(window=window_from_gt_bounds)
        train_data_part_full = src_img1.read()
        print("train_data_part.shape", train_data_part.shape)
        print("train_data_part_full.shape", train_data_part_full.shape)

        data_meta = src_img1.meta.copy()
        data_meta['width'] = window_from_gt_bounds.width
        data_meta['height'] = window_from_gt_bounds.height
        data_meta['transform'] = src_img1.window_transform(window_from_gt_bounds)


        with rio.open(out_path, 'w', **data_meta) as out_file:
            out_file.write(train_data_part)

        print(f"Cut {img1_path} to area of {img2_path}\n Saved here {out_path}")
