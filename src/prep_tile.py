import contextlib
from rasterio.enums import Resampling

from src.base import *

# ## Tile (all)

# Skip image if valid part < 30%
# SIZE is W = H
min_tile_size = SIZE * SIZE * 0.3 


def pad_partial_tile_to_size(a, size, pad_with=np.nan):
    '''
    possible shapes (RGB):
    (3, 244, 512)
    (3, 512, 244)
    (3, 244, 244)

    (0, 0): leave dim 0 as is
    (0, size - a.shape[1]): Add up to size rows
    (0, size - a.shape[2]): Add up to size cols

    ,mode='constant',constant_values=(np.nan,)
    '''
    return np.pad(a, ((0, 0), (0, size - a.shape[1]), (0, size - a.shape[2])), mode='constant', constant_values=pad_with)


def tile_images_aligned_from_all_sensors(all_in_paths, all_out_dirs, filename_format=data_tile_filename,
                                            read_from_disk=[], skipHole=None, DEBUG=False):
    """
    Tile all image files: RGB, HSI data and ground truth (GT)

    Opens all images at the same time to ensure tiles overlay correctly
    Ineffective but works

    all_in_paths:   RGB_data_file, HSI_data_file, LIDAR_data_file, GT_data_file
    all_out_dirs:  out_RGB_data_dir, out_HSI_data_dir, out_LIDAR_data_file, out_GT_data_dir

    """
    
    # Open flexible number of images
    with contextlib.ExitStack() as stack:
        # Like individual "with rio.open(path)" statements: stack.enter_context()
        src_files = [stack.enter_context(rio.open(path)) for path in all_in_paths]

        # GT is last in list
        gt_file = src_files[-1]

        print("Checking if images compatible: \nAll images should be the same or bigger than the first one supplied")

        check_heights = [src_file.height for src_file in src_files]
        check_widths = [src_file.width for src_file in src_files]
        check_bounds = [src_file.bounds for src_file in src_files]

        end_y = src_files[0].height
        end_x = src_files[0].width 
        
        """
        if not all(h >= end_y for h in check_heights):
            raise ValueError(f"Not all images have height >= first image. Heights: {check_heights}")
            
        if not all(w >= end_x for w in check_widths):
            raise ValueError(f"Not all images have width >= first image. Widths: {check_widths}") 
        """

        print("Bounds:", check_bounds)

        # For speedup
        print("Reading full data into memory...")

        """ full_data_images = [src_file.read() for src_file in src_files] """
        # Skip too big images (e.g. HSI)
        full_data_images = []
        for idx, src_file in enumerate(src_files):
            if not idx in read_from_disk:
                full_data_images.append(src_file.read())
            else:
                full_data_images.append(None)
                print(f"Skipping loading full image for {all_in_paths[idx]} (index {idx})")

        if skipHole is not None:
            skip_from = skipHole[0]
            skip_to = skipHole[1]

        tile_counter = 0

        # Tile SIZE x SIZE
        for off_y in tqdm(range(0, end_y, SIZE), desc="Going over y coords"):
            for off_x in range(0, end_x, SIZE):

                # Skip Hole: Check if with curr offsets a SIZE * SIZE tile would be fully on hole
                if skipHole is not None and off_y > skip_from[0] and off_x > skip_from[1] and \
                   off_y <= skip_to[0] and off_x <= skip_to[1]:
                    print(f"Skipping hole, off_y {off_y}, off_x {off_x}\
                    off_y > skip_from[0] {off_y > skip_from[0]}, off_x > skip_from[1] {off_x > skip_from[1]}, off_y <= skip_to[0] {off_y <= skip_to[0]}, off_x <= skip_to[1] {off_x <= skip_to[1]}")
                    continue

                ''' print(f"off_y {off_y}, off_x {off_x}\n off_y > skip_from[0] {off_y > skip_from[0]}, off_x > skip_from[1] {off_x > skip_from[1]}, off_y <= skip_to[0] {off_y <= skip_to[0]}, off_x <= skip_to[1] {off_x <= skip_to[1]}")
                continue '''

                # ERROR catchup
                """ if (tile_counter < 629):
                    tile_counter += 1
                    continue """
                ''' # TEST, one row = 37 imgs
                if (DEBUG and tile_counter >= 38): break '''

                data_h = min(SIZE, end_y - off_y)
                data_w = min(SIZE, end_x - off_x)

                # Check size big enough, correct if necessary
                if data_h != SIZE or data_w != SIZE:
                    # Skip image if valid part < 50%

                    if ((data_h * data_w) < min_tile_size):
                        print(f"\nWARNING\nSkipping image tile {tile_counter}: valid part too small: H {data_h} * W {data_w}")
                        # plot.show(data_tile)
                        continue

                # Window for tile
                tile_window = windows.Window(off_x, off_y, SIZE, SIZE)

                tile_bounds_from_gt = gt_file.window_bounds(tile_window)

                # Do same tile for each image
                for src_file, full_data, out_dir in zip(src_files, full_data_images, all_out_dirs):

                    if full_data is not None:
                        # For speedup read out of RAM
                        data_tile = full_data[:, off_y:off_y + data_h, off_x:off_x + data_w]
                    else:
                        # To safe RAM read from disk
                        # Expecting GT to be first in list
                        window_from_gt_bounds = windows.from_bounds(*tile_bounds_from_gt, src_file.transform)

                        # Use data_w and data_h to read edges correctly
                        data_tile = src_file.read(
                            window=window_from_gt_bounds,
                            out_shape=(src_file.count, data_h, data_w),
                            resampling=Resampling.cubic)

                        # data_tile = src_file.read(window=tile_window)

                    # If at edge w. or h. might be small, fill out with `None`` using np.pad
                    if data_h != SIZE or data_w != SIZE:
                        print("Correcting shape data tile", tile_counter, data_tile.shape)
                       
                        """ # pad all with 0 np.nan except gt with 0
                        pad_with = 0 if src_file == gt_file else np.nan 
                        print(f"File src_file == gt_file: {src_file == gt_file}. pad_with: {pad_with}")
                        """
                        
                        # pad all with 0 
                        pad_with = 0

                        data_tile = pad_partial_tile_to_size(data_tile, SIZE, pad_with=pad_with)

                    ''' # DEBUG show
                    fig, ax = plt.subplots()
                    ax.imshow(torch.from_numpy(data_tile).permute(1, 2, 0))
                    return '''

                    # Write data tile
                    data_meta = src_file.meta.copy()
                    data_meta['transform'] = src_file.window_transform(tile_window)
                    data_meta['bounds'] = src_file.window_bounds(tile_window)
                    data_meta['width'] = tile_window.width
                    data_meta['height'] = tile_window.height

                    output_data_path = out_dir + filename_format.format(tile_counter)

                    with rio.open(output_data_path, 'w', **data_meta) as out_data_tile_file:
                        out_data_tile_file.write(data_tile)

                    print(".", end='')

                # flush to stdout, wont get printed otherwise when running through ./attached_prep
                print(tile_counter, end=', ', flush=True)
                tile_counter += 1


    print(f"Finished tiling. Created {tile_counter} tiles at \n {all_out_dirs} \n")