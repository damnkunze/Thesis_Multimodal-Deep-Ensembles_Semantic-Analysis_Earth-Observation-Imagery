from src.base import *

from src.DFC18_Dataset import *
from src.MDAS_Dataset import *

# Channels selection:
# For selection / optimization unified: seperate modalities should not matter
#   Also allow data manipulation
# For DFC18Dataset src.read(): needs nested list with channels per modality

# So in Ensemble, to instatiate Models:
#   get channel_selection_func passed
#   channels_mask = channel_selection_func(modalities, channels_per_modality)
#
#   channels_mask:
#   only RGB: -> [True, True, True, False, ... False] -> [[1, 2, 3], [], []]
#   mixed: -> [True, True, True, False, ..., False, True] -> [[1, 2, 3], [], [3]]


from itertools import chain, combinations

def powerset(s):
    """ All possible combinations of items within a list """
    return list(chain.from_iterable(combinations(s, r) for r in range(1, len(s) + 1)))


class Channel_Modulation():
    """ 
    Each returns LIST! shape (MODELS, C, K)
    - C: count inputs (channels dataset provides)
    - K: count outputs (channels model will use)

    values are [0 - 1]

    Always:
    C = dataset.MODALITIES_TOTAL_CHANNELS
    -> (MODELS, C, K)
    """

    @staticmethod
    def apply_modulation(data_batch, modulation):
        """ Multiply all channels with channels_modulation 
        
        data:       (C, H, W)
        modulation: (C, K)

        single tensor
        ((C, H, W).T @ (C, K)).T -> (K, H, W)

        batch of tensors
        ((B, C, H, W).T @ (C, K)).T -> (B, K, H, W)
        
        # Instead of .T use permute() since is 3d tensor (which throws warning but works)
        # Is also faster on batches
        """

        # If single tensor, add batch dim
        if data_batch.dim() == 3:
            data_batch = data_batch.unsqueeze(0) 
            # (C, H, W) -> (B=1, C, H, W)

        data_batch = data_batch.permute(0, 2, 3, 1)
        # -> (B, H, W, C)

        data_batch = data_batch @ modulation
        # (B, H, W, C) @ (C, K) -> (B, H, W, K)

        data_batch = data_batch.permute(0, 3, 1, 2)
        # -> (B, K, H, W)

        # If was single tensor, remove batch dim
        if data_batch.shape[0] == 1:
            data_batch = data_batch.squeeze(0)
            # (B=1, K, H, W) -> (K, H, W)

        return data_batch


    #################### MODULATION FUNCS ############################

    @staticmethod
    def identity(Dataset_class, count_models=7):
        '''
        For test, all 1, so no change

        C = K = Dataset_class.MODALITIES_TOTAL_CHANNELS
        -> (MODELS, C, K)
        '''

        C = Dataset_class.MODALITIES_TOTAL_CHANNELS
        
        all_modulations = [torch.eye(C, dtype=torch.float32) for _ in range(count_models)]

        return all_modulations

    @staticmethod
    def powerset_of_all_modalities(Dataset_class, count_models=None):
        '''
        Cannot be passed count_models!

        in: ('RGB', 'HSI', 'LIDAR') and {'RGB': 3, 'HSI': 50, 'LIDAR': 3}

        out: 7 masks with channels for each of
        [('RGB',), ('HSI',), ('LIDAR',), ('RGB', 'HSI'), ('RGB', 'LIDAR'), ('HSI', 'LIDAR'), ('RGB', 'HSI', 'LIDAR')]
        
        ('RGB',) -> (56, 3):
        [[1, 0, 0],
         [0, 1, 0],
         [0, 0, 1],
         [0, 0, 0],
         [0, 0, 0],
            ...          
                  ]

        K = <varies per model>
        MODELS = size_powerset
        '''

        # Cannot be passed count_models
        if count_models is not None: 
            print("[Ignored] count_models: powerset_of_all_modalities() cannot be passed count_models, its simply dependent on modalities count")

        modalities = Dataset_class.MODALITIES
        channel_counts = Dataset_class.MODALITIES_CHANNEL_COUNTS
        C = Dataset_class.MODALITIES_TOTAL_CHANNELS

        powerset_modalities = powerset(modalities)
        print("Order powerset_modalities:", powerset_modalities)
        """ print(f"Dataset Modalities: {Dataset_class.MODALITIES_NAMES}")
        mods_as_names = [[Dataset_class.MODALITIES_NAMES[i] for i in curr_model] for curr_model in powerset_modalities]
        print(f"Modalities for Models: \n", mods_as_names) """

        size_powerset = len(powerset_modalities)

        all_modulations = [] # (MODELS, C, variable K)

        # for each part of the powerset M
        #   for each m of all possble modalities 
        #     mark curr part of curr_mask with 1 if m in M

        for i, curr_modalities in enumerate(powerset_modalities):
            
            # Curr MODEL modulation
            K = sum([channel_counts[m] for m in curr_modalities])
            modulation = torch.zeros((C, K), dtype=torch.float32) # CHECK IF CORRECT!
            
            # 0 ... K - 1
            channel_i = 0 
            channel_j = 0 

            for modality in modalities:
                count_channels = channel_counts[modality]
                
                if modality not in curr_modalities:
                    channel_i += count_channels
                
                else:
                    # All lie on diagonal
                    for _ in range(count_channels):
                        modulation[channel_i][channel_j] = 1
                        channel_i += 1
                        channel_j += 1
            
            all_modulations.append(modulation)

        return all_modulations


    @staticmethod
    def random_channel_drop(Dataset_class, count_models, drop_prob=0.3):
        '''
        Independently per channel: 
        Randomly drop each channel with probability drop_prob

        -> (MODELS, C, variable K)
        - K varies per model

        eg. drop_prob=0.3: 
            Keep ~70% of channels
        '''

        C = Dataset_class.MODALITIES_TOTAL_CHANNELS

        all_modulations = []  

        # Create random mask with 0 / 1 for each channel
        for i in range(count_models):
            keep_mask = NP_RAND.random(C) > drop_prob # (C,) with True / False
            kept_indices = np.nonzero(keep_mask)[0]   # (K,) with indices
            K = len(kept_indices)

            modulation = torch.zeros((C, K), dtype=torch.float32)
            for j, idx in enumerate(kept_indices):
                modulation[idx, j] = 1.0
            all_modulations.append(modulation)

        return all_modulations


    @staticmethod
    def random_drop_percentage_of_modality(Dataset_class, count_models, percentage=0.33):
        '''
        First equally choose a modality.
        Of this modality, randomly drop percentage of channels.

        -> (MODELS, C, variable K)
        -  K varies per model


        eg: percentage=0.33
            randomly choosen modality RGB
        -> Drop ~33% of RGB channels

        => Results eg
           Num of True in mask: 39 or 55
        '''
        modalities = Dataset_class.MODALITIES
        channel_counts = Dataset_class.MODALITIES_CHANNEL_COUNTS
        C = Dataset_class.MODALITIES_TOTAL_CHANNELS

        all_modulations = []

        for i in range(count_models):
            # Start with all channels selected
            keep_mask = np.full((C,), 1)

            # Select a modality
            chosen_modality = NP_RAND.choice(modalities)
            chosen_modality_channels = channel_counts[chosen_modality]

            # calc range of channels
            modality_start_idx = sum([channel_counts[m] for m in modalities if m < chosen_modality])
            modality_end_idx = modality_start_idx + chosen_modality_channels

            # Percentage of channels, ceil for at least 1 drop
            num_channels_to_drop = int(np.ceil(chosen_modality_channels * percentage))
            # Without repetition using replace=False
            drop_indices = NP_RAND.choice(range(modality_start_idx, modality_end_idx), size=num_channels_to_drop, replace=False)

            keep_mask[drop_indices] = 0

            kept_indices = np.nonzero(keep_mask)[0]
            K = len(kept_indices)

            modulation = torch.zeros((C, K), dtype=torch.float32)
            for j, idx in enumerate(kept_indices):
                modulation[idx, j] = 1.0
            all_modulations.append(modulation)

        return all_modulations


    ## MUTATION ------------------------------------------------------------------------------

    @staticmethod
    def random_linear_combination(Dataset_class, count_models, K=None):
        """
        Channel reduction if K != C
        Get random matrix to "weight" all channels to K combined channels
        -> (C, K)

        Each column sums to 1:
            result.sum(axis=0) = [1, 1, ...]

        TODO: CHECK IF CORRECT
        """
        C = Dataset_class.MODALITIES_TOTAL_CHANNELS

        if K is None:
            K = C

        all_modulations = []
        
        for i in range(count_models):
            rand_matrix = torch.rand(C, K, generator=generator)
            
            # Normalize columns to sum to 1
            modulation = rand_matrix / rand_matrix.sum(axis=0, keepdims=True)
            all_modulations.append(modulation)

        return all_modulations


    ## OPTIMIZING ------------------------------------------------------------------------------

    @staticmethod
    def random_subsets_PCA(Dataset_class, count_models, samples_percentage=0.75, components_count=None, 
                            max_components_count=None, components_percentage=1,
                            channel_dropout_percentage=0, previous_modulation=None):
        """ 
        Primary Component Analysis

        (C, K) should be
        K: Count of components

        Projektionsmatrix in den K-dim Unterraum des Eigenraumes (also die K ersten Eigenvektoren)

        For each model in ensemble:

            1. Get subset_percentage size subset of dataset samples <- (300, 56, 128, 128)
                -> X: (225, 56, 128, 128)

            2. Flatten to 2D (TOTAL_PIXELS=SAMPLES, CHANNELS=FEATURES)
                -> flat: (225*128*128, 56) = (3_686_400, 56)
            
            3. Center Data by subtracting mean per sample <- (56,)
                -> centered: (3_686_400, 56)

            4. Calc SVD(X) <- (SAMPLES, FEATURES)

            5. Return first K eigenvectors as projection matrix (C, K)
                -> (56, K)
        
        # Test PCA accuracy
        torch.dist(centered, U @ torch.diag(S) @ V)

        # Options:
        - Allow setting components by constant count or percentage
        """
        dataset = Dataset_class(TRAIN, augment=True)
        C = dataset.MODALITIES_TOTAL_CHANNELS

        # For efficiency, only load dataset once
        print("Reading full dataset into memory...")

        all_data_unmodified = []
        # (DATASET_SIZE, CHANNELS, SIZE, SIZE)
        
        for i in range(len(dataset)):
            data, _ = dataset[i]
            all_data_unmodified.append(data)
        
        # Get Modulation Matrix
        if previous_modulation is not None:
            previous_modulation = previous_modulation(Dataset_class, count_models=count_models)
            # Apply for each model down below
            # (MODELS, C, K_prev)

            if count_models != len(previous_modulation):
                print(f"[i] Adjusting count_models ({count_models}) to match previous_modulation size ({len(previous_modulation)})")
                count_models = len(previous_modulation) 

        # Count of samples, ceil for at least 1 drop
        num_samples = int(np.ceil(len(dataset) * samples_percentage))
        
        # Total of all values per channel, combined across all images
        TOTAL_PIXELS_PER_CHANNEL = num_samples * SIZE * SIZE

        # Each model gets k_components principal components, 
        all_modulations = [] # (MODELS, C, K)

        for model_idx in range(count_models):
            all_data = all_data_unmodified

            # 0. [Optional] Previous modulation
            if previous_modulation is not None:
                modulation = previous_modulation[model_idx] # (C, K_prev)
                all_data = [Channel_Modulation.apply_modulation(data, modulation) for data in all_data]
                # (DATASET_SIZE, K_prev, SIZE, SIZE)

                # Update C for PCA to new channel count after previous modulation
                C = modulation.shape[1] # K_prev

            # 1. Subset
            # Without repetition using replace=False
            subset_indices = NP_RAND.choice(len(dataset), size=num_samples, replace=False)

            # Get from already loaded data
            subset = torch.stack([all_data[idx] for idx in subset_indices])
            # (SAMPLES, CHANNELS, SIZE, SIZE)

            # 1.2 [Optional] channel dropout
            if channel_dropout_percentage > 0:
                keep_mask = NP_RAND.random(C) > channel_dropout_percentage # (C,) with True / False
                # [0] because nonzero() returns tuple
                kept_indices = np.nonzero(keep_mask)[0]   # (LESS_CHANNELS,) with indices

                subset = subset[:, kept_indices, :, :]
                C = len(kept_indices)
                # (SAMPLES, C, SIZE, SIZE)

            # 2. Flatten
            # Permute to correct order first in which it should be flattened
            subset = subset.permute(0, 2, 3, 1)
            # (SAMPLES, SIZE, SIZE, CHANNELS)

            subset = subset.reshape(TOTAL_PIXELS_PER_CHANNEL, C)
            # (Equivalent to reshape(TOTAL_PIXELS_PER_CHANNEL, -1))
            
            # So in PCA terms (with SAMPLES=TOTAL_PIXELS_PER_CHANNEL):
            #  (SAMPLES, FEATURES) eg. (4_096_000, 56)

            # 3. Center
            subset_mean = subset.mean(axis=0, keepdims=True)
            subset = subset - subset_mean
            # Test with print(subset.sum(axis=0)), will be good at ~xxe-6, 

            # 4. PCA (using SVD)
            U, S, Vh = torch.linalg.svd(subset, full_matrices=False)
            # Vh, since V is transposed

            # 5. Get first K eigenvectors as projection matrix (C, K)
            # Count components, at least 1
            if components_count is None and max_components_count is None:
                K = int(np.ceil(C * components_percentage))
                
            elif max_components_count is not None:
                K = min(int(np.ceil(C * components_percentage)), max_components_count)

            else:
                K = components_count

            Vk = Vh[:K, :].T
            all_modulations.append(Vk)

            # [Optional] CHECK IF VALID
            # Explained variance ratio
            explained_var = (S[:K] ** 2).sum() / (S ** 2).sum()
            # High value (e.g., > 0.95) means K components capture most variance

            print("explained variance:", explained_var)

        # [Optional] Chain previous_modulation and all_modulations
        if previous_modulation is not None:
            chained_modulations = []
            for i in range(count_models):
                chained_modulation = previous_modulation[i] @ all_modulations[i]
                # (C, K_prev) @ (K_prev, K) -> (C, K)
                chained_modulations.append(chained_modulation)

            return chained_modulations

        return all_modulations

    """ # CHECK IF SVD VALID
    dataset = DFC18Dataset(TRAIN_FILE_ROOT, augment=True)
    for i in range(3):
        img1 = dataset[i][0] # (C, H, W)
        M = ensemble1.channel_modulations[i] # (C, K)

        # PCA: project to K dims and back
        img_pca = Channel_Modulation.apply_modulation(img1, M)  # (K, H, W)
        img_pca_recon = Channel_Modulation.apply_modulation(img_pca, M.T)  # (C, H, W) - lossy reconstruction

        # Random: same process
        W_random = torch.rand((56, M.shape[1]), generator=generator)
        W_random = W_random / W_random.norm(dim=0, keepdim=True)  # make columns ~orthonormal
        img_rand = Channel_Modulation.apply_modulation(img1, W_random)
        img_rand_recon = Channel_Modulation.apply_modulation(img_rand, W_random.T)

        # Compare reconstruction errors
        error_pca = torch.dist(img1, img_pca_recon)
        error_random = torch.dist(img1, img_rand_recon)
        print(f"PCA error: {error_pca:.4f}, Random error: {error_random:.4f}")
        # PCA should be much lower if it's meaningful
    """


    @staticmethod
    def random_classes_simple_LDA(Dataset_class, count_models, K=None):
        """
        # DATA VALUE BASED - Stochastic LDA variant
        - Simplified LDA without covariance matrix, using random pairwise sample differences.
        - Get matrix with differences between classes pixels
        Default K = C

        if K != None: Channel reduction
        - "weight" all channels to K combined channels
        -> (C, K)


        1. Wähle zufällig 2 Klassen A, B
        2. Wähle aus den beiden Klassen zufällig je ein Sample, x_A und x_B
        - Ein Sample: Ein Pixel mit der gewählten Klasse im GT 
        -> (C,)

        3. Der Projektionsvektor P ist x_A - x_B
        - X_hat = P * X
        -> (C,)

        4. Wiederhole das K mal für jeden Ensemble-Member
        -> channel_modulation: (C, K)
        """
        
        C = Dataset_class.MODALITIES_TOTAL_CHANNELS

        if K is None:
            K = C

        dataset = Dataset_class(TRAIN, augment=False)

        all_modulations = []

        # Prepare class-wise seperation
        # For efficiency, only load dataset once
        full_data = []
        # (DATASET_SIZE, N_CHANNELS, SIZE, SIZE)

        full_mask_indxs_per_class = [[] for _ in Dataset_class.CLASS_NUMBERS]
        # (NUM_CLASSES, DATASET_SIZE, NUM_CLASS_PIXELS, 2)
        
        print("Reading full dataset into memory and saving per class seperations...")
        
        for i in range(len(dataset)):
            data, mask = dataset[i]
            full_data.append(data)

            for class_num in Dataset_class.CLASS_NUMBERS:
                curr_class_mask = (mask == class_num)
                # (H, W) with 1 where class

                # All idxs of pixels with this class
                curr_class_indxs = torch.nonzero(curr_class_mask)
                # (NUM_CLASS_PIXELS, 2) or (0, 2) if empty

                idx = class_num - 1
                full_mask_indxs_per_class[idx].append(curr_class_indxs)
        
        # Check all classes present
        for class_idx, indxs_list in enumerate(full_mask_indxs_per_class):
            # indxs_list: (DATASET_SIZE, NUM_CLASS_PIXELS, 2)
            # if not any data img in indxs_list has class, its empty

            if all(len(indxs) == 0 for indxs in indxs_list):
                raise ValueError(f"No samples of class {class_idx + 1} found in dataset")

        # To a single Tensor
        full_data = torch.stack(full_data)
        # (DATASET_SIZE, N_CHANNELS, SIZE, SIZE)

        print("Precomputing flat pixel lists for each class...")

        # Cannot stack full_mask_indxs_per_class here since lens differ
        # Precompute flat pixel lists: for each class, list of (img_idx, h, w) tuples
        flat_pixels_per_class = []
        for class_idx, indxs_list in enumerate(full_mask_indxs_per_class):
            flat_pixels = []
            for img_idx, pixel_indxs in enumerate(indxs_list):
                # pixel_indxs: (NUM_CLASS_PIXELS, 2) with (h, w) coordinates
                for h_idx, w_idx in pixel_indxs:
                    flat_pixels.append((img_idx, int(h_idx), int(w_idx)))
            flat_pixels_per_class.append(flat_pixels)

        print("Generating modulations...")

        for i in range(count_models):
            modulation = torch.zeros((C, K), dtype=torch.float32)

            for k in range(K):
                class_A = NP_RAND.choice(Dataset_class.CLASS_NUMBERS)
                
                rest_classes = Dataset_class.CLASS_NUMBERS.copy()
                rest_classes.remove(class_A)
                class_B = NP_RAND.choice(rest_classes)

                # Sample random pixels from each class
                class_A_idx = class_A - 1
                class_B_idx = class_B - 1

                pixel_A_idx = NP_RAND.integers(len(flat_pixels_per_class[class_A_idx]))
                pixel_B_idx = NP_RAND.integers(len(flat_pixels_per_class[class_B_idx]))

                pixel_A = flat_pixels_per_class[class_A_idx][pixel_A_idx]
                pixel_B = flat_pixels_per_class[class_B_idx][pixel_B_idx]

                img_idx_A, h_A, w_A = pixel_A
                img_idx_B, h_B, w_B = pixel_B

                x_A = full_data[img_idx_A, :, h_A, w_A]  # (C,)
                x_B = full_data[img_idx_B, :, h_B, w_B]  # (C,)

                # Projection vector: P = x_A - x_B
                modulation[:, k] = x_A - x_B

            all_modulations.append(modulation)

        return all_modulations


    @staticmethod
    def random_classes_LDA(Dataset_class, count_models, subset_percentage=0.75):
        """ 
        # Linear Discriminant Analysis
        
        """
        # TODO implement
        pass


    """ # UNUSED: Same results as with random_channel_drop
    @staticmethod
    def random_drop_percentage(Dataset_class, count_models, percentage=0.33):
        '''
        Randomly drop percentage of channels.

        -> (MODELS, C, variable K)
        - K same for all models
        '''
        C = Dataset_class.MODALITIES_TOTAL_CHANNELS

        all_modulations = []

        for i in range(count_models):
            # Start with all channels selected
            keep_mask = np.full((C,), 1)

            # Percentage of channels, ceil for at least 1 drop
            num_channels_to_drop = int(np.ceil(C * percentage))
            # Without repetition using replace=False
            drop_indices = NP_RAND.choice(C, size=num_channels_to_drop, replace=False)

            keep_mask[drop_indices] = 0

            kept_indices = np.nonzero(keep_mask)[0]
            K = len(kept_indices)

            modulation = torch.zeros((C, K), dtype=torch.float32)
            for j, idx in enumerate(kept_indices):
                modulation[idx, j] = 1.0
            all_modulations.append(modulation)

        return all_modulations """