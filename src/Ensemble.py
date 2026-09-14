from torch.utils.data.dataset import Dataset
from torch.utils.data import RandomSampler
from torchmetrics.classification import MulticlassAccuracy, MulticlassCohenKappa

from src.base import *
from src.Channels import *
from src.ModelManager import *
from src.metrics import shannon_entropy


def get_ensemble_name(dataset_name, func_name, modulation_opts, n_models):
    date = datetime.now().strftime("%d_%m_%y")

    base_name = f"{date}-{func_name}-{n_models}m"

    # Add all modulation ops if given
    if modulation_opts:
        base_name += "-"
        # Format ops: eg. a=0.5_b=func_name
        base_name += "_".join(f"{k}={v.__name__ if callable(v) else v}" for k, v in modulation_opts.items())

    base_name += f"-{dataset_name}"

    save_name = get_unique_save_name(ENSEMBLES_SAVE_DIR, base_name)

    return save_name

    
def get_ensemble_model_save_paths(ensemble_folder, count):
    return [f"{ensemble_folder}models/{i}" for i in range(count)]


class Ensemble():
    """ 
    Ensemble of Models each with different channels
    - dataset: 
    - channel_func: 

    count_models optional, since some channel_funcs determine it 
    """

    def __init__(self, Dataset_class, Model_class, channel_modulation_func, count_models=None, modulation_opts={},
                    reload_just_stats=False, reload_channel_modulations=None, reload_name=None):
        
        # Allow reloading from file

        self.Dataset_class = Dataset_class
        self.Model_class = Model_class
        self.channel_modulation_func = channel_modulation_func
        self.modulation_opts = modulation_opts
        self.just_stats = reload_just_stats

        # -- Channel Modulation
        if reload_channel_modulations is None:
            self.channel_modulations = channel_modulation_func(self.Dataset_class, count_models=count_models, 
                                                                **modulation_opts)
        else:
            self.channel_modulations = reload_channel_modulations
        # -> (MODELS, C, K)

        self.n_models = len(self.channel_modulations)

        # Testset
        # Without channels_modulation, individual modulations per model are applied during test
        self.test_dataset = self.Dataset_class(TEST, augment=False)
        
        """ # UNUSED: Improve speed by only loading used channels
        - has to be determined on init if channel_alter_matrix only inlcudes 0 and 1
        - if so, can be converted to list of channel indices per modality 
        # Channel indices per modality, to pass directly to DFC18Dataset src.read() 
        self.channels_modalities_indices = [channel_mask_to_modality_indices(mask, self.dataset) for mask in channel_masks] """

        # -- Name and Paths
        if reload_name is None:
            self.name = get_ensemble_name(self.Dataset_class.NAME_SHORT, self.channel_modulation_func.__name__, 
                                            modulation_opts, self.n_models)
        else:
            self.name = reload_name

        print(f"\n\nEnsemble Name: \n{self.name}\n\n")

        self.save_dir_path = ENSEMBLES_SAVE_DIR + self.name + "/"

        save_paths = get_ensemble_model_save_paths(self.save_dir_path, self.n_models)

        # -- Models
        self.models = []

        # Mode all self.channel_modulations to CPU
        self.channel_modulations = [m.cpu() for m in self.channel_modulations]

        # Save Ensemble meta if not reloaded
        if reload_name is None:
            self.save_to_file()

        # Check if training was already done for all models
        self.was_fully_reloaded = True

        for modulation, path in zip(self.channel_modulations, save_paths):
            # modulation: (C, K)
            # If saved models train stats exists, reload from file
            # (dont check for model, since might be "just_stats")
            if os.path.exists(path + MODEL_METADATA_TRAIN_SUFFIX):
                model = ModelManager.from_file(modulation, path, self.Dataset_class, self.Model_class, just_stats=reload_just_stats)

            # Create new model
            else:
                model = ModelManager(modulation, path, self.Dataset_class, self.Model_class)
                self.was_fully_reloaded = False

            self.models.append(model)

        # Default device
        self.device = DEVICE
        
        # -- Loss for testing whole ensemble
        class_weights = Dataset_class.CLASS_WEIGHTS.to(self.device)
        self.criterion = nn.CrossEntropyLoss(weight=class_weights, ignore_index=0)

        # -- Stats
        self.test_stats = None


    @classmethod
    def load_from_file(cls, path, just_stats=False):
        ''' Recover Ensemble from saved file
        
        Allow all of the following for path:
        ./ensembles/DFC18Dataset-01_04_26-func_powerset_of_all_modalities/
        /DFC18Dataset-01_04_26-func_powerset_of_all_modalities/
        DFC18Dataset-01_04_26-func_powerset_of_all_modalities
        DFC18Dataset-01_04_26-func_powerset_of_all_modalities/meta.pth
        '''
        # "cls" is static version of this class, no instance

        if not os.path.exists(path):
            # Check if path is just folder name
            path = os.path.join(ENSEMBLES_SAVE_DIR, path)

        if not os.path.exists(path):
            raise ValueError(f"No Ensemble found at: {path}")

        if not path.endswith(ENSEMBLE_METADATA_FILENAME):
            path = os.path.join(path, ENSEMBLE_METADATA_FILENAME)

        print("Loading Ensemble from file:", path)
        meta = torch.load(path, weights_only=False, map_location=torch.device('cpu'))

        # Dataset: meta["dataset"]
        dataset_name = meta["dataset"]
        dataset = None

        if dataset_name in globals() and issubclass(globals()[dataset_name], Dataset):
            dataset = globals()[dataset_name]
            print("Dataset found:", dataset)
        else:
            # Try legacy name with "Dataset" suffix 
            dataset_name_legacy = dataset_name + "Dataset"
            if dataset_name_legacy in globals() and issubclass(globals()[dataset_name_legacy], Dataset):
                dataset = globals()[dataset_name_legacy]
                print("Dataset found (legacy):", dataset)
            else:
                raise ValueError(f"Dataset does not exist: {dataset_name} or {dataset_name_legacy}")


        # Model: meta["model"]
        # Default to UNet
        model = UNet
        
        if "model" in meta:
            model_name = meta["model"]

            if model_name in globals() and issubclass(globals()[model_name], nn.Module):
                model = globals()[model_name]
                print("Model found:", model)


        # channel_func: meta["channel_modulation_func"]
        channel_func_name = meta["channel_modulation_func"]
        channel_func = None

        if hasattr(Channel_Modulation, channel_func_name):
            channel_func = getattr(Channel_Modulation, channel_func_name)
            print("Channel selection function found:", channel_func)
        else:
            raise ValueError(f"Channel_Modulation function does not exist: {channel_func_name}")

        # modulation_opts: meta["modulation_opts"] - .get() so ensembles saved
        # before this key existed still load fine, just with no opts on record
        modulation_opts = meta.get("modulation_opts", {})

        # Channel Modulation
        channel_modulations = meta["channel_modulations"]

        # Convert to torch tensors and float32 (if not already in that format)
        channel_modulations = [m.float() if torch.is_tensor(m) else torch.tensor(m, dtype=torch.float32) for m in channel_modulations]
        print(f"Channel modulations loaded, len: {len(channel_modulations)} shape of [0]: {channel_modulations[0].shape}")

        # Parent folder name is ensemble name
        reloaded_name = os.path.basename(os.path.dirname(path))
        print("Reloaded name:", reloaded_name)
            
        # new instance returned by calling __init__ of class
        new_instance = cls(dataset, model, channel_func, modulation_opts=modulation_opts, reload_just_stats=just_stats,
                            reload_channel_modulations=channel_modulations, reload_name=reloaded_name)

        # Load test stats
        if "test_time" in meta:
            new_instance.test_stats = meta
            print("Test stats loaded from file")

        return new_instance


    def save_to_file(self):
        ''' 
        Save Ensemble meta to file.
        Not Models: They save themselves directly after each is trained / tested.
        '''

        # create all folders along path, exist_ok: do not throw error if exists
        os.makedirs(f"{self.save_dir_path}models", exist_ok=True)

        meta_path = self.save_dir_path + ENSEMBLE_METADATA_FILENAME

        channel_modulations = [m.cpu() for m in self.channel_modulations]

        metadata = {
            "dataset": self.Dataset_class.NAME,
            "model": self.Model_class.__name__,
            "channel_modulation_func": self.channel_modulation_func.__name__,
            "modulation_opts": self.modulation_opts,  # save just in case for diagnosis
            "channel_modulations": channel_modulations
        }

        torch.save(metadata, meta_path)
        print("Saved metadata to:", meta_path)


    def train(self, model_nums=None):
        """
        Train all untrained models, or only those in model_nums if given.
        Until model converged OR until MAX_EPOCHS.
        """

        if self.was_fully_reloaded:
            print("Ensemble not trained nor saved since was fully reloaded from file.")
            return

        if model_nums is not None:
            invalid = [n for n in model_nums if n < 0 or n >= len(self.models)]
            if invalid:
                raise ValueError(
                    f"train_model_nums contains out-of-range indices {invalid} "
                    f"for ensemble with {len(self.models)} models (valid range: 0-{len(self.models) - 1})")

        for idx, model in enumerate(self.models):
            if model_nums is not None and idx not in model_nums:
                continue

            if not model.is_trained:
                model.device = DEVICE

                print(f"\n[!] Model Nr. {idx} of models [0 - {len(self.models) - 1}]")
                model.train()
                print(f"\n[Yay] Finished Model Nr. {idx} of models [0 - {len(self.models) - 1}]")

        print("\nAll training completed!" if model_nums is None
              else f"\nRequested subset of models {model_nums} completed (or already trained)!")

    
    def check_all_models_available(self):
        """ Check if all models in ensemble are trained and not just_stats """
        # Check all models trained and not just_stats
        if any(m.is_just_stats for m in self.models):
            raise ValueError("Some models are just_stats")

        if not all(m.is_trained for m in self.models):
            raise ValueError("Not all models trained yet")


    def predict(self, input_tensor_batch, return_model_outputs=False):
        """ 
        Get prediction on input tensor for all models in ensemble 
        Apply each model's channel modulation to input tensor before prediction
        
        Either one image or batch
        """

        self.check_all_models_available()

        input_tensor_batch = input_tensor_batch.to(DEVICE)
        # modulations already are on DEVICE
        
        predictions = []
        # (N_MODELS, BATCH_SIZE, NUM_CLASSES, SIZE, SIZE)

        # Channel Modulation
        for model, modulation in zip(self.models, self.channel_modulations):
            # TODO scope
            model.model.to(DEVICE) 
            modulation = modulation.to(DEVICE)

            # Multiply all channels with channels_modulation
            modulated_input_batch = Channel_Modulation.apply_modulation(input_tensor_batch, modulation)
            # ((C, H, W).T @ (C, K)).T -> (K, H, W)

            # Unsqueeze single modulated_input to batch dim
            if modulated_input_batch.dim() == 3:
                modulated_input_batch = modulated_input_batch.unsqueeze(0)
                # (C, H, W) -> (1, C, H, W)

            prediction_batch = model.predict(modulated_input_batch)
            # -> "class logits": (BATCH_SIZE, NUM_CLASSES, SIZE, SIZE)

            prediction_batch = torch.softmax(prediction_batch, dim=1)
            # -> "probalities": still same shape, just bounded to [0, 1]

            predictions.append(prediction_batch)

        # Covert to np array
        predictions = torch.stack(predictions)
        # -> (N_MODELS, BATCH_SIZE, NUM_CLASSES, SIZE, SIZE)

        # Average (probalities) of all models for each input in batch
        # eg (8, 12, 20, 128, 128) -> (12, 20, 128, 128)
        # torch complains without dtype 
        ensemble_prediction_batch = predictions.mean(dim=0, dtype=torch.float32)
        # -> (BATCH_SIZE, NUM_CLASSES, SIZE, SIZE)

        if return_model_outputs:
            return ensemble_prediction_batch, predictions

        return ensemble_prediction_batch


    def test(self, subsample_count=None):
        """
        Input data to all models, then take mean.
        Testset is whole testset OR subset with size subsampleCount 

        Calculate OA, AA, DICE and Loss
        - Each for each model and for Ensemble (mean)
        """

        # Check all models trained and not just_stats
        self.check_all_models_available()

        # TODO SPEEDUP: Calc only once per image for OA, AA
        # TODO REFACTOR into ModelManager class and test.py run_test()

        # Subsample testset if sample_size given
        if subsample_count is None or subsample_count > len(self.test_dataset):
            test_dataloader = DataLoader(dataset=self.test_dataset, **DATALOADER_ARGS)

        else:
            testset_sampler = RandomSampler(self.test_dataset, num_samples=subsample_count, generator=generator)
            
            test_dataloader = DataLoader(dataset=self.test_dataset, sampler=testset_sampler, batch_size=BATCH_SIZE)


        n_classes     = self.Dataset_class.NUM_CLASSES - 1 # exclude class 0
        n_all_classes = self.Dataset_class.NUM_CLASSES     # include class 0

        # Metrics accumulators for each model and ensemble

        # Loss
        loss_ensemble = 0
        loss_models = torch.zeros(self.n_models)

        # OA, AA
        # Correctly labeled pixels, total labeled pixels, per class and overall
        corr_lb_pix_ensemble = 0
        tot_lb_pix_ensemble = 0
        corr_lb_pix_per_class_ensemble = torch.zeros(n_classes)
        tot_lb_pix_per_class_ensemble = torch.zeros(n_classes)

        corr_lb_pix_models = torch.zeros(self.n_models)
        tot_lb_pix_models = torch.zeros(self.n_models)
        corr_lb_pix_per_class_models = torch.zeros((self.n_models, n_classes))
        tot_lb_pix_per_class_models = torch.zeros((self.n_models, n_classes))

        # torchmetrics OA, AA
        #  validate own implementation above
        #  Include class 0 in num_classes
        #  average="micro" -> OA (globally pooled accuracy)
        #  average="macro" -> AA (mean of per-class accuracy)
        #  Stateful like Kappa below: .update() accumulates stats per image
        #  At end: .compute() 
        #  https://lightning.ai/docs/torchmetrics/stable/classification/accuracy.html
        
        oa_metric_ensemble = MulticlassAccuracy(num_classes=n_all_classes, ignore_index=0, average="micro").to(DEVICE)
        aa_metric_ensemble = MulticlassAccuracy(num_classes=n_all_classes, ignore_index=0, average="macro").to(DEVICE)
        oa_metric_models   = [MulticlassAccuracy(num_classes=n_all_classes, ignore_index=0, average="micro").to(DEVICE)
                                for _ in range(self.n_models)]
        aa_metric_models   = [MulticlassAccuracy(num_classes=n_all_classes, ignore_index=0, average="macro").to(DEVICE)
                                for _ in range(self.n_models)]

        # Kappa
        kappa_metric_ensemble = MulticlassCohenKappa(num_classes=n_all_classes, ignore_index=0).to(DEVICE)
        kappa_metric_models = [MulticlassCohenKappa(num_classes=n_all_classes, ignore_index=0).to(DEVICE)
                                for _ in range(self.n_models)]

        # DICE
        dice_per_class_ensemble = torch.zeros(n_classes)
        dice_per_class_models = torch.zeros((self.n_models, n_classes))

        num_batches = 0
        num_samples = 0

        # Set models to eval mode
        with torch.no_grad():
            for m in self.models:
                m.ensure_model_loaded()
                m.model.eval()

            for data_batch in tqdm(test_dataloader, desc="Testing Ensemble", leave=True):
                num_batches += 1

                img_batch = data_batch[0].float().to(DEVICE)
                mask_batch = data_batch[1].long().to(DEVICE)

                ensemble_prediction_batch, model_predictions_batch = self.predict(img_batch, return_model_outputs=True)
                # -> (BATCH_SIZE, NUM_CLASSES, SIZE, SIZE)

                # STATS OF ENSEMBLE
                
                # Loss
                #  For each batch
                loss = self.criterion(ensemble_prediction_batch.to(DEVICE), mask_batch)
                loss_ensemble += loss.item()

                # ENSEMBLE: For each img in batch
                for mask, pred, img in zip(mask_batch, ensemble_prediction_batch, img_batch):
                    num_samples += 1

                    predicted_classes = torch.argmax(pred, dim=0)
                    # (NUM_CLASSES, SIZE, SIZE) -> (SIZE, SIZE)

                    predicted_flat = predicted_classes.flatten().to(DEVICE) 
                    mask_flat      = mask.flatten().to(DEVICE)
                    # To 1D: (SIZE, SIZE) -> (SIZE^2)

                    # OO, AA: 
                    curr_correct, curr_total, curr_correct_per_class, curr_total_per_class = get_correct_pixels_count(predicted_flat, mask_flat, self.Dataset_class.NUM_CLASSES)

                    corr_lb_pix_ensemble += curr_correct
                    tot_lb_pix_ensemble += curr_total

                    corr_lb_pix_per_class_ensemble += curr_correct_per_class
                    tot_lb_pix_per_class_ensemble += curr_total_per_class

                    # torchmetrics OA, AA
                    oa_metric_ensemble.update(predicted_flat, mask_flat)
                    aa_metric_ensemble.update(predicted_flat, mask_flat)

                    # Kappa
                    kappa_metric_ensemble.update(predicted_flat, mask_flat)

                    # DICE
                    dice_per_class_ensemble += dice_coefficient_multiclass(predicted_flat, mask_flat, n_all_classes, already_flattened=True)


                # STATS OF INDIVIDUAL MODELS
                for model_i, pred_batch in enumerate(model_predictions_batch):
                    # Loss of whole batch for model_i
                    loss = self.models[model_i].criterion(pred_batch.to(DEVICE), mask_batch)
                    loss_models[model_i] += loss.item()

                    # For each img in batch
                    for mask, pred in zip(mask_batch, pred_batch):
                        
                        predicted_classes = torch.argmax(pred, dim=0)
                        # (NUM_CLASSES, SIZE, SIZE) -> (SIZE, SIZE)

                        predicted_flat = predicted_classes.flatten().to(DEVICE) 
                        mask_flat    = mask.flatten().to(DEVICE)
                        # To 1D: (SIZE, SIZE) -> (SIZE^2)

                        # OO, AA: 
                        curr_correct, curr_total, curr_correct_per_class, curr_total_per_class = get_correct_pixels_count(predicted_flat, mask_flat, self.Dataset_class.NUM_CLASSES)

                        corr_lb_pix_models[model_i] += curr_correct
                        tot_lb_pix_models[model_i] += curr_total

                        corr_lb_pix_per_class_models[model_i] += curr_correct_per_class
                        tot_lb_pix_per_class_models[model_i] += curr_total_per_class

                        # torchmetrics OA, AA
                        oa_metric_models[model_i].update(predicted_flat, mask_flat)
                        aa_metric_models[model_i].update(predicted_flat, mask_flat)

                        # Kappa
                        kappa_metric_models[model_i].update(predicted_flat, mask_flat)


                        # DICE
                        dice_per_class_models[model_i] += dice_coefficient_multiclass(predicted_flat, mask_flat, n_all_classes, already_flattened=True)

                        """ 
                        # TODO speedup: 
                        #   Is equal:
                        #       tot_lb_pix_ensemble and tot_lb_pix_models (just as list)
                        #       tot_lb_pix_per_class_ensemble and tot_lb_pix_per_class_models (just as list)
                        #   Calc only once per image:
                        #       target_flat, valid_pixels_mask, target_flat_valid, total_valid_pixels, total_per_class, 
                        """


        # ENSEMBLE
        loss_ensemble = loss_ensemble / num_batches
        
        overall_accuracy_ensemble = corr_lb_pix_ensemble / tot_lb_pix_ensemble

        accuracy_in_classes_ensemble = corr_lb_pix_per_class_ensemble / tot_lb_pix_per_class_ensemble # (NUM_CLASSES)
        average_accuracy_ensemble = sum(accuracy_in_classes_ensemble) / n_classes

        overall_accuracy_torchmetrics_ensemble = oa_metric_ensemble.compute()
        average_accuracy_torchmetrics_ensemble = aa_metric_ensemble.compute()

        average_kappa_ensemble = kappa_metric_ensemble.compute()

        dice_per_class_ensemble_res = dice_per_class_ensemble / num_samples
        average_dice_ensemble = sum(dice_per_class_ensemble_res) / n_classes


        # MODELS
        loss_models = loss_models / num_batches

        overall_accuracy_models = corr_lb_pix_models / tot_lb_pix_models

        accuracy_in_classes_models = corr_lb_pix_per_class_models / tot_lb_pix_per_class_models # (N_MODELS, NUM_CLASSES)
        average_accuracy_models = accuracy_in_classes_models.sum(dim=1) / n_classes # (N_MODELS,)

        overall_accuracy_torchmetrics_models = torch.stack([m.compute() for m in oa_metric_models])
        average_accuracy_torchmetrics_models = torch.stack([m.compute() for m in aa_metric_models])

        average_kappa_models = torch.stack([m.compute() for m in kappa_metric_models])
        
        dice_per_class_models_res = dice_per_class_models / num_samples
        average_dice_models = dice_per_class_models_res.sum(dim=1)  / n_classes


        # ENSEMBLE:
        print(f"\nENSEMBLE RESULTS:")
        print(f"OA: {overall_accuracy_ensemble:.4f}, AA: {average_accuracy_ensemble:.4f}")
        print(f"torchmetrics OA: {overall_accuracy_torchmetrics_ensemble:.4f}, AA: {average_accuracy_torchmetrics_ensemble:.4f}")
        print(f"test_loss: {loss_ensemble:.4f}, random guessing would be ln({n_classes}) = {torch.log(torch.tensor(n_classes - 1))} since testset is (almost) class balanced")
        print(f"Kappa: {average_kappa_ensemble:.4f}")
        print(f"Average DICE: {average_dice_ensemble:.4f}")

        # Save ensemble results to meta file
        self.save_test_stats_to_file(test_stats={
            "test_time": datetime.now().strftime("%d_%m_%y-%H:%M:%S"),
            "loss": loss_ensemble,
            "overall_accuracy": overall_accuracy_ensemble,
            "accuracy_in_classes": accuracy_in_classes_ensemble,
            "average_accuracy": average_accuracy_ensemble,
            "overall_accuracy_torchmetrics": overall_accuracy_torchmetrics_ensemble,
            "average_accuracy_torchmetrics": average_accuracy_torchmetrics_ensemble,
            "kappa": average_kappa_ensemble,
            "dice_per_class": dice_per_class_ensemble_res,
            "average_dice": average_dice_ensemble,
        })

        
        # MODELS:
        for model_i in range(self.n_models):
            print(f"\nMODELS {model_i} RESULTS:")
            print(f"OA: {overall_accuracy_models[model_i]:.4f}, AA: {average_accuracy_models[model_i]:.4f}")
            print(f"torchmetrics OA: {overall_accuracy_torchmetrics_models[model_i]:.4f}, AA: {average_accuracy_torchmetrics_models[model_i]:.4f}")
            print(f"test_loss: {loss_models[model_i]:.4f}")
            print(f"Kappa: {average_kappa_models[model_i]:.4f}")
            print(f"Average DICE: {average_dice_models[model_i]:.4f}")

            # TODO in refactoring
            test_stats = {
                "test_time": datetime.now().strftime("%d_%m_%y-%H:%M:%S"),
                "loss": loss_models[model_i],
                "overall_accuracy": overall_accuracy_models[model_i],
                "accuracy_in_classes": accuracy_in_classes_models[model_i],
                "average_accuracy": average_accuracy_models[model_i],
                "overall_accuracy_torchmetrics": overall_accuracy_torchmetrics_models[model_i],
                "average_accuracy_torchmetrics": average_accuracy_torchmetrics_models[model_i],
                "kappa": average_kappa_models[model_i],
                "dice_per_class": dice_per_class_models_res[model_i],
                "average_dice": average_dice_models[model_i],
            }
            self.models[model_i].test_stats = test_stats
            self.models[model_i].save_test_stats_to_file()


    def test_predict_full_dataset(self):
        """ Predict on each test image, combine to single prediction matrix
        Return for each model and ensemble list of predictions for each test image, in order of test set

        -> (1 + N_MODELS, N_TEST_IMAGES, 2, H, W)
        - per-pixel predicted class index
        - per-pixel entropy
        """

        self.check_all_models_available()

        test_dataloader = DataLoader(dataset=self.test_dataset, **DATALOADER_ARGS)

        all_preds = [[] for _ in range(self.n_models + 1)]
        # 1 for ensemble, N_MODELS for models

        def class_and_entropy(probs):
            """ probs: (NUM_CLASSES, H, W) softmax probabilities for one tile
            -> (2, H, W) stack of [class index, entropy] """
            class_pred = torch.argmax(probs, dim=0).float()
            entropy = shannon_entropy(probs, dim=0)
            return torch.stack([class_pred, entropy]).cpu().numpy()

        with torch.no_grad():
            for m in self.models:
                m.model.eval()

            for data_batch in tqdm(test_dataloader, desc="Predicting whole test dataset", leave=True):
                # data_batch: ((BATCH_SIZE, C, H, W), (BATCH_SIZE, H, W))

                img_batch = data_batch[0].float().to(DEVICE)
                # (BATCH_SIZE, C, H, W)

                ensemble_pred_batch, models_pred_batch = self.predict(img_batch, return_model_outputs=True)
                # ensemble_pred_batch: (BATCH_SIZE, NUM_CLASSES, H, W)
                # models_pred_batch: (N_MODELS, BATCH_SIZE, NUM_CLASSES, H, W)

                for ensemble_pred in ensemble_pred_batch:
                    all_preds[0].append(class_and_entropy(ensemble_pred))

                for i, model_pred_batch in enumerate(models_pred_batch):
                    for model_pred in model_pred_batch:
                        all_preds[i + 1].append(class_and_entropy(model_pred))

        # -> (1 + N_MODELS, N_TEST_IMAGES, 2, H, W)

        # Save to file
        path = os.path.join(self.save_dir_path, ENSEMBLE_FULL_TEST_FILENAME)

        torch.save(all_preds, path)
        print("Saved ensemble test_predict_full_dataset results to:", path)


    def save_test_stats_to_file(self, test_stats):
        """ Add to meta.pth """

        meta_path = os.path.join(self.save_dir_path, ENSEMBLE_METADATA_FILENAME)
        meta = torch.load(meta_path, weights_only=False)

        for key, value in test_stats.items():
            meta[key] = value

        torch.save(meta, meta_path)
        print("Saved ensemble test stats to:", meta_path)


    def show_predict_dataset_index(self, idx):
        """ Working, but doesnt look good since channels are mixed. Duh. """
        
        data = self.test_dataset[idx]
        
        img = data[0].float().to(DEVICE)
        mask = data[1].long().to(DEVICE)

        ensemble_pred, models_preds = self.predict(img, 
                return_model_outputs=True)

        for pred in ensemble_pred + models_preds:

            predicted_classes = torch.argmax(pred, dim=0)

            # Plot input img and pred
            convert_and_plot_img_data(img, self.Dataset_class, mask=mask, 
                    pred=predicted_classes, save=True)


# datasetName, channelSelectionFunc

# save_to_file()
# load_from_file()

""" # Check channel masks:
# [('RGB',), ('HSI',), ('LIDAR',), ('RGB', 'HSI'), ('RGB', 'LIDAR'), ('HSI', 'LIDAR'), 
# ('RGB', 'HSI', 'LIDAR')]

print(*emsemble1.channels_modalities_indices, sep="\n") """

#

""" print(*emsemble1.channels_modalities_indices, sep="\n")



a = emsemble1.models[1].train_dataset[0]

a[0].shape """
