from src.base import *

from src.DFC18_Dataset import *
from src.MDAS_Dataset import *

from src.UNet import *
from src.train import *
from src.plot import *

from torch import optim
from torch.utils.data import random_split
from torch.utils.data import DataLoader

# Use persistent_workers ?!

class ModelManager():
    def __init__(self, channels_modulation, save_path, Dataset_class, Model_class, just_stats=False):
        """ Initialize ModelManager
        Only setup (or move to GPU) some things later in train to save space on GPU
        
        Expects to be set externally:
        DEVICE
        DEFAULT_LEARNING_RATE
        """

        # channels_modulation: (C, K)
        self.channels_modulation = channels_modulation
        self.channels_count = channels_modulation.shape[1]

        # Train Stats: train_losses, train_dcs, val_losses, val_dcs, lrs
        self.train_stats = None
        # Test Stats: test_loss, test_oa, test_aa
        self.test_stats = None
        self.is_just_stats = just_stats
        self.is_trained = False

        # Skip loading model etc. if just_stats, saves time & GPU
        if self.is_just_stats:
            return

        self.Dataset_class = Dataset_class
        self.Model_class = Model_class

        # Default device
        self.device = DEVICE

        # "Full" being the part of the Dataset that is supposed to be used for train, that includes validation
        full_dataset = self.Dataset_class(TRAIN, channels_modulation=self.channels_modulation, augment=True)

        # Train and val part of Train Data
        self.train_dataset, self.val_dataset = random_split(full_dataset, lengths=[0.8, 0.2], generator=generator)
        """ print("DATASETS lens:", len(self.train_dataset), len(self.val_dataset)) """

        self.train_dataloader = DataLoader(dataset=self.train_dataset, shuffle=True, **DATALOADER_ARGS)
        self.val_dataloader = DataLoader(dataset=self.val_dataset, shuffle=True, **DATALOADER_ARGS)

        # -- Model, load only when needed
        self.save_model_path = save_path + ".pth"
        self.model = None
        # Get model size (param count)
        # trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
        # print("Model params:", trainable_params)

        # -- Loss
        # reduction standard: reduction='mean' 
        # Weight classes against imbalance & ignore class 0 for loss
        class_weights = Dataset_class.CLASS_WEIGHTS.to(self.device)
        self.criterion = nn.CrossEntropyLoss(weight=class_weights, ignore_index=0)


    @classmethod
    def from_file(cls, channels_modulation, model_path, Dataset_class, Model_class, just_stats=False):

        # save_path should be the path WITHOUT suffix
        save_path = model_path.replace(".pth", "")

        # model_path should be the path WITH suffix
        if not model_path.endswith(".pth"):
            model_path += ".pth"

        metadata_train_path = save_path + MODEL_METADATA_TRAIN_SUFFIX

        """ print(f"Loading saved {'stats from file' if just_stats else 'model & stats from files'}: {save_path}") """

        # Create new instance
        new_instance = cls(channels_modulation, save_path, Dataset_class, Model_class, just_stats=just_stats)

        # Load train stats
        if not os.path.exists(metadata_train_path):
            raise FileNotFoundError(f"Train stats file not found: {metadata_train_path}")

        # Load all stats to cpu
        new_instance.train_stats = torch.load(metadata_train_path, weights_only=False, map_location=torch.device('cpu'))

        # Load model weights if not just_stats
        if not new_instance.is_just_stats:
            if not os.path.exists(model_path):
                raise FileNotFoundError(f"Model file not found: {model_path}\n\n Maybe you want to load with just_stats=True ?")

            new_instance.ensure_model_loaded()

        # Try to load test stats if it exists
        metadata_test_path = save_path + MODEL_METADATA_TEST_SUFFIX
        if os.path.exists(metadata_test_path):
            new_instance.test_stats = torch.load(metadata_test_path, weights_only=False, map_location=torch.device('cpu'))

        return new_instance


    def ensure_model_loaded(self, model_file_has_to_exist=True):
        if self.model is None:
            self.model = self.Model_class(in_channels=self.channels_count, num_classes=self.Dataset_class.NUM_CLASSES)
            
            model_exists = os.path.exists(self.save_model_path)
            
            if model_file_has_to_exist and not model_exists:
                raise ValueError(f"Model file not found: {self.save_model_path}")

            if model_exists:
                # weights_only to avoid 'Maclicous' warning
                self.model.load_state_dict(torch.load(self.save_model_path, weights_only=True, map_location=self.device))
                self.is_trained = True

            self.model = self.model.to(self.device)

    def train(self):
        """ Train all models until converted or until MAX_EPOCHS reached, save models and stats to file """

        if self.is_just_stats:
            print("ERROR: Cannot train Model initialized with just_stats=True")
            return

        # Setup model and if exist load from file since, but okay if not
        self.ensure_model_loaded(model_file_has_to_exist=False)

        # Transfer over to CORRECT! GPU only now that needed
        self.criterion = self.criterion.to(self.device)
        
        # -- Optimizer: standard Adam with weight decay
        self.optimizer = optim.AdamW(self.model.parameters(), lr=DEFAULT_LEARNING_RATE, weight_decay=0.01) # default 

        ''' # Alternative SGD (untested)
        momentum = 0.9
        self.optimizer = optim.SGD(self.model.parameters(), lr=DEFAULT_LEARNING_RATE, momentum=momentum) '''

        # -- LR Scheduler: Reduce lr over epochs
        patience = 10
        self.lr_scheduler = optim.lr_scheduler.ReduceLROnPlateau(self.optimizer, patience=patience,
                                                                  factor=0.1, threshold=1e-4) # defaults
        ''' # Alternative: ExponentialLR
        gamma = 0.9 # standard value
        self.lr_scheduler = optim.lr_scheduler.ExponentialLR(self.optimizer, gamma=gamma) '''

        """ print("Pre- train memory usage:")
        print_memory_usage() """

        # TRAIN
        self.train_stats = run_train(self.train_dataloader, self.val_dataloader, self.model, 
                                self.criterion, self.optimizer, self.lr_scheduler, num_classes=self.Dataset_class.NUM_CLASSES,
                                device=self.device)

        self.is_trained = True

        # Save
        self.save_to_file()

        # After Train cleanup
        #  Delete from RAM / GPU MEM
        self.model = None
        del self.optimizer
        del self.lr_scheduler
        
        if torch.cuda.is_available():
            torch.cuda.empty_cache()

        if torch.backends.mps.is_available():
            print("Clearing MPS GPU")
            torch.mps.empty_cache()

        """ print("Post-train memory usage:")
        print_memory_usage() """


    def save_to_file(self):
        """ Save Model and its stats to file
        (channel_modulation per model is kept by Ensemble) """

        if self.is_just_stats:
            print("Model is JUST STATS, cannot save")
            return

        self.ensure_model_loaded()

        # Save model + Train stats
        if self.is_trained:
            # create all folders along path, exist_ok: do not throw error if exists
            parent_folder_path = "/".join(self.save_model_path.split("/")[:-1])
            os.makedirs(parent_folder_path, exist_ok=True)

            # Save model weights
            torch.save(self.model.state_dict(), self.save_model_path)
            print("SAVED model to\n", self.save_model_path)

            # Save train stats metadata
            metadata = self.train_stats
            """ metadata = {
                "train_stats": self.train_stats
            } """

            metadata_path = self.save_model_path.replace(".pth", MODEL_METADATA_TRAIN_SUFFIX)
            torch.save(metadata, metadata_path)
            print("Saved train stats to:", metadata_path)


    def save_test_stats_to_file(self):
        # Save Test stats metadata
        if self.test_stats is not None:
            metadata = self.test_stats
            metadata_test_path = self.save_model_path.replace(".pth", MODEL_METADATA_TEST_SUFFIX)
            torch.save(metadata, metadata_test_path)
            print("Saved test stats to:", metadata_test_path)

    
    def predict(self, input_tensor):
        """ 
        Get prediction on input tensor 
        Always with batch dim
        (BATCH_SIZE, 53, 512, 512) 
        -> (BATCH_SIZE, NUM_CLASSES, 512, 512)
        """

        if self.is_trained and not self.is_just_stats:
            self.ensure_model_loaded()
    
            # inference settings
            self.model.eval()

            with torch.no_grad():
                input_tensor = input_tensor.to(self.device)
                output = self.model(input_tensor)
                # move back to CPU for further processing
                return output.cpu()

        else:
            print("Model NOT TRAINED yet, cannot predict")
            return None

    
    # test() is implemented in Ensemble


# Checking
""" 
modelM1 = ModelManager(channels_per_modality=only_rgb_channels, save_path=save_path) 
modelM1.save_to_file()

modelM1.train_dataset[0]
# modelM2_loaded = ModelManager.from_file(
#   channels_per_modality=only_rgb_channels, save_path=save_path)

modelM1.train()
"""