import argparse
import json

from src.base import *

from src.DFC18_Dataset import *
from src.MDAS_Dataset import *
from src.UNet import *
from src.Channels import *

def parse_args():
    # Dataset:      DFC18Dataset, MDAS_Dataset
    # Model:        UNet (default, = UNet_big_5layers), UNet_standard_4layers,
    #               UNet_small_3layers, UNet_huge_6layers
    # channel_mod:  any staticmethod of Channel_Modulation, eg.
    #               identity, powerset_of_all_modalities, random_channel_drop,
    #               random_drop_percentage_of_modality, random_subsets_PCA,
    #               random_classes_simple_LDA, random_linear_combination

    def dataset_type(name):
        obj = globals().get(name)
        if not (isinstance(obj, type) and issubclass(obj, Dataset)):
            raise argparse.ArgumentTypeError(f"Unknown dataset: {name}")
        return obj

    def model_type(name):
        obj = globals().get(name)
        if not (isinstance(obj, type) and issubclass(obj, nn.Module)):
            raise argparse.ArgumentTypeError(f"Unknown model: {name}")
        return obj

    def channel_mod_type(name):
        if not hasattr(Channel_Modulation, name):
            raise argparse.ArgumentTypeError(f"Unknown channel_mod: {name}")
        return getattr(Channel_Modulation, name)

    def modulation_opts_type(value):
        opts = json.loads(value)

        # previous_modulation (eg. for random_subsets_PCA) is called directly
        # as a function in Channels.py - JSON has no notion of a callable, so
        # accept it as a channel_mod name string here and resolve it the same
        # way --channel_mod itself does.
        if "previous_modulation" in opts and isinstance(opts["previous_modulation"], str):
            opts["previous_modulation"] = channel_mod_type(opts["previous_modulation"])

        return opts

    def reload_type(value):
        # Either a single ensemble name, or a "[name,name,...]" list of them
        value = value.strip()
        if value.startswith("[") and value.endswith("]"):
            return [name.strip() for name in value[1:-1].split(",") if name.strip()]
        return [value]

    def train_model_nums_type(value):
        # Empty (or omitted) means "no restriction" -> train all untrained models.
        # Unlike reload_type, an empty "[,]" also collapses to None here since
        # "train zero models" isn't a meaningful case.
        value = value.strip()
        if not value:
            return None
        if not (value.startswith("[") and value.endswith("]")):
            raise argparse.ArgumentTypeError(f'--train_model_nums must be empty or "[n,n,...]", got: {value}')

        parts = [p.strip() for p in value[1:-1].split(",") if p.strip()]
        try:
            nums = [int(p) for p in parts]
        except ValueError:
            raise argparse.ArgumentTypeError(f"--train_model_nums must contain only integers, got: {value}")

        if not nums:
            return None
        if len(set(nums)) != len(nums):
            raise argparse.ArgumentTypeError(f"--train_model_nums contains duplicate indices: {value}")

        return sorted(nums)

    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset", default="DFC18Dataset", type=dataset_type)
    parser.add_argument("--model", default="UNet", type=model_type)
    parser.add_argument("--channel_mod", default="identity", type=channel_mod_type)
    parser.add_argument("--count_models", type=int, default=7)
    parser.add_argument("--just_stats", type=bool, default=False)
    # JSON dict for channel_mod-specific kwargs, eg. '{"drop_prob": 0.3}'.
    # previous_modulation may be given as a channel_mod name string, eg.
    # '{"previous_modulation": "powerset_of_all_modalities"}' - see modulation_opts_type
    parser.add_argument("--modulation_opts", type=modulation_opts_type, default={})
    parser.add_argument("--reload", type=reload_type, default=None,
                            help='Ensemble name(s) to reload instead of training a new one, '
                                    'eg. "name" or "[name1,name2]"')
    parser.add_argument("--train_model_nums", type=train_model_nums_type, default=None,
                            help='Restrict training to these model indices of the ensemble, '
                                    'eg. "[10,11,12,13]". Requires --reload. Default: train all untrained models.')

    return parser.parse_args()
