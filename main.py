print("START")

# Own Files
from src.base import *
from src.show import *

from src.DFC18_Dataset import *
from src.MDAS_Dataset import *

from src.UNet import *
from src.Ensemble import *
from src.ModelManager import *
from src.Channels import *
from src.class_counts_weights import *
from src.plot import *
from src.train import *
from src.metrics import *

from src.args_parser import *

""" 
prep.py has to be run at least once before running this.

# HOW TO RUN:

# new 
./attached_main --dataset=DFC18Dataset --model=UNet --count_models=7 --channel_mod=random_channel_drop
./attached_main --dataset=DFC18Dataset --model=UNet --count_models=7 --channel_mod=

# reload (single or "[name,name,...]" list):
./attached_main --reload="02_08_26-random_subsets_PCA-7m-DFC18"
./attached_main --reload=""
./attached_main --reload="[02_08_26-random_subsets_PCA-7m-DFC18,03_08_26-random_channel_drop-7m-DFC18]"
./attached_main --reload="[,]"

# local, directly python
python main.py --reload=""
python main.py --reload="[,]"
python main.py --reload="" --just_stats=True

# train only part of models
,/attached_main --reload="20_08_26-powerset_of_all_modalities-15m-DFC18" --train_model_nums=[13,14,15]
,/attached_main --reload="" --train_model_nums=[]
"""

# Datasets: --dataset
#  DFC18Dataset
#  MDAS_Dataset

# Model: --model
#  default: UNet = UNet_standard_4layers
#  UNet_big_5layers
#  UNet_standard_4layers
#  UNet_small_3layers

# CHANNEL SELECTION / OPTIMIZATION: --channel_mod
# - Functions of Channel_Modulation
# No Modulation:
#   identity()
# With Fixed Ensemble Size:
#   powerset_of_all_modalities()
# Variable Ensemble Size:
#   Random subsets:
#       random_channel_drop()
#       random_drop_percentage_of_modality()
#   Optimized subsets:
#       random_subsets_PCA()
#       random_linear_combination()
#       random_classes_simple_LDA()

# Ensemble Paths:
#  Ensembles are saved to   ensembles/<ENSEMBLE_NAME>
#    Ensemble metadata in   ensembles/<ENSEMBLE_NAME>/meta.pth
#      Includes: 
#    Models:                ensembles/<ENSEMBLE_NAME>/models/
#      model themselves     ensembles/<ENSEMBLE_NAME>/models/<MODEL_NUM>.pth
#      train stats          ensembles/<ENSEMBLE_NAME>/models/<MODEL_NUM>_train_stats.pth
#      test stats           ensembles/<ENSEMBLE_NAME>/models/<MODEL_NUM>_test_stats.pth


def get_ensembles(args):
    """ Yield the ensemble(s) to run: reloaded ones if --reload was given, else a freshly constructed one """
    if args.reload is not None:
        for name in args.reload:
            yield Ensemble.load_from_file(name, just_stats=args.just_stats)

    else:
        yield Ensemble(args.dataset, args.model, args.channel_mod,
                        count_models=args.count_models, modulation_opts=args.modulation_opts)

def load_ALL_ensembles(yielded_ensembles):
    return [e for e in yielded_ensembles]

if __name__ == "__main__":

    args = parse_args()
    ensembles = get_ensembles(args)
    
    # Visualize multiple ensemble
    visualize_test_ensembles(load_ALL_ensembles(ensembles), 
        num_top_models=None, 
        save=True, 
        show_class_dice=False, 
        show_average_dice=True,
        show_title=False,
        show_legend=False) 
        #, show_val=True

    """ visualize_ensemble_variance(load_ALL_ensembles(ensembles),
        show_title=False) """

    for ensemble1 in ensembles:
        
        """ # if model nums given pass them
        ensemble1.train(model_nums=args.train_model_nums) 
        
        ensemble1.test() """
    
        # Visualize single ensemble
        
        """ ensemble1.show_predict_dataset_index(0) """

        """ ensemble1.test_predict_full_dataset() """

        """ show_full_area_test(ensemble1, show_models=False, 
            save=True, show_prediction=True, show_entropy=True,
            show_title=False) """

        """ visualize_ensemble_modulations(ensemble1, save=True, 
            force_cells_quadratic=False, 
            model_indices=[0,1],
            show_title=False) 
        # , num_plots=3
        # , [12,13,14] """

        """ visualize_train_ensemble(ensemble1, save=False, 
            model_indices=[0,1,2], highlight_best=False,
            show_title=False) """

        
        """ # COMPARE PCA PARAMS WITH RESULTS
        params_data = populate_pca_params_with_results(ensemble_names, ensembles_to_compare)

        visualize_pca_parameter_correlation(params_data, save=True) """

        pass


print("END")



