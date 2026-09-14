from src.base import *

from src.DFC18_Dataset import *
from src.MDAS_Dataset import *

# SHANNON ENTROPY

def shannon_entropy(probs, dim=0, epsilon=1e-12):
    '''
    H = -sum_c p_c * log(p_c)

    Low where one class dominates (confident), high where probability is spread evenly across classes (uncertain). 
    Max approaches log(num_classes), min approaches `-inf`. 

    probs: class probabilities p_c, have to sum to 1 along 'dim'
    '''
    
    # Avoid log(0) -> -inf -> nan
    probs = probs.clamp_min(epsilon)

    return -(probs * torch.log(probs)).sum(dim=dim)


# DICE COEFF METRIC

def dice_coefficient(prediction, target, epsilon=1e-07):
    '''
    ( 2 * |A ∩ B| ) / ( |A| + |B| )
    '''
    intersection = abs(torch.sum(prediction * target)) # TP
    union = abs(torch.sum(prediction) + torch.sum(target))

    return (2.0 * intersection + epsilon) / (union + epsilon)


def dice_coefficient_multiclass(prediction, target, num_classes, already_flattened=False):
    ''' On single instances of prediction, target, NOT BATCHED

    -> [ dice_class1, dice_class2, ... ] '''


    if not already_flattened:
        # prediction is [NUM_CLASSES, SIZE, SIZE]
        # prediction = [class1: [pixel1: prob_this_class, pixel2: prob_this_class, ...],
        #               class2: [pixel1: prob_this_class, pixel2: prob_this_class, ...], ...]

        # eg. shapes compatible: [21, 512, 512] & [512, 512]

        if prediction.shape[1] != target.shape[0] or prediction.shape[2] != target.shape[1]:
            raise ValueError("Shape incompatible: prediction and target must compatible shapes. prediction:", prediction.shape, " target:", target.shape)

        # Prediction to class indices with argmax
        predicted_classes = torch.argmax(prediction, dim=0) # collapse dim 0 of classes
        # [SIZE, SIZE]
        # -> e.g. [pixel1: class14, pixel2: class4, ...]

        # To 1D
        predicted_flat = predicted_classes.flatten()
        target_flat = target.flatten()

    else:
        predicted_flat = prediction
        target_flat = target

    if predicted_flat.shape != target_flat.shape:
        raise ValueError("Shape incompatible: predicted_flat and target_flat must be same shapes. predicted_flat:", predicted_flat.shape, " target_flat:", target_flat.shape)

    # valid: Ignore unclassified anywhere when comparing
    valid_pixels_mask = (target_flat != 0) # mask [True, False, ... ]

    # pred values in pixels that have GT
    # <- only positives: TP, FP
    # invert mask with ~ to only keep valid
    predicted_valid = predicted_flat
    predicted_valid[~valid_pixels_mask] = 0


    dice_scores = np.zeros(num_classes - 1)

    for class_id in range(1, num_classes): # skip unclassified
        # Binary masks for curr class
        predicted_binary = (predicted_valid == class_id).float()
        target_binary = (target_flat == class_id).float()

        dice = dice_coefficient(predicted_binary, target_binary)
        dice_scores[class_id - 1] = dice

    return dice_scores


def get_correct_pixels_count(predicted_flat, target_flat, num_classes):
    """ 
    pred & target have to be argmax'd and flattened
    in shapes each: (SIZE^2)
    """

    # valid: Ignore unclassified anywhere when comparing
    valid_pixels_mask = (target_flat != 0)
    predicted_flat_valid = predicted_flat[valid_pixels_mask]
    target_flat_valid = target_flat[valid_pixels_mask]

    # For OA
    correct_predictions = (predicted_flat_valid == target_flat_valid).sum().item()
    total_valid_pixels = valid_pixels_mask.sum().item()


    # For AA
    correct_per_class = []
    total_per_class = []

    for class_id in range(1, num_classes): # Skip 0
        class_mask = (target_flat_valid == class_id)
        correct_in_class = (predicted_flat_valid[class_mask] == class_id).sum().item()

        total_in_class = class_mask.sum().item()

        correct_per_class.append(correct_in_class)
        total_per_class.append(total_in_class)

    # Convert lists to torch tensors
    correct_per_class = torch.tensor(correct_per_class, dtype=torch.float32)
    total_per_class = torch.tensor(total_per_class, dtype=torch.float32)

    return correct_predictions, total_valid_pixels, correct_per_class, total_per_class



# FOR OA and AA
""" def get_correct_pixels_count_batched(pred_batch, target_batch):
    for target, pred in zip(target_batch, pred_batch):
        num_samples += 1

        curr_correct, curr_total, curr_correct_per_class, curr_total_per_class = get_correct_pixels_count(pred, target)

        correctly_labeled_pixels += curr_correct
        total_labeled_pixels += curr_total

        correctly_labeled_pixels_per_class += curr_correct_per_class
        total_labeled_pixels_per_class += curr_total_per_class
 """
    