from src.base import *

# CALC CLASS IMBALANCE & WEIGHTS

def get_class_counts(dataloader, classes, device):
    class_counts = torch.zeros(classes, device=device)

    for data in tqdm(dataloader):
        mask = data[1].to(device)

        # count "bins" (different values)
        mask_flat = mask.flatten().long()

        counts = torch.bincount(mask_flat, minlength=classes)
        class_counts += counts

    class_counts[0] = 0
    print("\nclass_counts = torch.", class_counts)
    return class_counts


def get_class_weigths_inv_freq(counts):
    '''
    Inverse frequency weights
    '''
    weights = 1.0 / (counts + 1e-6)

    # clamp
    clamp_max = 4.0
    clamp_min = 0.25

    weights = torch.clamp(weights, min=clamp_min, max=clamp_max)

    weights[0] = 0
    # Normalize weights to sum to 1
    return weights / weights.sum()


def get_class_weigths_mean_count_by_count(counts):
    '''
    Scale weights by mean_count / count so the average weight is ~1
    clamp extremes ?
    '''

    # avoid div-by-zero
    mean_count = counts[counts > 0].mean()
    weights = mean_count / counts

    # clamp
    clamp_max = 6.0
    clamp_min = 0.25

    weights = torch.clamp(weights, min=clamp_min, max=clamp_max)

    weights[0] = 0.0

    # normalize
    return weights / weights.sum()
