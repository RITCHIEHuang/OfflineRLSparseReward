from collections import Counter

import numpy as np

from scipy.ndimage import gaussian_filter1d, convolve1d
from scipy.signal.windows import triang


def get_lds_kernel_window(kernel, ks, sigma):
    assert kernel in ["gaussian", "triang", "laplace"]
    half_ks = (ks - 1) // 2
    if kernel == "gaussian":
        base_kernel = [0.0] * half_ks + [1.0] + [0.0] * half_ks
        kernel_window = gaussian_filter1d(base_kernel, sigma=sigma) / max(
            gaussian_filter1d(base_kernel, sigma=sigma)
        )
    elif kernel == "triang":
        kernel_window = triang(ks)
    else:
        laplace = lambda x: np.exp(-abs(x) / sigma) / (2.0 * sigma)
        kernel_window = list(
            map(laplace, np.arange(-half_ks, half_ks + 1))
        ) / max(map(laplace, np.arange(-half_ks, half_ks + 1)))

    return kernel_window


def get_bin_idx(label):
    # TODO
    return 0


def process_lds_strategy(labels):
    bin_index_per_label = [get_bin_idx(label) for label in labels]

    # calculate empirical (original) label distribution: [Nb,]
    # "Nb" is the number of bins
    Nb = max(bin_index_per_label) + 1
    num_samples_of_bins = dict(Counter(bin_index_per_label))
    emp_label_dist = [num_samples_of_bins.get(i, 0) for i in Nb]

    # lds_kernel_window: [ks,], here for example, we use gaussian, ks=5, sigma=2
    lds_kernel_window = get_lds_kernel_window(kernel="gaussian", ks=5, sigma=2)
    # calculate effective label distribution: [Nb,]
    eff_label_dist = convolve1d(
        np.array(emp_label_dist), weights=lds_kernel_window, mode="constant"
    )

